/**
 * Offline contracts against the installed Pi runtime (no Claude/auth requests).
 * Internal imports intentionally fail on incompatible Pi changes rather than
 * silently skipping pricing or cache-warming coverage.
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { calculateCost } from "@earendil-works/pi-ai";
import { buildModels } from "../src/models.js";

const piEntry = import.meta.resolve("@earendil-works/pi-coding-agent");
const { ModelRuntime } = await import(new URL("./core/model-runtime.js", piEntry));
const { ModelRegistry } = await import(new URL("./core/model-registry.js", piEntry));
const { AuthStorage } = await import(new URL("./core/auth-storage.js", piEntry));
const { CacheWarmer } = await import(new URL("./core/cache-warmer.js", piEntry));

// Integer dollars per token make the independent expectations easy to audit.
const cost = {
	input: 1_000_000, output: 2_000_000, cacheRead: 3_000_000, cacheWrite: 4_000_000,
	tiers: [{ inputTokensAbove: 100, input: 5_000_000, output: 6_000_000, cacheRead: 7_000_000, cacheWrite: 8_000_000 }],
};
const catalogModel = {
	id: "claude-haiku-4-5", name: "Contract model", reasoning: true,
	input: ["text"], contextWindow: 200000, maxTokens: 8000,
	cost, promptCache: { short: 20, long: 3600 },
};

async function registerModel() {
	const runtime = await ModelRuntime.create({
		credentials: AuthStorage.inMemory(), modelsPath: null,
		allowModelNetwork: false, refreshOnCreate: false,
	});
	const registry = new ModelRegistry(runtime);
	registry.registerProvider("claude-bridge", {
		api: "claude-bridge", apiKey: "offline-contract", baseUrl: "https://example.invalid",
		models: buildModels([catalogModel]),
		streamSimple: () => { throw new Error("pricing contracts must not send requests"); },
	});
	const model = registry.find("claude-bridge", catalogModel.id);
	assert.ok(model, "real provider registration exposes the projected model");
	return model;
}

const emptyUsage = () => ({
	input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
});

function warmingHarness(t) {
	t.mock.timers.enable({ apis: ["Date", "setTimeout"], now: 1_000_000 });
	const calls = [];
	const entries = [];
	const warmer = new CacheWarmer({
		streamSimple: (model, context, options) => {
			calls.push({ model, context, options });
			return { result: async () => ({
				provider: model.provider, model: model.id, stopReason: "stop", usage: emptyUsage(),
			}) };
		},
	}, {
		getBranch: () => [{ type: "message", message: { role: "assistant", usage: { ...emptyUsage(), input: 100_000 } } }],
		appendUsage: (...args) => { entries.push(args); return { type: "usage" }; },
	}, () => "streaming");
	t.after(() => warmer.cancel());
	return { warmer, calls, entries };
}

describe("installed Pi model contracts", () => {
	it("preserves cost tiers through registerProvider and applies request-wide thresholds", async () => {
		const model = await registerModel();
		assert.deepEqual(model.cost, cost);
		assert.equal(model.promptCache, undefined);
		// The threshold includes input + cacheRead + cacheWrite; output is excluded.
		// Exactly 100 prompt tokens still uses base rates; only >100 uses the tier.
		for (const [input, expected] of [
			[49, { input: 49, output: 14, cacheRead: 60, cacheWrite: 120, total: 243 }],
			[50, { input: 50, output: 14, cacheRead: 60, cacheWrite: 120, total: 244 }],
			[51, { input: 255, output: 42, cacheRead: 140, cacheWrite: 240, total: 677 }],
		]) {
			const usage = { ...emptyUsage(), input, output: 7, cacheRead: 20, cacheWrite: 30 };
			assert.deepEqual(calculateCost(model, usage), expected, `${input + 50} prompt tokens`);
		}
	});

	it("cache warmer replays a declared cache lifetime with a one-token cap", async (t) => {
		const registered = await registerModel();
		// Positive control: opt into a short lifetime only for this offline spy.
		const model = {
			...registered, promptCache: catalogModel.promptCache,
			cost: { input: 1, output: 5, cacheRead: 0.1, cacheWrite: 1.25 },
		};
		const context = { messages: [{ role: "user", content: "offline probe", timestamp: 0 }] };
		const options = { cacheRetention: "short", maxTokens: 8000, maxRetries: 3 };
		const { warmer, calls, entries } = warmingHarness(t);
		warmer.start({ model, context, options }, () => true);
		assert.equal(warmer.status.state, "scheduled");
		assert.equal(warmer.status.decision.action, "warm");
		t.mock.timers.tick(10_000);
		// Drain the async decision and result without advancing another timer.
		await new Promise(setImmediate);
		assert.equal(calls.length, 1);
		assert.equal(calls[0].model, model);
		assert.equal(calls[0].context, context);
		assert.equal(calls[0].options.maxTokens, 1);
		assert.equal(calls[0].options.maxRetries, 0);
		assert.equal(calls[0].options.cacheRetention, "short");
		assert.ok(calls[0].options.signal instanceof AbortSignal);
		assert.equal(options.maxTokens, 8000, "replay does not mutate original options");
		assert.equal(entries.length, 1);
		assert.equal(entries[0][0], "cache_warm");
	});

	it("projected models do not schedule or send cache-warming requests", async (t) => {
		const model = await registerModel();
		const { warmer, calls, entries } = warmingHarness(t);
		for (const retention of ["short", "long"]) {
			warmer.start({ model, context: { messages: [] }, options: { cacheRetention: retention } }, () => true);
			assert.deepEqual(warmer.status, { state: "inactive", reason: "cache lifetime unavailable" });
			t.mock.timers.tick(3_600_000);
			await new Promise(setImmediate);
			assert.equal(calls.length, 0);
			assert.equal(entries.length, 0);
		}
	});
});
