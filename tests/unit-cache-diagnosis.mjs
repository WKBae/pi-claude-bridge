/**
 * Tests for src/cache-diagnosis.ts and its wiring in consumeQuery.
 * Pins: the API's cache_miss_reason is kept on the turn's assistant message, the latest
 * request's diagnosis replaces an earlier one (and clears it when the latest has none),
 * other diagnostics survive; a reason is shown only for completed responses whose missed
 * tokens reach pi's 20k notice floor or whose missed cost reaches its $0.10 floor.
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { QueryContext } from "../src/query-state.js";
import {
	CACHE_MISS_DIAGNOSTIC, formatCacheMissReason, noticeableCacheMiss, recordCacheMissReason,
} from "../src/cache-diagnosis.js";

const { __test } = await import("../src/index.js");

const modelChanged = { type: "model_changed", cache_missed_input_tokens: 22330 };
const usage = (input, cacheWrite, cost = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 }, cacheRead = 0) =>
	({ input, output: 1, cacheRead, cacheWrite, totalTokens: 0, cost });
const message = (diagnostics, extra = {}) => ({ role: "assistant", stopReason: "stop", usage: usage(0, 0), diagnostics, ...extra });
const recorded = (details) => ({ type: CACHE_MISS_DIAGNOSTIC, timestamp: 1, details });

describe("recordCacheMissReason", () => {
	it("records the reason from the API message", () => {
		const output = {};
		recordCacheMissReason(output, { diagnostics: { cache_miss_reason: modelChanged } });
		assert.equal(output.diagnostics.length, 1);
		assert.equal(output.diagnostics[0].type, CACHE_MISS_DIAGNOSTIC);
		assert.deepEqual(output.diagnostics[0].details, modelChanged);
	});

	it("records nothing for a hit, a pending diagnosis, or an undiagnosed request", () => {
		for (const apiMessage of [{ diagnostics: null }, { diagnostics: { cache_miss_reason: null } }, {}, undefined]) {
			const output = {};
			recordCacheMissReason(output, apiMessage);
			assert.equal(output.diagnostics, undefined);
		}
	});

	it("keeps only the latest request's reason and leaves other diagnostics alone", () => {
		const other = { type: "anthropic_input_transformations", timestamp: 1 };
		const output = { diagnostics: [other, recorded(modelChanged)] };
		recordCacheMissReason(output, { diagnostics: { cache_miss_reason: { type: "system_changed" } } });
		assert.deepEqual(output.diagnostics.map((d) => d.details?.type ?? d.type), ["anthropic_input_transformations", "system_changed"]);
		recordCacheMissReason(output, { diagnostics: null });
		assert.deepEqual(output.diagnostics, [other]);
	});
});

describe("noticeableCacheMiss", () => {
	it("returns a reason whose missed tokens reach pi's notice floor", () => {
		assert.deepEqual(noticeableCacheMiss(message([recorded(modelChanged)])), modelChanged);
		assert.equal(noticeableCacheMiss(message([recorded({ type: "model_changed", cache_missed_input_tokens: 19_999 })])), undefined);
	});

	it("falls back to the uncached prompt tokens when the API gives no count", () => {
		const reason = { type: "previous_message_not_found" };
		assert.deepEqual(noticeableCacheMiss(message([recorded(reason)], { usage: usage(5_000, 15_000) })), reason);
		assert.equal(noticeableCacheMiss(message([recorded(reason)], { usage: usage(5_000, 14_999) })), undefined);
	});

	it("returns a smaller miss whose cost over the cache-read rate reaches pi's $0.10 floor", () => {
		// 10k missed tokens paid at $10/M against a $0.5/M read rate cost $0.095 extra; at $11/M, $0.105.
		const reason = { type: "system_changed", cache_missed_input_tokens: 10_000 };
		const priced = (paidPerMillion) => usage(10_000, 0,
			{ input: paidPerMillion / 100, output: 0, cacheRead: 0.5 / 100, cacheWrite: 0, total: 0 }, 10_000);
		assert.equal(noticeableCacheMiss(message([recorded(reason)], { usage: priced(10) })), undefined);
		assert.deepEqual(noticeableCacheMiss(message([recorded(reason)], { usage: priced(11) })), reason);
	});

	it("skips aborted and failed responses and messages without a reason", () => {
		assert.equal(noticeableCacheMiss(message([recorded(modelChanged)], { stopReason: "aborted" })), undefined);
		assert.equal(noticeableCacheMiss(message([recorded(modelChanged)], { stopReason: "error" })), undefined);
		assert.equal(noticeableCacheMiss(message(undefined)), undefined);
	});
});

describe("formatCacheMissReason", () => {
	it("names the reason and the missed tokens", () => {
		assert.equal(formatCacheMissReason(modelChanged), "Cache miss reason: model changed (22,330 tokens)");
		assert.equal(formatCacheMissReason({ type: "unavailable" }), "Cache miss reason: unavailable");
	});
});

describe("consumeQuery", () => {
	const fakeModel = { api: "anthropic-messages", provider: "anthropic", id: "test-model",
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
	const fakeStream = () => {
		const events = [];
		return { events, push: (e) => events.push(e), end: () => events.push({ type: "end" }) };
	};

	it("keeps message_start's diagnosis on the turn's assistant message", async () => {
		const c = new QueryContext();
		c.currentPiStream = fakeStream();
		c.resetTurnState(fakeModel);
		async function* gen() {
			yield { type: "stream_event", event: { type: "message_start", message: {
				id: "msg_1", usage: { input_tokens: 3, cache_read_input_tokens: 0, cache_creation_input_tokens: 0, output_tokens: 1 },
				diagnostics: { cache_miss_reason: modelChanged },
			} } };
		}
		await __test.consumeQuery(gen(), new Map(), fakeModel, () => false, c);
		assert.deepEqual(c.turnOutput.diagnostics?.map((d) => d.details), [modelChanged]);
	});
});
