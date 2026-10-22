/**
 * An extension calling the model outside any pi agent turn sends its own system prompt,
 * so projection has nothing to forward and the live path used to fail the call with
 * "prompt-capture: no capture". A call may ask one question or be one step of the
 * extension's own tool loop. Both now run in an isolated
 * CC process, and only when the call cannot be a pi turn: no session id the bridge's hooks
 * reported, a prompt no capture accounts for, and no tool result a live query is parked on.
 *
 * The live path must keep every other shape, so each guard is asserted from the side
 * that must NOT take the isolated route.
 */
import { afterEach, beforeEach, describe, it } from "node:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// The refusal path writes a diag entry into pi's agent dir, resolved at import time.
const tmpRoot = mkdtempSync(join(tmpdir(), "claude-bridge-one-shot-"));
process.env.PI_CODING_AGENT_DIR = join(tmpRoot, "agent");
// The tool step writes its throwaway session under CC's config dir.
process.env.CLAUDE_CONFIG_DIR = join(tmpRoot, "claude");
delete process.env.CLAUDE_BRIDGE_DEBUG;
process.on("exit", () => rmSync(tmpRoot, { recursive: true, force: true }));

const { default: activate, __test } = await import("../src/index.js");
const { PromptCaptures, PI_PREAMBLE } = await import("../src/prompt-capture.js");
const { QueryContext } = await import("../src/query-state.js");
const { openSession } = await import("cc-session-io");

let providerConfig;
const handlers = {};
activate({
	on: (name, fn) => { (handlers[name] ??= []).push(fn); },
	registerProvider: (_name, config) => { providerConfig = config; },
	registerTool: () => {},
});
const model = providerConfig.models[0];
const streamSimple = providerConfig.streamSimple;
const fire = (name, sessionId) => {
	for (const fn of handlers[name]) fn({}, { sessionManager: { getSessionId: () => sessionId }, getSystemPrompt: () => undefined });
};

// An extension's own prompt for a single question: the point is a prompt no capture
// recorded, and a recognisable one makes a regression obvious in the failure text.
const ONE_SHOT_SYSTEM = "You are a senior engineer giving a brief second opinion to an autonomous coding agent.";
const user = (text) => ({ role: "user", content: [{ type: "text", text }], timestamp: 0 });
const PNG_BASE64 =
	"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg==";
const imageTurn = (text) => ({
	role: "user",
	content: [
		{ type: "text", text },
		{ type: "image", data: PNG_BASE64, mimeType: "image/png" },
	],
	timestamp: 0,
});

/** Captures the SDK request and answers it, the way tests/unit-isolated-effort does. */
function captureQuery(calls) {
	return ({ prompt, options }) => {
		calls.push({ prompt, options });
		async function* gen() {
			yield { type: "result", subtype: "success", is_error: false, result: "advice" };
		}
		const q = gen();
		q.interrupt = async () => {};
		q.close = () => {};
		return q;
	};
}

/** The prompt as a one-message iterable hands it to CC. */
async function collectPrompt(prompt) {
	if (typeof prompt === "string") return prompt;
	const messages = [];
	for await (const message of prompt) messages.push(message);
	return messages;
}

let calls;
beforeEach(() => {
	calls = [];
	__test.setQuery(captureQuery(calls));
});
afterEach(() => __test.setQuery(null));

const call = (context, options) => streamSimple(model, context, options);

describe("prompt capture canResolve", () => {
	it("agrees with resolveOrDerive on every route it answers", () => {
		const promptCaptures = __test.promptCaptures;
		const known = "a recorded prompt for the canResolve comparison";
		promptCaptures.record(known, { custom: known, contextFiles: [], skills: [] });

		assert.equal(promptCaptures.canResolve(known), true, "an exact key resolves");
		assert.ok(promptCaptures.resolveOrDerive(known), "and resolveOrDerive returns it");

		const wrapped = `wrapper text\n\n${known}\n\ntrailing wrapper text`;
		assert.equal(promptCaptures.canResolve(wrapped), true, "an embedding resolves");
		assert.ok(promptCaptures.resolveOrDerive(wrapped), "and resolveOrDerive derives it");

		// The third route: the lookup key was evicted while an inheritance edge kept the
		// node alive, so resolveOrDerive revives it. canResolve must see that too, even
		// though no key equals the prompt at that moment.
		const bounded = new PromptCaptures(2);
		const parent = "parent prompt kept alive by its child";
		bounded.record(parent, { custom: parent, contextFiles: [], skills: [] });
		// The child embeds the parent verbatim, which is what holds the parent node.
		bounded.record("child prompt", { custom: `${parent}\nchild only text`, contextFiles: [], skills: [] });
		bounded.record("unrelated", { custom: "unrelated", contextFiles: [], skills: [] });
		assert.equal(bounded.resolve(parent), undefined, "precondition: the key is gone");
		assert.equal(bounded.canResolve(parent), true, "a reachable capture still resolves");
		assert.ok(bounded.resolveOrDerive(parent));

		const unknown = "a prompt composed outside pi's pipeline by a host";
		assert.equal(promptCaptures.canResolve(unknown), false, "an unrelated prompt does not");
		assert.throws(() => promptCaptures.resolveOrDerive(unknown), /no capture/);
	});
});

describe("an out-of-band one-shot", () => {
	it("runs in an isolated CC process instead of failing the capture lookup", async () => {
		// Seed a live shared session: without one, "untouched" would hold for any
		// implementation, including one that syncs. The one-shot must leave it alone.
		__test.setSharedSession(null, { sessionId: "cc-session-kept", cursor: 3, cwd: "/somewhere" });
		const result = await call(
			{ systemPrompt: ONE_SHOT_SYSTEM, messages: [user("Review the draft.")] },
			{ reasoning: "low" },
		).result();

		assert.equal(result.stopReason, "stop");
		assert.equal(result.errorMessage, undefined);
		assert.equal(result.content.map((part) => part.text).join(""), "advice");
		assert.equal(calls.length, 1);

		const { prompt, options } = calls[0];
		assert.equal(options.systemPrompt, ONE_SHOT_SYSTEM, "the extension's own prompt reaches CC");
		assert.equal(options.persistSession, false, "no CC session is written");
		assert.deepEqual(options.tools, [], "no tools are served");
		assert.equal(options.maxTurns, 1);
		assert.equal(options.effort, "low", "the caller's reasoning level carries over");
		assert.equal(await collectPrompt(prompt), "Review the draft.");
		assert.deepEqual(
			__test.getSharedSession(null),
			{ sessionId: "cc-session-kept", cursor: 3, cwd: "/somewhere" },
			"the shared session is untouched",
		);
		__test.setSharedSession(null, null);
		assert.equal(__test.activeQueryContexts.size, 0, "no live query was opened");
	});

	it("carries a follow-up's consecutive user messages", async () => {
		const result = await call(
			{ systemPrompt: ONE_SHOT_SYSTEM, messages: [user("Review the draft."), user("Follow-up focus:\nAnd the tests?")] },
			{},
		).result();

		assert.equal(result.stopReason, "stop");
		assert.equal(
			await collectPrompt(calls[0].prompt),
			"Review the draft.\nFollow-up focus:\nAnd the tests?",
			"a string prompt joins the turn, as the summarizer path does",
		);
	});

	it("sends an image with no caption, which the text extraction alone cannot express", async () => {
		// A screenshot may be the whole question; extractUserPrompt drops
		// the text-less message, so only the blocks keep the run alive.
		const result = await call(
			{ systemPrompt: ONE_SHOT_SYSTEM, messages: [{ role: "user", content: [{ type: "image", data: PNG_BASE64, mimeType: "image/png" }], timestamp: 0 }] },
			{},
		).result();

		assert.equal(result.stopReason, "stop");
		const messages = await collectPrompt(calls[0].prompt);
		assert.equal(messages.length, 1);
		assert.deepEqual(messages[0].message.content, [
			{ type: "image", source: { type: "base64", media_type: "image/png", data: PNG_BASE64 } },
		]);
	});

	it("collects images from every message of a follow-up turn", async () => {
		// A follow-up re-sends the original question, so a screenshot attached there is
		// in an earlier message of the same user turn, not the last one.
		const result = await call(
			{ systemPrompt: ONE_SHOT_SYSTEM, messages: [imageTurn("What color is this image?"), user("Follow-up focus:\nAnd its size?")] },
			{},
		).result();

		assert.equal(result.stopReason, "stop");
		const [message] = await collectPrompt(calls[0].prompt);
		assert.deepEqual(message.message.content.map((block) => block.type), ["text", "image", "text"]);
	});

	it("sends attached images as blocks, which a string prompt cannot carry", async () => {
		const result = await call(
			{ systemPrompt: ONE_SHOT_SYSTEM, messages: [imageTurn("What color is this image?")] },
			{},
		).result();

		assert.equal(result.stopReason, "stop");
		const messages = await collectPrompt(calls[0].prompt);
		assert.equal(messages.length, 1, "one user message, streamed");
		assert.equal(messages[0].type, "user");
		assert.equal(messages[0].parent_tool_use_id, null);
		assert.equal(messages[0].message.role, "user");
		assert.deepEqual(messages[0].message.content, [
			{ type: "text", text: "What color is this image?" },
			{ type: "image", source: { type: "base64", media_type: "image/png", data: PNG_BASE64 } },
		]);
	});
});

describe("shapes that must stay on the live path", () => {
	/** The live path's own refusal: no query starts and the turn ends on the stream. */
	const assertLivePathRefused = async (context, options) => {
		const result = await call(context, options).result();
		assert.equal(result.stopReason, "error");
		assert.match(result.errorMessage, /prompt-capture: no capture/);
		assert.equal(calls.length, 0, "the isolated path must not run");
		assert.equal(__test.activeQueryContexts.size, 0);
	};

	it("keeps a turn of a session the bridge has seen on the live path", async () => {
		fire("turn_start", "pi-session");
		await assertLivePathRefused(
			{ systemPrompt: ONE_SHOT_SYSTEM, messages: [user("hi")] },
			{ sessionId: "pi-session" },
		);
	});

	it("keeps a session seen at turn_start live until session_shutdown", () => {
		const context = { systemPrompt: ONE_SHOT_SYSTEM, messages: [user("hi")] };
		fire("turn_start", "seen-then-closed");
		assert.equal(__test.isOutOfBandCall(context, { sessionId: "seen-then-closed" }), false);
		fire("session_shutdown", "seen-then-closed");
		assert.equal(__test.isOutOfBandCall(context, { sessionId: "seen-then-closed" }), true, "a closed session's id is no longer a pi conversation");
	});

	it("keeps a tool result a live query is parked on with that query", () => {
		const parked = new QueryContext();
		parked.turnToolCallIds = ["call_parked"];
		__test.activeQueryContexts.add(parked);
		try {
			const context = {
				systemPrompt: ONE_SHOT_SYSTEM,
				messages: [user("hi"), assistantCalls([toolCall("call_parked")]), toolResult("call_parked", "out")],
				tools: [INSPECT_TOOL],
			};
			assert.equal(__test.isOutOfBandCall(context, undefined), false, "the parked query delivers its own result");
			parked.turnToolCallIds = [];
			assert.equal(__test.isOutOfBandCall(context, undefined), true, "a result nobody is parked on is the caller's");
		} finally {
			__test.activeQueryContexts.delete(parked);
		}
	});

	it("refuses an empty one-shot instead of sending an empty user message", async () => {
		const result = await call(
			{ systemPrompt: ONE_SHOT_SYSTEM, messages: [user("")] },
			{},
		).result();

		assert.equal(result.stopReason, "error");
		assert.match(result.errorMessage, /without a user prompt/);
		assert.equal(calls.length, 0, "no query starts for an empty turn");
	});

	it("predicts the live path's decision without touching the registry", () => {
		// A local registry, not the shared one: recording into sharedPromptCaptures leaks
		// into every later test in this file, and asks the routing check about a prompt
		// the other tests rely on being unknown.
		const diag = [];
		const captures = new PromptCaptures(2, (d) => diag.push(d));
		const recorded = "a prompt this registry recorded";
		captures.record(recorded, { custom: recorded, contextFiles: [], skills: [] });
		captures.record("another recorded prompt", { custom: "another recorded prompt", contextFiles: [], skills: [] });

		// Recency is observable through eviction: at size 2, one more write drops exactly
		// the least recently used key. If canResolve touched its match, that would be the
		// other key — which is what an eviction of live policy would look like.
		const other = "another recorded prompt";
		assert.equal(captures.canResolve(recorded), true, "a recorded prompt resolves");
		captures.record("one more", { custom: "one more", contextFiles: [], skills: [] });
		// `recorded` was recorded first, so it is the one an untouched registry drops. If
		// canResolve had touched it, it would have moved to the newest end and `other`
		// would be the casualty instead.
		assert.equal(captures.resolve(recorded), undefined, "the oldest key was evicted");
		assert.ok(captures.resolve(other), "the key canResolve never looked at survived");

		assert.equal(captures.canResolve("a prompt nobody recorded"), false);
		assert.deepEqual(diag, [], "a routing check writes no diagnostic");
	});

	it("declines a prompt that carries pi's own harness, which the one-shot route must not forward", async () => {
		// The routing check must not become a way around the sendability guard: this text
		// is exactly what assertSendablePrompt refuses on the live path, and forwarding it
		// to CC is a 400 or a charge to the user's extra usage.
		const context = {
			systemPrompt: `${PI_PREAMBLE}, a coding agent harness. You help users.`,
			messages: [user("hi")],
		};
		assert.equal(__test.isOutOfBandCall(context, undefined), false, "pi's harness keeps the live refusal");
		assert.equal(__test.isOutOfBandCall(context, { sessionId: "f3b1c2d4-routing-id" }), false, "so does an unseen routing id");

		const result = await call(context, {}).result();
		assert.equal(result.stopReason, "error");
		assert.match(result.errorMessage, /prompt-capture: no capture/);
		assert.equal(calls.length, 0, "no query starts for a prompt the guard refuses");
	});

	it("declines the phrase pair Claude Code's subscription gate rejects", async () => {
		const context = {
			systemPrompt: "Read docs/custom-provider.md and docs/packages.md before answering.",
			messages: [user("hi")],
		};
		assert.equal(__test.isOutOfBandCall(context, undefined), false);
		assert.equal(calls.length, 0);
	});

	it("runs a call carrying a routing id no hook reported, as pi-btw sends", async () => {
		const result = await call(
			{ systemPrompt: ONE_SHOT_SYSTEM, messages: [user("Side question.")] },
			{ sessionId: "5d0c9a7e-2f1b-4c3a-9e8d-routing" },
		).result();

		assert.equal(result.stopReason, "stop", result.errorMessage);
		assert.equal(calls.length, 1);
		assert.equal(calls[0].options.persistSession, false, "no CC session is written");
	});

	it("reports the usage its answer cost instead of zero", async () => {
		__test.setQuery(() => {
			async function* gen() {
				yield { type: "result", subtype: "success", is_error: false, result: "advice",
					usage: { input_tokens: 1200, output_tokens: 300, cache_read_input_tokens: 800 } };
			}
			const q = gen();
			q.interrupt = async () => {};
			q.close = () => {};
			return q;
		});

		const result = await call({ systemPrompt: ONE_SHOT_SYSTEM, messages: [user("Review.")] }, {}).result();
		assert.equal(result.stopReason, "stop");
		assert.equal(result.usage.input, 1200);
		assert.equal(result.usage.output, 300);
		assert.equal(result.usage.cacheRead, 800);
		assert.equal(result.usage.totalTokens, 2300);
	});
});

// An extension running its own tool loop: its own prompt, the tools it offers, and a
// loop that calls back with the tool results it gathered.
const TOOL_LOOP_SYSTEM = "You check a coding agent's tool call before it runs, using the tools you are given.";
const INSPECT_TOOL = { name: "inspect", description: "Read a file", parameters: { type: "object", properties: { path: { type: "string" } } } };
const ZERO_USAGE = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
function toolCall(id, args = { path: "README.md" }) {
	return { type: "toolCall", id, name: "inspect", arguments: args };
}
function assistantCalls(content) {
	return { role: "assistant", content, api: "claude-bridge", provider: "claude-bridge", model: model.id,
		usage: ZERO_USAGE, stopReason: "toolUse", timestamp: 1 };
}
function toolResult(toolCallId, text) {
	return { role: "toolResult", toolCallId, toolName: "inspect", content: [{ type: "text", text }], isError: false, timestamp: 2 };
}
const streamEvent = (event) => ({ type: "stream_event", event });

/** A CC stand-in that streams `events`, then either finishes with `result` or, like CC
 *  waiting on a tools/call nobody answers, blocks until the bridge closes it. */
function streamingQuery(calls, events, { result } = {}) {
	return ({ prompt, options }) => {
		// Read at spawn, as the SDK does: the step fails its prompt stream once it ends.
		const call = { firstPrompt: prompt.next(), options, closed: false, sessionAtStart: null };
		if (options.resume) {
			const { jsonlPath } = openSession({ sessionId: options.resume, projectPath: options.cwd, claudeDir: process.env.CLAUDE_CONFIG_DIR });
			call.sessionAtStart = { jsonlPath, records: readFileSync(jsonlPath, "utf8").trim().split("\n").map((line) => JSON.parse(line)) };
		}
		calls.push(call);
		let wake;
		const closed = new Promise((resolve) => { wake = resolve; });
		async function* gen() {
			for (const event of events) yield streamEvent(event);
			if (result) {
				yield result;
				return;
			}
			await closed;
		}
		const q = gen();
		q.interrupt = async () => {};
		q.close = () => { call.closed = true; wake(); };
		return q;
	};
}

const SDK_TOOL = "mcp__custom-tools__inspect";
const toolUseEvents = (ids) => [
	{ type: "message_start", message: { id: "msg_1", usage: { input_tokens: 900, output_tokens: 1 } } },
	...ids.flatMap((id, index) => [
		{ type: "content_block_start", index, content_block: { type: "tool_use", id, name: SDK_TOOL, input: {} } },
		{ type: "content_block_delta", index, delta: { type: "input_json_delta", partial_json: `{"path":"${id}.md"}` } },
		{ type: "content_block_stop", index },
	]),
	{ type: "message_delta", delta: { stop_reason: "tool_use" }, usage: { output_tokens: 40 } },
	{ type: "message_stop" },
];

describe("an out-of-band tool step", () => {
	it("hands every tool call of the turn back to the caller and closes the query", async () => {
		__test.setSharedSession(null, { sessionId: "cc-session-kept", cursor: 3, cwd: "/somewhere" });
		__test.setQuery(streamingQuery(calls, toolUseEvents(["call_a", "call_b"])));
		const result = await call(
			{ systemPrompt: TOOL_LOOP_SYSTEM, messages: [user("Environment."), user("Review this tool call.")], tools: [INSPECT_TOOL] },
			{ reasoning: "low" },
		).result();

		assert.equal(result.stopReason, "toolUse", result.errorMessage);
		assert.deepEqual(result.content.map((block) => [block.type, block.id, block.name, block.arguments]), [
			["toolCall", "call_a", "inspect", { path: "call_a.md" }],
			["toolCall", "call_b", "inspect", { path: "call_b.md" }],
		], "both parallel calls reach the caller under pi's own tool name");
		assert.equal(result.usage.input, 900);
		assert.equal(result.usage.output, 40);

		assert.equal(calls.length, 1);
		const { options, firstPrompt } = calls[0];
		assert.ok(calls[0].closed, "the query is closed instead of waiting on handlers nobody will answer");
		assert.equal(options.systemPrompt, TOOL_LOOP_SYSTEM, "the extension's own prompt reaches CC");
		assert.equal(options.persistSession, false);
		assert.ok(options.maxTurns > 1, "CC may answer an unserved tool name and retry within the step");
		assert.deepEqual(options.tools, [], "no builtin tools, only the caller's over MCP");
		assert.ok(options.mcpServers?.["custom-tools"], "the caller's tools are served");
		assert.equal(options.resume, undefined, "a first round has no history to resume");
		assert.equal(options.effort, "low");
		const first = await firstPrompt;
		assert.deepEqual(first.value.message.content, [{ type: "text", text: "Environment.\nReview this tool call." }]);

		assert.deepEqual(__test.getSharedSession(null), { sessionId: "cc-session-kept", cursor: 3, cwd: "/somewhere" },
			"the shared session is untouched");
		__test.setSharedSession(null, null);
		assert.equal(__test.activeQueryContexts.size, 0, "no live query was opened");
	});

	it("continues from the caller's tool results in a throwaway session it then deletes", async () => {
		__test.setQuery(streamingQuery(calls, [
			{ type: "message_start", message: { id: "msg_2", usage: { input_tokens: 1000, output_tokens: 1 } } },
			{ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
			{ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "{\"outcome\":\"allow\"}" } },
			{ type: "content_block_stop", index: 0 },
			{ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 12 } },
			{ type: "message_stop" },
		], { result: { type: "result", subtype: "success", is_error: false, result: "{\"outcome\":\"allow\"}" } }));
		const result = await call({
			systemPrompt: TOOL_LOOP_SYSTEM,
			messages: [
				user("Review this tool call."),
				assistantCalls([toolCall("call_a"), toolCall("call_b")]),
				toolResult("call_a", "alpha contents"),
				toolResult("call_b", "beta contents"),
			],
			tools: [INSPECT_TOOL],
		}, {}).result();

		assert.equal(result.stopReason, "stop", result.errorMessage);
		assert.equal(result.content.map((block) => block.text).join(""), "{\"outcome\":\"allow\"}");

		const { options, firstPrompt, sessionAtStart } = calls[0];
		assert.ok(options.resume, "the history is resumed from a session");
		const content = sessionAtStart.records.flatMap((record) => Array.isArray(record.message?.content) ? record.message.content : []);
		assert.deepEqual(content.filter((block) => block.type === "tool_use").map((block) => block.name), [SDK_TOOL, SDK_TOOL],
			"the calls are imported under the names CC serves them by");
		assert.deepEqual(content.filter((block) => block.type === "tool_result").map((block) => block.tool_use_id), ["call_a", "call_b"]);
		const first = await firstPrompt;
		assert.match(first.value.message.content[0].text, /tool results above/, "a turn ending at tool results still gets a user message");
		assert.equal(existsSync(sessionAtStart.jsonlPath), false, "the throwaway session is deleted");
	});

	it("reports a failed result as an error the caller can retry", async () => {
		__test.setQuery(streamingQuery(calls, [], {
			result: { type: "result", subtype: "error_during_execution", is_error: true, errors: ["API Error: 529 overloaded"] },
		}));
		const result = await call(
			{ systemPrompt: TOOL_LOOP_SYSTEM, messages: [user("Review.")], tools: [INSPECT_TOOL] }, {},
		).result();
		assert.equal(result.stopReason, "error");
		assert.match(result.errorMessage, /529/);
	});

	it("ends aborted when the caller's deadline fires mid-turn", async () => {
		const controller = new AbortController();
		__test.setQuery(streamingQuery(calls, [
			{ type: "message_start", message: { id: "msg_3", usage: { input_tokens: 10, output_tokens: 1 } } },
		]));
		const pending = call(
			{ systemPrompt: TOOL_LOOP_SYSTEM, messages: [user("Review.")], tools: [INSPECT_TOOL] }, { signal: controller.signal },
		).result();
		setTimeout(() => controller.abort(), 10);
		const result = await pending;
		assert.equal(result.stopReason, "aborted");
		assert.ok(calls[0].closed);
	});
});
