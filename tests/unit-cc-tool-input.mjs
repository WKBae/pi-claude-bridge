/**
 * Claude Code coerces a streamed tool input to the tool's schema before it
 * records, re-sends, and dispatches it: an array the model streamed as a JSON
 * string becomes an array. pi records the tool call the bridge builds, and a
 * rebuilt session re-sends that record, so a call built from the raw stream
 * diverged from what CC had sent and missed the prompt cache from that call on.
 * tests/int-cc-contracts.mjs pins the CC side (the assistant message carries the
 * coerced input, before content_block_stop).
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { QueryContext } from "../src/query-state.js";

const { __test } = await import("../src/index.js");

const fakeModel = { api: "anthropic-messages", provider: "anthropic", id: "test-model" };
const toolMap = new Map([["mcp__custom-tools__compress", "compress"]]);

function fakeStream() {
	const events = [];
	return { events, push: (e) => events.push(e), end: () => events.push({ type: "end" }) };
}

function makeCtx() {
	const c = new QueryContext();
	c.currentPiStream = fakeStream();
	c.resetTurnState(fakeModel);
	return c;
}

async function consume(c, messages) {
	async function* gen() { for (const m of messages) yield m; }
	await __test.consumeQuery(gen(), toolMap, fakeModel, () => false, c);
}

const streamEvent = (event) => ({ type: "stream_event", event });
const ranges = [{ startId: "m00001", endId: "m00002", summary: "s" }];
const raw = JSON.stringify({ content: JSON.stringify(ranges) });
const coerced = { content: ranges };
const ccAssistant = {
	type: "assistant",
	message: { id: "msg_1", content: [{ type: "tool_use", id: "toolu_1", name: "mcp__custom-tools__compress", input: coerced }] },
};

const start = [
	streamEvent({ type: "message_start", message: { id: "msg_1" } }),
	streamEvent({ type: "content_block_start", index: 0, content_block: { type: "tool_use", name: "mcp__custom-tools__compress", id: "toolu_1", input: {} } }),
	streamEvent({ type: "content_block_delta", index: 0, delta: { type: "input_json_delta", partial_json: raw } }),
];
const stop = [
	streamEvent({ type: "content_block_stop", index: 0 }),
	streamEvent({ type: "message_stop" }),
];

describe("streamed tool input", () => {
	it("takes the input Claude Code coerced, delivered before content_block_stop", async () => {
		const c = makeCtx();
		const stream = c.currentPiStream;
		await consume(c, [...start, ccAssistant, ...stop]);

		const [call] = c.turnOutput.content.filter((b) => b.type === "toolCall");
		assert.deepStrictEqual(call.arguments, coerced);
		assert.deepStrictEqual(Object.keys(call).sort(), ["arguments", "id", "name", "type"]);
		const end = stream.events.find((e) => e.type === "toolcall_end");
		assert.deepStrictEqual(end.toolCall.arguments, coerced);
	});

	it("keeps the streamed input when no coerced input arrived before the block stopped", async () => {
		const c = makeCtx();
		await consume(c, [...start, ...stop, ccAssistant]);

		const [call] = c.turnOutput.content.filter((b) => b.type === "toolCall");
		assert.deepStrictEqual(call.arguments, JSON.parse(raw));
	});
});
