/**
 * Regression tests for syncSharedSession's session reuse decisions.
 */
import { describe, it, afterEach } from "node:test";
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createSession, deleteSession, openSession } from "cc-session-io";

const { __test } = await import("../src/index.js");

describe("syncSharedSession", () => {
	afterEach(() => {
		__test.resetSharedSession();
		__test.setPiUI(null);
	});

	// Fresh-session transcript: the system prompt arrives as a leading system message
	// (issue #106). It is prompt state, not history — a fresh session must still take
	// the clean-start path (empty priors) rather than rebuild a session file holding nothing
	// but a system head, which made --resume fail with "No conversation found".
	it("takes the clean-start path when a transcript system message precedes the first user message", () => {
		const cwd = mkdtempSync(join(tmpdir(), "sync-shared-session-"));
		try {
			const result = __test.syncSharedSession([
				{ role: "system", content: "You are Claude Code.", timestamp: Date.now() },
				{ role: "user", content: "Hello", timestamp: Date.now() },
			], cwd);

			assert.equal(result.sessionId, null, "a fresh session with only prompt state as priors is a clean start");
			assert.equal(result.preserveSharedSession, undefined);
			assert.equal(__test.getSharedSession(null), null, "a clean start must not create a session state");
		} finally {
			rmSync(cwd, { recursive: true, force: true });
		}
	});

	// Mid-conversation tool-loadout updates land in the transcript as system messages. They must not inflate the cursor or be imported as history, or the next turn's
	// reuse check (priors >= cursor) fails and every turn rebuilds the session.
	it("keeps cursor arithmetic consistent when system messages punctuate the history", () => {
		const cwd = mkdtempSync(join(tmpdir(), "sync-shared-session-"));
		const sessionId = randomUUID();
		try {
			const seeded = createSession({ sessionId, projectPath: cwd });
			seeded.importMessages([
				{ role: "user", content: "Hi" },
				{ role: "assistant", content: [{ type: "text", text: "Hello." }] },
			]);
			seeded.save();
			__test.setSharedSession(null, { sessionId, cursor: 2, cwd });

			const result = __test.syncSharedSession([
				{ role: "user", content: "Hi", timestamp: Date.now() },
				{ role: "assistant", content: [{ type: "text", text: "Hello." }], timestamp: Date.now() },
				{ role: "system", content: "", toolsAdded: [{ name: "grep", description: "", parameters: {} }], timestamp: Date.now() },
				{ role: "user", content: "Next", timestamp: Date.now() },
			], cwd);

			assert.equal(result.sessionId, sessionId, "2 priors at cursor 2 must resume, not rebuild");
			assert.equal(__test.getSharedSession(null)?.cursor, 2, "cursor counts non-system messages only");
			const session = openSession({ sessionId, projectPath: cwd });
			assert.deepEqual(
				session.messages.map((m) => m.type),
				["user", "assistant"],
				"the resumed session file must hold the non-system history",
			);
		} finally {
			deleteSession(sessionId, cwd);
			rmSync(cwd, { recursive: true, force: true });
		}
	});

	// The branch this exercises is the guard that stops a reentrant subagent from
	// resuming — and then overwriting — the parent's session: a subagent's context
	// is shorter than the parent's cursor, so it starts fresh and the parent's
	// session is preserved. It was previously described here as the compact-summary
	// path, which cannot reach syncSharedSession at all, so the branch read as
	// covered for a case that never happens.
	it("starts a fresh session for a shorter context and preserves the parent's", () => {
		const cwd = mkdtempSync(join(tmpdir(), "sync-shared-session-"));
		try {
			const mainSession = {
				sessionId: "11111111-1111-4111-8111-111111111111",
				cursor: 42,
				cwd,
			};
			__test.setSharedSession(null, mainSession);

			const result = __test.syncSharedSession([
				{
					role: "user",
					content: "Summarize this conversation.",
					timestamp: Date.now(),
				},
			], cwd);

			assert.equal(
				result.sessionId,
				null,
				"a context shorter than the cursor — a subagent, or AskClaude — must start a fresh Claude Code session instead of resuming the parent's",
			);
			assert.equal(
				result.preserveSharedSession,
				true,
				"the fresh session must not replace the parent's when it completes",
			);
			assert.deepEqual(__test.getSharedSession(null), mainSession);
		} finally {
			rmSync(cwd, { recursive: true, force: true });
		}
	});

	// The rebuilt file holds one line per record, and a carried `@file` expansion
	// is an `attachment` record — which `session.messages` filters out. Counting
	// messages told every user who at-mentioned a file before switching providers
	// that their session was corrupt, and asked them to open an issue about it.
	it("does not report a count mismatch when a rebuild carries an attachment", () => {
		const cwd = mkdtempSync(join(tmpdir(), "sync-shared-session-"));
		const sessionId = randomUUID();
		const prompt = "Review @fixture.txt and remember it.";
		const notices = [];
		try {
			const seeded = createSession({ sessionId, projectPath: cwd });
			seeded.importMessages(
				[
					{ role: "user", content: prompt },
					{ role: "assistant", content: [{ type: "text", text: "Noted." }] },
				],
				{
					attachments: [{
						afterIndex: 0,
						attachment: {
							type: "file",
							filename: join(cwd, "fixture.txt"),
							content: { type: "text", file: { filePath: join(cwd, "fixture.txt"), content: "token" } },
						},
					}],
				},
			);
			seeded.save();

			__test.setSharedSession(null, { sessionId, cursor: 0, cwd });
			__test.setPiUI({ notify: (message) => notices.push(message) });
			__test.syncSharedSession([
				{ role: "user", content: prompt, timestamp: Date.now() },
				{ role: "assistant", content: [{ type: "text", text: "Noted." }], timestamp: Date.now() },
				{ role: "user", content: "Now what did it say?", timestamp: Date.now() },
			], cwd);

			assert.equal(
				openSession({ sessionId, projectPath: cwd }).attachments.length,
				1,
				"the rebuild did not carry the attachment, so this proves nothing about the count",
			);
			assert.deepEqual(notices, []);
		} finally {
			deleteSession(sessionId, cwd);
			rmSync(cwd, { recursive: true, force: true });
		}
	});

	// Imported records carry zero usage, which Claude Code skips when it sizes a resumed
	// session, so it estimated the whole transcript from its bytes and refused a long one
	// as "Prompt is too long" without sending it. The rebuild hands it pi's measurement.
	it("stamps pi's last measured usage on the last assistant record of a rebuild", () => {
		const cwd = mkdtempSync(join(tmpdir(), "sync-shared-session-"));
		const usage = (input, cacheRead) => ({ input, output: 50, cacheRead, cacheWrite: 10, totalTokens: input + 50 + cacheRead + 10 });
		try {
			const result = __test.syncSharedSession([
				{ role: "user", content: "One", timestamp: Date.now() },
				{ role: "assistant", content: [{ type: "text", text: "First." }], usage: usage(5, 1000), stopReason: "stop", timestamp: Date.now() },
				{ role: "user", content: "Two", timestamp: Date.now() },
				{ role: "assistant", content: [{ type: "text", text: "Second." }], usage: usage(7, 2000), stopReason: "stop", timestamp: Date.now() },
				{ role: "user", content: "Three", timestamp: Date.now() },
			], cwd);

			const assistants = openSession({ sessionId: result.sessionId, projectPath: cwd }).messages
				.filter((m) => m.type === "assistant")
				.map((m) => m.message.usage);
			assert.deepEqual(assistants, [
				{ input_tokens: 0, output_tokens: 0 },
				{ input_tokens: 7, output_tokens: 50, cache_read_input_tokens: 2000, cache_creation_input_tokens: 10 },
			]);
			deleteSession(result.sessionId, cwd);
		} finally {
			rmSync(cwd, { recursive: true, force: true });
		}
	});

	// An errored turn keeps its partial content but has no usage. The measurement belongs on
	// the turn that took it, or Claude Code would count the messages after it as measured too.
	it("stamps the measured turn when an unmeasured turn follows it", () => {
		const cwd = mkdtempSync(join(tmpdir(), "sync-shared-session-"));
		const usage = { input: 7, output: 50, cacheRead: 2000, cacheWrite: 10, totalTokens: 2067 };
		const none = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0 };
		try {
			const result = __test.syncSharedSession([
				{ role: "user", content: "One", timestamp: Date.now() },
				{ role: "assistant", content: [{ type: "text", text: "First." }], usage, stopReason: "stop", timestamp: Date.now() },
				{ role: "user", content: "x".repeat(50_000), timestamp: Date.now() },
				{ role: "assistant", content: [{ type: "text", text: "Partial" }], usage: none, stopReason: "error", timestamp: Date.now() },
				{ role: "assistant", content: [], usage: none, stopReason: "aborted", timestamp: Date.now() },
				{ role: "user", content: "Again", timestamp: Date.now() },
			], cwd);

			const assistants = openSession({ sessionId: result.sessionId, projectPath: cwd }).messages
				.filter((m) => m.type === "assistant")
				.map((m) => m.message.usage);
			assert.deepEqual(assistants, [
				{ input_tokens: 7, output_tokens: 50, cache_read_input_tokens: 2000, cache_creation_input_tokens: 10 },
				{ input_tokens: 0, output_tokens: 0 },
			]);
			deleteSession(result.sessionId, cwd);
		} finally {
			rmSync(cwd, { recursive: true, force: true });
		}
	});
});
