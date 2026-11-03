#!/usr/bin/env node

/**
 * agent_settled records the prompt pi renders while idle.
 *
 * pi renders ctx.getSystemPrompt() from the per-run options during a run and from its
 * base options between runs; the per-run copy is dropped before agent_settled. Custom
 * sections that extensions add at before_agent_start (`<acp_delegate>`, ...) live only
 * in the per-run copy. pi-btw opens its side session while idle with the parent's
 * ctx.getSystemPrompt() plus an addendum, so the child embeds the base prompt, which no
 * run boundary records.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { createFauxCore, fauxAssistantMessage } from "@earendil-works/pi-ai";
import { createAgentSession, DefaultResourceLoader, formatSkillsForPrompt, SessionManager, SettingsManager } from "@earendil-works/pi-coding-agent";
import { PI_PREAMBLE, projectPromptCapture } from "../src/prompt-capture.js";
import { activateWithMockPi } from "./lib/mock-pi.mjs";

const { default: activate, __test } = await import("../src/index.js");

const section = (name, content) => `<${name}>\n${content}\n</${name}>`;
const BASE = [
	`${PI_PREAMBLE}, a coding agent harness. Idle-base fixture.`,
	section("tools", "- read: Read a file\n- bash: Run idle-base commands"),
	section("project_context", "Project-specific instructions and guidelines:\n\n<project_instructions path=\"/AGENTS.md\">\nproject rules\n</project_instructions>"),
	section("cwd", "/idle-base"),
].join("\n\n");
const DELEGATE = "Idle-base delegate notifications arrive asynchronously.";
const RUN = `${BASE}\n\n${section("acp_delegate", DELEGATE)}`;
const BTW = "Answer the idle-base side question briefly.";
const btwChild = (parent) => `${parent}\n\n<addendum>\n${BTW}\n</addendum>\n\n<cwd>\n/idle-base\n</cwd>`;
const CHILD = btwChild(BASE);

function runParent(handlers) {
	handlers.get("before_agent_start")({
		systemPrompt: RUN,
		systemPromptOptions: {
			contextFiles: [{ path: "/AGENTS.md", content: "project rules" }],
			skills: [],
			selectedTools: ["read", "bash"],
			sections: { acp_delegate: DELEGATE },
		},
	});
	handlers.get("agent_start")({}, { getSystemPrompt: () => RUN });
	handlers.get("turn_start")({}, { sessionManager: { getSessionId: () => "pi-session" }, getSystemPrompt: () => RUN });
}

describe("agent_settled idle prompt capture", () => {
	it("resolves a side session that embeds the idle prompt", () => {
		const handlers = activateWithMockPi();
		runParent(handlers);

		// Only the run prompt is known. The child embeds the idle prompt, which lacks the
		// run's acp_delegate section, so it embeds no known key.
		assert.throws(() => __test.promptCaptures.resolveOrDerive(CHILD), /no capture/);

		handlers.get("agent_settled")({ aborted: false }, { getSystemPrompt: () => BASE });

		const resolved = __test.promptCaptures.resolveOrDerive(CHILD);
		assert.ok(resolved, "the side session's prompt must resolve");
		assert.equal(resolved.inherited.length, 1);
		assert.equal(resolved.inherited[0].parent.assembledPrompt, BASE, "the child inherits from the idle prompt");

		const projected = projectPromptCapture(resolved, { skillReadTool: "none" });
		assert.match(projected, /project rules/, "the idle capture carries the stashed context files");
		assert.match(projected, new RegExp(BTW), "the side session's addendum is carried through");
		assert.doesNotMatch(projected, new RegExp(PI_PREAMBLE), "pi's harness is replaced by the portable parts");
		assert.doesNotMatch(projected, /<acp_delegate>/, "a per-run section is not attributed to the idle prompt");
	});

	it("is idempotent when the same idle prompt settles again", () => {
		const handlers = activateWithMockPi();
		runParent(handlers);
		handlers.get("agent_settled")({ aborted: false }, { getSystemPrompt: () => BASE });

		const before = __test.promptCaptures.size;
		handlers.get("agent_settled")({ aborted: false }, { getSystemPrompt: () => BASE });
		assert.equal(__test.promptCaptures.size, before, "re-settling an unchanged prompt must not grow the registry");
	});

	it("still resolves a child embedding the run prompt against the run capture", () => {
		const handlers = activateWithMockPi();
		runParent(handlers);
		handlers.get("agent_settled")({ aborted: false }, { getSystemPrompt: () => BASE });

		const resolved = __test.promptCaptures.resolveOrDerive(btwChild(RUN));
		assert.equal(resolved.inherited.length, 1);
		assert.equal(resolved.inherited[0].parent.assembledPrompt, RUN, "the longest embedded key wins");
		assert.match(projectPromptCapture(resolved, { skillReadTool: "none" }), /<acp_delegate>/);
	});

	it("attaches only the stashed skills the idle prompt lists", () => {
		// A handler that edits one run's selectedTools or skills leaves pi's base ones,
		// which the idle prompt renders, apart from both the stash and the active tools.
		const handlers = activateWithMockPi();
		const skill = (name) => ({
			name,
			description: `${name} description`,
			filePath: `/skills/${name}/SKILL.md`,
			baseDir: `/skills/${name}`,
			sourceInfo: { source: "test", scope: "temporary", origin: "top-level" },
			disableModelInvocation: false,
		});
		const base = skill("idle-base-skill");
		const runOnly = skill("idle-run-skill");
		// pi never lists a skill hidden from the model, so its empty entry must not match.
		const hidden = { ...skill("idle-hidden-skill"), disableModelInvocation: true };
		handlers.get("before_agent_start")({
			systemPrompt: RUN,
			systemPromptOptions: { contextFiles: [], skills: [base, runOnly, hidden], selectedTools: ["bash"] },
		});

		const listed = `${BASE}${formatSkillsForPrompt([base])}`;
		handlers.get("agent_settled")({ aborted: false }, { getSystemPrompt: () => listed });
		assert.deepEqual(__test.promptCaptures.resolve(listed).skills.map((s) => s.name), ["idle-base-skill"]);

		const quoted = `${BASE}\n\nA context file mentioning <available_skills> without listing any.`;
		handlers.get("agent_settled")({ aborted: false }, { getSystemPrompt: () => quoted });
		assert.deepEqual(__test.promptCaptures.resolve(quoted).skills, []);
	});

	it("records nothing when no run has stashed options", () => {
		const handlers = activateWithMockPi();
		const unseen = `${BASE}\n\nIdle-base prompt settled before any run.`;
		handlers.get("agent_settled")({ aborted: false }, { getSystemPrompt: () => unseen });
		assert.equal(__test.promptCaptures.resolve(unseen), undefined);
	});
});

// Real pi session and extension runner on a faux provider: pins pi's own ordering
// (per-run options dropped before agent_settled) rather than a mocked one.
it("real pi: a side session opened after a run resolves against the idle prompt", { timeout: 10000 }, async () => {
	const cwd = mkdtempSync(join(process.env.HOME, "idle-base-"));
	writeFileSync(join(cwd, "AGENTS.md"), "idle-base real project rules");
	const modelConfig = { id: "faux-idle-model", name: "Faux", reasoning: false, input: ["text"], contextWindow: 100000, maxTokens: 1000, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
	const faux = createFauxCore({ api: "faux-idle", provider: "faux-idle", models: [modelConfig] });
	faux.setResponses([fauxAssistantMessage("parent answer")]);
	const settings = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } });
	const runPrompts = [];
	const loader = new DefaultResourceLoader({
		cwd, agentDir: process.env.PI_CODING_AGENT_DIR, settingsManager: settings,
		noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true,
		// The section extension loads before the bridge, so the bridge's
		// before_agent_start already sees the run prompt with the section added.
		extensionFactories: [(pi) => {
			pi.registerProvider("faux-idle", { baseUrl: "faux", apiKey: "faux", api: "faux-idle", streamSimple: faux.streamSimple, models: [modelConfig] });
			pi.on("before_agent_start", (event) => {
				event.systemPromptOptions.sections = { ...event.systemPromptOptions.sections, acp_delegate: "idle-base real delegate notes" };
			});
			pi.on("turn_start", (_event, ctx) => runPrompts.push(ctx.getSystemPrompt()));
		}, activate],
	});
	let session;
	try {
		await loader.reload();
		assert.deepEqual(loader.getExtensions().errors, []);
		({ session } = await createAgentSession({ cwd, agentDir: process.env.PI_CODING_AGENT_DIR, resourceLoader: loader, sessionManager: SessionManager.inMemory(cwd), settingsManager: settings, noTools: "all", model: faux.getModel() }));
		await session.prompt("hello");

		const idle = session.systemPrompt;
		assert.ok(runPrompts.length > 0 && runPrompts.every((p) => p.includes("<acp_delegate>")), "the run prompt carries the section");
		assert.ok(!idle.includes("<acp_delegate>"), "the idle prompt lacks the per-run section");

		// pi-btw's child prompt: the parent's ctx.getSystemPrompt() plus an addendum.
		const child = `${idle}\n\n<addendum>\nidle-base real side question\n</addendum>\n\n<cwd>\n${cwd}\n</cwd>`;
		const resolved = __test.promptCaptures.resolveOrDerive(child);
		assert.equal(resolved.inherited[0]?.parent.assembledPrompt, idle);
		const projected = projectPromptCapture(resolved, { skillReadTool: "none" });
		assert.match(projected, /idle-base real project rules/);
		assert.match(projected, /idle-base real side question/);
		assert.doesNotMatch(projected, new RegExp(PI_PREAMBLE));
	} finally {
		if (session) await session.extensionRunner.emit({ type: "session_shutdown" });
		session?.dispose();
	}
});
