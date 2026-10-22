// Token accounting: map Claude Code's SDK usage counters onto pi's AssistantMessage.
// Logging-free like convert.ts — it returns what it derived and index.ts logs it.
// Extracted from index.ts so tests can reach it without activating the extension.

import { calculateCost, type Api, type AssistantMessage, type Message, type Model } from "@earendil-works/pi-ai";

/**
 * The usage counters Claude Code reports, mirroring `Usage` / `MessageDeltaUsage` in
 * @anthropic-ai/sdk. Every field is optional because `message_start`, `message_delta` and a
 * complete assistant message each carry a different subset.
 *
 * Not `Record<string, number | undefined>`: an index signature makes every property access
 * typecheck, which is what hid a read of `usage.reasoning_tokens` — a field no SDK version
 * has — for months.
 */
export type SdkUsage = {
	input_tokens?: number | null;
	output_tokens?: number | null;
	cache_read_input_tokens?: number | null;
	cache_creation_input_tokens?: number | null;
	output_tokens_details?: { thinking_tokens?: number | null } | null;
};

/** What updateUsage derived but does not store on the message, for the debug line in index.ts. */
export type UsageReport = {
	/** Cache-read share of the prompt, rounded to a whole percentage; 0 when the prompt is empty. */
	cachePct: number;
	/** Thinking tokens carried by THIS frame, absent when it carried none. Distinct from
	 *  `output.usage.reasoning`, which keeps the last value seen across frames. */
	reasoning?: number;
};

/**
 * Applies one SDK usage frame onto `output.usage`, then recomputes the token total and cost.
 *
 * Counters the frame omits keep their previous value, so the partial frames the three call
 * sites deliver never zero out what an earlier one set.
 */
export function updateUsage(output: AssistantMessage, usage: SdkUsage, model: Model<Api>): UsageReport {
	if (usage.input_tokens != null) output.usage.input = usage.input_tokens;
	if (usage.output_tokens != null) output.usage.output = usage.output_tokens;
	if (usage.cache_read_input_tokens != null) output.usage.cacheRead = usage.cache_read_input_tokens;
	if (usage.cache_creation_input_tokens != null) output.usage.cacheWrite = usage.cache_creation_input_tokens;
	// A subset of output_tokens, never a separate addend in the total or in cost.
	const reasoning = usage.output_tokens_details?.thinking_tokens;
	if (reasoning != null) output.usage.reasoning = reasoning;
	output.usage.totalTokens = output.usage.input + output.usage.output + output.usage.cacheRead + output.usage.cacheWrite;
	calculateCost(model, output.usage);
	const promptTokens = output.usage.input + output.usage.cacheRead + output.usage.cacheWrite;
	const report: UsageReport = { cachePct: promptTokens > 0 ? Math.round(output.usage.cacheRead / promptTokens * 100) : 0 };
	if (reasoning != null) report.reasoning = reasoning;
	return report;
}

/** The opening of the user message pi substitutes for compacted history
 *  (`COMPACTION_SUMMARY_PREFIX` in pi-coding-agent's core/messages.js, not exported). */
const COMPACTION_SUMMARY_PREFIX = "The conversation history before this point was compacted into the following summary:";

function isCompactionSummary(msg: Message): boolean {
	if (msg.role !== "user") return false;
	const first = typeof msg.content === "string" ? msg.content : msg.content.find((b) => b.type === "text")?.text;
	return first?.startsWith(COMPACTION_SUMMARY_PREFIX) ?? false;
}

/** Claude Code's usage counters with every field present, as written onto a transcript record. */
export type ContextUsage = {
	input_tokens: number;
	output_tokens: number;
	cache_read_input_tokens: number;
	cache_creation_input_tokens: number;
};

/**
 * The context size pi last measured in `messages`, as the counters Claude Code reads off an
 * assistant record, with the index of the message that carries it; undefined when pi has no
 * trustworthy measurement.
 *
 * Claude Code sizes a resumed session from the usage on its last assistant record with a
 * nonzero total, and estimates everything after that record from the transcript's bytes. A
 * rebuilt transcript carries zero usage on every record, so the whole history is estimated —
 * and a long one that a proxy shrinks upstream is refused locally as "Prompt is too long"
 * before any request is sent. Stamping pi's own measurement restores the anchor.
 *
 * Follows pi's context accounting: aborted and errored turns carry no usable usage, and an
 * assistant from before the latest compaction measured the pre-compaction context.
 */
export function lastContextUsage(messages: readonly Message[]): { usage: ContextUsage; index: number } | undefined {
	for (let i = messages.length - 1; i >= 0; i--) {
		const msg = messages[i];
		if (isCompactionSummary(msg)) return undefined;
		if (msg.role !== "assistant" || msg.stopReason === "aborted" || msg.stopReason === "error" || !msg.usage) continue;
		const { input, output, cacheRead, cacheWrite } = msg.usage;
		if (input + output + cacheRead + cacheWrite > 0) {
			return { usage: { input_tokens: input, output_tokens: output, cache_read_input_tokens: cacheRead, cache_creation_input_tokens: cacheWrite }, index: i };
		}
	}
	return undefined;
}
