// Anthropic's prompt-cache diagnosis: with the cache-diagnosis beta, Claude Code names the
// previous response on each request and the API answers, on message_start, why this one
// missed the cache. pi's own cache miss notice is derived from usage alone, so the bridge
// keeps the API's reason on the assistant message and shows it as a session entry.
//
// Logging-free like usage.ts; index.ts wires it in.

import type { AssistantMessage, JsonObject } from "@earendil-works/pi-ai";

export const CACHE_MISS_DIAGNOSTIC = "anthropic_cache_miss_reason";
export const CACHE_MISS_ENTRY = "claude-bridge:cache-miss-reason";

/** pi's notice skips misses under both floors. */
const NOTICE_MIN_TOKENS = 20_000;
const NOTICE_MIN_COST = 0.1;

export type CacheMissReason = { type: string; cache_missed_input_tokens?: number };

/** The API message's diagnosed reason, or undefined when there is none: no divergence,
 *  diagnosis not requested, or `cache_miss_reason: null` while the API is still working
 *  it out. */
function cacheMissReason(apiMessage: unknown): CacheMissReason | undefined {
	const reason = (apiMessage as { diagnostics?: { cache_miss_reason?: unknown } | null } | undefined)?.diagnostics?.cache_miss_reason;
	if (reason === null || typeof reason !== "object" || typeof (reason as { type?: unknown }).type !== "string") return undefined;
	return reason as CacheMissReason;
}

/** Records the API message's diagnosis on `output`, replacing one from an earlier request
 *  of the same turn: the turn's usage is the latest request's, and so is its reason. */
export function recordCacheMissReason(output: AssistantMessage, apiMessage: unknown): void {
	const reason = cacheMissReason(apiMessage);
	const kept = (output.diagnostics ?? []).filter((d) => d.type !== CACHE_MISS_DIAGNOSTIC);
	if (reason) kept.push({ type: CACHE_MISS_DIAGNOSTIC, timestamp: Date.now(), details: reason as JsonObject });
	if (kept.length > 0 || output.diagnostics) output.diagnostics = kept;
}

/** The recorded reason when it explains a miss pi reports: a completed response whose
 *  missed tokens, or their price over the cache-read rate, reach pi's notice floor.
 *  Without a count from the API, the prompt tokens this response paid for full price
 *  stand in. */
export function noticeableCacheMiss(message: AssistantMessage): CacheMissReason | undefined {
	if (message.stopReason === "aborted" || message.stopReason === "error") return undefined;
	const reason = message.diagnostics?.find((d) => d.type === CACHE_MISS_DIAGNOSTIC)?.details as CacheMissReason | undefined;
	if (!reason) return undefined;
	const { input, cacheRead, cacheWrite, cost } = message.usage;
	const missed = reason.cache_missed_input_tokens ?? input + cacheWrite;
	const paidPerToken = input + cacheWrite > 0 ? (cost.input + cost.cacheWrite) / (input + cacheWrite) : 0;
	const readPerToken = cacheRead > 0 ? cost.cacheRead / cacheRead : 0;
	const missedCost = missed * Math.max(0, paidPerToken - readPerToken);
	return missed >= NOTICE_MIN_TOKENS || missedCost >= NOTICE_MIN_COST ? reason : undefined;
}

export function formatCacheMissReason(reason: CacheMissReason): string {
	const tokens = typeof reason.cache_missed_input_tokens === "number"
		? ` (${reason.cache_missed_input_tokens.toLocaleString("en-US")} tokens)`
		: "";
	return `Cache miss reason: ${reason.type.replaceAll("_", " ")}${tokens}`;
}
