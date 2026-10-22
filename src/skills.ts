import { formatSkillsForPrompt, type Skill } from "@earendil-works/pi-coding-agent";

export const MCP_SERVER_NAME = "custom-tools";
export const MCP_TOOL_PREFIX = `mcp__${MCP_SERVER_NAME}__`;

export type SkillReadTool = "mcp" | "native" | "none";

export function renderSkillsBlock(skills: Skill[], readTool: SkillReadTool): string | undefined {
	if (readTool === "none" || skills.length === 0) return undefined;
	const block = formatSkillsForPrompt(skills).trim();
	if (!block) return undefined;
	return readTool === "mcp" ? rewriteSkillsBlock(block) : block;
}

// The skills whose entry pi rendered into `prompt`. Matching each skill's whole entry,
// rather than the block's tag, keeps a context file that quotes the tag from counting
// and leaves out skills the prompt does not list.
export function skillsListedIn(prompt: string, skills: Skill[]): Skill[] {
	return skills.filter((skill) => {
		const block = formatSkillsForPrompt([skill]);
		const start = block.indexOf("  <skill>");
		const end = block.indexOf("</available_skills>");
		return start >= 0 && end > start && prompt.includes(block.slice(start, end));
	});
}

export function rewriteSkillsBlock(skillsBlock: string): string {
	return skillsBlock.replace(
		"Use the read tool to load a skill's file",
		`Use the read tool (mcp__${MCP_SERVER_NAME}__read) to load a skill's file`,
	);
}
