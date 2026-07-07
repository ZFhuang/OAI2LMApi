import * as vscode from 'vscode';

/**
 * Model override configuration from user settings.
 * This interface mirrors the `oai2lmapi.modelOverrides` schema in package.json.
 */
export interface ModelOverrideConfig {
    maxInputTokens?: number;
    maxOutputTokens?: number;
    /**
     * Numeric cost/scale multiplier shown by VS Code/Copilot model picker surfaces.
     */
    multiplierNumeric?: number;
    /** Display pricing label, e.g. "Free", "2x", "$0.01/request". */
    pricing?: string;
    /** Input cost in credits per 1M tokens. */
    inputCost?: number;
    /** Output cost in credits per 1M tokens. */
    outputCost?: number;
    /** Cached input (read) cost in credits per 1M tokens. */
    cacheCost?: number;
    /** Cache write cost in credits per 1M tokens. */
    cacheWriteCost?: number;
    /** Long-context input cost (only when differs from default). */
    longContextInputCost?: number;
    /** Long-context output cost (only when differs from default). */
    longContextOutputCost?: number;
    /** Long-context cache read cost (only when differs from default). */
    longContextCacheCost?: number;
    /** Long-context cache write cost (only when differs from default). */
    longContextCacheWriteCost?: number;
    /** Relative pricing category: "low" | "medium" | "high" | "very_high". */
    priceCategory?: string;
    /** Model tier: "lightweight" | "versatile" | "powerful". */
    category?: string;
    supportsToolCalling?: boolean;
    supportsImageInput?: boolean;
    /**
     * Default temperature for matching models.
     */
    temperature?: number;
    /**
     * Thinking level: number for token budget, or 'low'/'medium'/'high'/'auto'/'none'.
     */
    thinkingLevel?: string | number;
    /**
     * Reasoning effort levels the model supports, used to build the configurationSchema.
     */
    supportedReasoningEfforts?: string[];
    /**
     * Default reasoning effort for the model.
     */
    defaultReasoningEffort?: string;
    /**
     * Body shape for reasoning effort: "chat" (top-level `reasoning_effort`) or
     * "responses" (nested `reasoning.effort`). Defaults to "chat".
     */
    reasoningEffortFormat?: 'chat' | 'responses';
    /**
     * When enabled, tools are converted to XML-format instructions in the system prompt
     * instead of using native function calling.
     */
    usePromptBasedToolCalling?: boolean;

    /**
     * When true, use the OpenAI Responses API for matching models in the OpenAI channel.
     * When false, force the legacy Chat Completions API for matching models.
     */
    useResponsesApi?: boolean;

    /**
     * When true, suppress chain-of-thought transmission for matching models.
     * See `oai2lmapi.suppressChainOfThought` for exact behavior.
     */
    suppressChainOfThought?: boolean;

    /**
     * When true, trims leading/trailing whitespace from XML tool call parameter values.
     * Default is false (whitespace is preserved).
     */
    trimXmlToolParameterWhitespace?: boolean;
}

export type ModelOverrideMap = Record<string, ModelOverrideConfig>;

export type ChannelModelOverrides = Record<string, ModelOverrideMap>;

/**
 * Escapes special regex characters in a string.
 */
export function escapeRegex(str: string): string {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Converts a wildcard pattern (e.g., 'gpt-*') into a RegExp.
 * Wildcard '*' matches any sequence of characters. Matching is case-insensitive.
 */
export function wildcardPatternToRegex(pattern: string): RegExp {
    const regexPattern = escapeRegex(pattern)
        .replace(/\\\*/g, '.*')
        .replace(/\\\?/g, '.');
    return new RegExp(`^${regexPattern}$`, 'i');
}

function isWildcardPattern(pattern: string): boolean {
    return pattern.includes('*') || pattern.includes('?');
}

function matchesModelPattern(modelId: string, pattern: string): boolean {
    if (pattern === modelId) {
        return true;
    }
    if (!isWildcardPattern(pattern)) {
        return false;
    }
    const regex = wildcardPatternToRegex(pattern);
    return regex.test(modelId);
}

function collectMatchingOverrides(modelId: string, overrides: ModelOverrideMap): ModelOverrideConfig[] {
    const matches: ModelOverrideConfig[] = [];
    for (const [pattern, override] of Object.entries(overrides)) {
        if (matchesModelPattern(modelId, pattern)) {
            matches.push(override);
        }
    }
    return matches;
}

/**
 * The complete set of edit tool names recognized by VSCode/Copilot editing flows.
 *
 * These are returned as a hint to the editor when a model supports tool calling.
 * VSCode will make all recognized tools available to the model and its
 * `EditToolLearningService` will automatically disable tools that perform
 * poorly for a given model, so it is safe to declare all tools unconditionally.
 *
 * See the JSDoc on `LanguageModelChatCapabilities.editTools` in
 * `vscode.proposed.chatProvider.d.ts`:
 * "If not provided or if none of the tools are recognized, the editor will try
 * multiple edit tools and pick the best one. ... all of the recognized edit
 * tools will be made available to the model."
 */
export const ALL_EDIT_TOOLS: readonly string[] = [
    'apply-patch',
    'multi-find-replace',
    'find-replace',
    'code-rewrite'
];

/**
 * Returns the edit tools to declare for a model.
 *
 * When the model supports tool calling, all recognized edit tools are returned
 * so VSCode/Copilot can pick the best one at runtime (and learn from past
 * attempts). When the model does not support tool calling, `undefined` is
 * returned so the editor falls back to its default behavior.
 *
 * @param supportsToolCalling - whether the model supports tool/function calling
 * @returns the list of edit tool names, or `undefined` if tool calling is unsupported
 */
export function getEditTools(supportsToolCalling: boolean): string[] | undefined {
    return supportsToolCalling ? [...ALL_EDIT_TOOLS] : undefined;
}

/**
 * Gets model override configuration for a given model ID from VSCode settings.
 * Supports wildcard patterns like 'gpt-*' with case-insensitive matching.
 * 
 * @param modelId - The model ID to look up
 * @returns The model override configuration if found, undefined otherwise
 */
export function getModelOverride(modelId: string, channel?: string): ModelOverrideConfig | undefined {
    const config = vscode.workspace.getConfiguration('oai2lmapi');
    const globalOverrides = config.get<ModelOverrideMap>('modelOverrides', {});
    const channelOverrides = config.get<ChannelModelOverrides>('channelModelOverrides', {});

    const mergedOverrides: ModelOverrideConfig[] = [];
    mergedOverrides.push(...collectMatchingOverrides(modelId, globalOverrides));
    if (channel && channelOverrides[channel]) {
        mergedOverrides.push(...collectMatchingOverrides(modelId, channelOverrides[channel]));
    }

    if (mergedOverrides.length === 0) {
        return undefined;
    }

    return mergedOverrides.reduce<ModelOverrideConfig>((acc, override) => ({
        ...acc,
        ...override
    }), {});
}
