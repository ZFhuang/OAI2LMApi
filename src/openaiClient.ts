import OpenAI from 'openai';
import { logger } from './logger';

export interface OpenAIConfig {
    apiEndpoint: string;
    apiKey: string;
}

/**
 * Model information returned from the /v1/models API.
 * Extended with optional fields that some providers include.
 *
 * Pricing fields mirror the VS Code proposed `LanguageModelChatInformation`
 * (see `vscode.proposed.languageModelPricing.d.ts`) so that values reported by
 * gateways (OpenRouter, CodeBuddy-style, custom proxies) can be surfaced to
 * the model picker hover. Costs are expressed in credits per 1M tokens, the
 * same unit VS Code renders in the hover.
 */
export interface APIModelInfo {
    id: string;
    object: string;
    created?: number;
    owned_by?: string;
    name?: string;
    display_name?: string;
    description?: string;
    canonical_slug?: string;
    type?: string;
    created_at?: string | number;
    credits?: string;
    credit_multiplier?: number;
    vendor?: string;
    maxInputTokens?: number;
    maxOutputTokens?: number;
    maxAllowedSize?: number;
    supported_parameters?: string[];
    default_parameters?: Record<string, unknown>;
    architecture?: {
        modality?: string;
        input_modalities?: string[];
        output_modalities?: string[];
        tokenizer?: string | null;
        instruct_type?: string | null;
    };
    top_provider?: {
        context_length?: number;
        max_completion_tokens?: number;
        is_moderated?: boolean;
    };
    // Extended fields from some providers (e.g., OpenRouter and CodeBuddy-style gateways)
    context_length?: number;
    max_completion_tokens?: number;
    supports_vision?: boolean;
    supports_tools?: boolean;
    supports_tool_use?: boolean;
    supports_function_calling?: boolean;
    supports_reasoning?: boolean;
    supportsImages?: boolean;
    supportsToolCall?: boolean;
    supportsReasoning?: boolean;
    disabledMultimodal?: boolean;
    capabilities?: {
        tool_calling?: boolean;
        vision?: boolean;
        function_calling?: boolean;
        tool_use?: boolean;
        tools?: boolean;
        reasoning?: boolean;
    };
    // --- Pricing / cost fields (credits per 1M tokens) ---
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
    /** Reasoning effort levels supported by the model (e.g. ["low","medium","high"]). */
    supportedReasoningEfforts?: string[];
    /** Default reasoning effort for the model. */
    defaultReasoningEffort?: string;
    /** Body shape for reasoning effort: "chat" (top-level) or "responses" (nested). */
    reasoningEffortFormat?: 'chat' | 'responses';
}

/**
 * Normalizes the API endpoint URL by removing trailing slashes.
 * The OpenAI SDK expects baseURL to be the full path (e.g., https://api.openai.com/v1).
 * This function ensures trailing slashes are removed for consistent URL construction.
 * @param endpoint - The API endpoint URL (should include /v1 path for OpenAI-compatible APIs)
 * @returns The normalized endpoint URL without trailing slashes
 */
function normalizeApiEndpoint(endpoint: string): string {
    // Remove trailing slashes for consistent URL construction
    return endpoint.replace(/\/+$/, '');
}

/**
 * Represents a tool call made by the model
 */
export interface ToolCall {
    id: string;
    type: 'function';
    function: {
        name: string;
        arguments: string;
    };
}

/**
 * Represents a tool definition for the OpenAI API
 */
export interface ToolDefinition {
    type: 'function';
    function: {
        name: string;
        description?: string;
        parameters?: Record<string, unknown>;
    };
}

/**
 * Tool choice options for the OpenAI API
 */
export type ToolChoice = 'none' | 'auto' | 'required' | { type: 'function'; function: { name: string } };

export type ChatMessageContentPart =
    | { type: 'text'; text: string }
    | { type: 'image_url'; image_url: { url: string; detail?: 'auto' | 'low' | 'high' } };

export type ChatMessageContent = string | ChatMessageContentPart[] | null;

export interface ChatMessage {
    role: 'system' | 'user' | 'assistant' | 'tool';
    content: ChatMessageContent;
    tool_calls?: ToolCall[];
    tool_call_id?: string;
}

export interface OpenAIUsage {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    prompt_tokens_details?: {
        cached_tokens: number;
        cache_creation_tokens?: number;
        cache_creation_input_tokens?: number;
    };
    completion_tokens_details?: {
        reasoning_tokens: number;
        accepted_prediction_tokens: number;
        rejected_prediction_tokens: number;
    };
}

export interface OpenAIResponseMetadata {
    id?: string;
    object?: string;
    model?: string;
    created?: number;
    system_fingerprint?: string;
    service_tier?: string;
    finish_reason?: string;
    status?: string;
    error?: unknown;
    incomplete_details?: unknown;
}

export interface OpenAIRequestOptions {
    temperature?: number;
    topP?: number;
    stop?: string | string[];
    reasoning?: Record<string, unknown>;
    includeReasoning?: boolean;
    responseFormat?: Record<string, unknown>;
    serviceTier?: string;
    /**
     * Body shape for reasoning effort.
     * - "chat" (default): top-level `reasoning_effort` field (OpenAI Chat Completions, OpenRouter).
     * - "responses": nested `reasoning.effort` field (OpenAI Responses API style).
     * When unset, the client falls back to top-level `reasoning` (OpenRouter passthrough).
     */
    reasoningEffortFormat?: 'chat' | 'responses';
}

// Type-safe message format for OpenAI API
interface OpenAIChatMessage {
    role: 'system' | 'user' | 'assistant' | 'tool';
    content: ChatMessageContent;
    tool_calls?: ToolCall[];
    tool_call_id?: string;
}

/**
 * Represents a tool call chunk received during streaming
 */
export interface ToolCallChunk {
    id: string;
    name: string;
    arguments: string;
}

/**
 * Represents a complete tool call after streaming is finished
 */
export interface CompletedToolCall {
    id: string;
    name: string;
    arguments: string;
}

/**
 * Represents a thinking tag pair (start and end tags).
 */
interface ThinkingTagPair {
    tagName: 'think' | 'thinking';
    startTag: string;  // lowercase start tag, e.g., '<think>'
    endTag: string;    // lowercase end tag, e.g., '</think>'
    handling: 'thinking' | 'drop';
    onlyAtStart: boolean;  // if true, only match at the beginning of the stream (before any text emitted)
    onlyAtLineStart: boolean;  // if true, only match at line start (after \n or at position 0)
    requireNoThinking: boolean;  // if true, only match when no thinking content has been received yet
}

export interface ThinkTagStreamParserOptions {
    /**
     * How to handle `<think>...</think>` blocks (only matched at the beginning of the stream).
     * - 'thinking': forward the inner content to `onThinking`
     * - 'drop': strip the block from visible text and do NOT forward anywhere
     */
    thinkTagHandling?: 'thinking' | 'drop';
    /**
     * How to handle `<thinking>...</thinking>` blocks (matched at line start).
     */
    thinkingTagHandling?: 'thinking' | 'drop';
}

/**
 * Parses model output that embeds chain-of-thought inside thinking tags.
 *
 * Supported tag formats (case-insensitive):
 * - <think>...</think> - only matches at the beginning of the stream, and only when
 *   no thinking content has been received yet (via onThinking or reasoning_content)
 * - <thinking>...</thinking> - matches at line start (after \n or at position 0)
 *
 * Some OpenAI-compatible providers/models do not use a separate `reasoning_content` field,
 * and instead prepend the assistant content with thinking blocks.
 *
 * This parser is streaming-safe: tags may be split across chunks.
 *
 * Behavior:
 * - If `onThinking` is provided, content inside thinking tags is sent to `onThinking`
 *   and is NOT forwarded to `onText`.
 * - If `onThinking` is not provided, all input is forwarded to `onText` unchanged.
 * - Nested tags are NOT supported: inner tags are treated as literal text content.
 *   e.g., `<thinking><thinking></thinking>` -> thinking = "<thinking>", text = ""
 * - Unmatched closing tags are passed through as text.
 *   e.g., `<thinking></thinking></thinking>` -> thinking = "", text = "</thinking>"
 */
export class ThinkTagStreamParser {
    private carry = '';
    private inThink = false;
    private currentEndTag = '';  // The end tag we're looking for when inside a thinking block
    private currentHandling: 'thinking' | 'drop' = 'thinking';
    private hasEmittedText = false;  // Track if any text has been emitted (for onlyAtStart tags)
    private hasReceivedThinking = false;  // Track if any thinking content has been received

    private readonly tagPairs: ThinkingTagPair[];

    // Supported thinking tag pairs (order matters - longer tags should come first for proper matching)
    private static readonly thinkingTags: ThinkingTagPair[] = [
        { tagName: 'thinking', startTag: '<thinking>', endTag: '</thinking>', handling: 'thinking', onlyAtStart: false, onlyAtLineStart: true, requireNoThinking: false },
        { tagName: 'think', startTag: '<think>', endTag: '</think>', handling: 'thinking', onlyAtStart: true, onlyAtLineStart: false, requireNoThinking: true },
    ];

    // Longest possible start tag prefix for carry handling
    private static readonly maxStartTagLength = Math.max(
        ...ThinkTagStreamParser.thinkingTags.map(t => t.startTag.length)
    );

    constructor(
        private readonly handlers: {
            onText?: (chunk: string) => void;
            onThinking?: (chunk: string) => void;
        },
        options?: ThinkTagStreamParserOptions
    ) {
        // Apply per-tag handling overrides.
        this.tagPairs = ThinkTagStreamParser.thinkingTags.map((tp) => {
            if (tp.tagName === 'think' && options?.thinkTagHandling) {
                return { ...tp, handling: options.thinkTagHandling };
            }
            if (tp.tagName === 'thinking' && options?.thinkingTagHandling) {
                return { ...tp, handling: options.thinkingTagHandling };
            }
            return tp;
        });
    }

    /**
     * Notify the parser that thinking content has been received from an external source
     * (e.g., reasoning_content field). This disables <think> tag matching.
     */
    notifyThinkingReceived(): void {
        this.hasReceivedThinking = true;
    }

    ingest(fragment: string): void {
        if (!fragment) {
            return;
        }

        // If the consumer doesn't support thinking parts, do not strip tags.
        if (!this.handlers.onThinking) {
            this.handlers.onText?.(fragment);
            return;
        }

        let text = this.carry + fragment;
        this.carry = '';

        while (text.length > 0) {
            const lower = text.toLowerCase();

            if (this.inThink) {
                // Looking for the matching end tag
                const endIdx = lower.indexOf(this.currentEndTag);
                if (endIdx === -1) {
                    const split = this.splitKeepingPossibleTagPrefix(text, this.currentEndTag);
                    if (split.emit) {
                        if (this.currentHandling === 'thinking') {
                            this.handlers.onThinking?.(split.emit);
                        }
                        // If currentHandling === 'drop', we intentionally discard it.
                        this.hasReceivedThinking = true;
                    }
                    this.carry = split.carry;
                    return;
                }

                const thinkingPart = text.slice(0, endIdx);
                if (thinkingPart) {
                    if (this.currentHandling === 'thinking') {
                        this.handlers.onThinking?.(thinkingPart);
                    }
                    this.hasReceivedThinking = true;
                }

                text = text.slice(endIdx + this.currentEndTag.length);
                this.inThink = false;
                this.currentEndTag = '';
                this.currentHandling = 'thinking';
                continue;
            }

            // Look for any start tag, find the earliest one
            // Note: <think> only matches at position 0 when no text emitted and no thinking received
            // <thinking> matches at line start (after \n or at position 0)
            let earliestIdx = -1;
            let matchedTag: ThinkingTagPair | null = null;

            for (const tagPair of this.tagPairs) {
                // Skip onlyAtStart tags if we've already emitted text
                if (tagPair.onlyAtStart && this.hasEmittedText) {
                    continue;
                }

                // Skip requireNoThinking tags if thinking content has already been received
                if (tagPair.requireNoThinking && this.hasReceivedThinking) {
                    continue;
                }

                // Find all occurrences and check position constraints
                let searchStart = 0;
                while (searchStart < lower.length) {
                    const idx = lower.indexOf(tagPair.startTag, searchStart);
                    if (idx === -1) {
                        break;
                    }

                    // For onlyAtStart tags, only match at position 0
                    if (tagPair.onlyAtStart && idx !== 0) {
                        break;  // No point searching further
                    }

                    // For onlyAtLineStart tags, check if at line start
                    if (tagPair.onlyAtLineStart && idx !== 0) {
                        // Must be preceded by a newline
                        if (text[idx - 1] !== '\n') {
                            searchStart = idx + 1;
                            continue;
                        }
                    }

                    // Valid match found
                    if (earliestIdx === -1 || idx < earliestIdx) {
                        earliestIdx = idx;
                        matchedTag = tagPair;
                    }
                    break;
                }
            }

            if (earliestIdx === -1 || !matchedTag) {
                // No start tag found, but keep potential partial tag in carry
                const split = this.splitKeepingPossibleStartTagPrefix(text);
                if (split.emit) {
                    this.handlers.onText?.(split.emit);
                    this.hasEmittedText = true;
                }
                this.carry = split.carry;
                return;
            }

            const visiblePart = text.slice(0, earliestIdx);
            if (visiblePart) {
                this.handlers.onText?.(visiblePart);
                this.hasEmittedText = true;
            }

            text = text.slice(earliestIdx + matchedTag.startTag.length);
            this.inThink = true;
            this.currentEndTag = matchedTag.endTag;
            this.currentHandling = matchedTag.handling;
        }
    }

    flush(): void {
        if (!this.carry) {
            return;
        }

        // If no thinking handler, carry would never be used, but be safe.
        if (!this.handlers.onThinking) {
            this.handlers.onText?.(this.carry);
            this.hasEmittedText = true;
            this.carry = '';
            return;
        }

        if (this.inThink) {
            if (this.currentHandling === 'thinking') {
                this.handlers.onThinking?.(this.carry);
            }
            this.hasReceivedThinking = true;
        } else {
            this.handlers.onText?.(this.carry);
            this.hasEmittedText = true;
        }
        this.carry = '';
    }

    /**
     * Checks if text ends with a possible prefix of any applicable start tag.
     * Returns the split point to keep potential partial tags in carry.
     * Only considers tags that are still applicable (e.g., onlyAtStart tags
     * are skipped if text has already been emitted).
     */
    private splitKeepingPossibleStartTagPrefix(text: string): { emit: string; carry: string } {
        const lower = text.toLowerCase();
        const maxPrefixLen = Math.min(ThinkTagStreamParser.maxStartTagLength - 1, text.length);

        for (let k = maxPrefixLen; k > 0; k--) {
            const suffix = lower.slice(-k);
            // Check if this suffix is a prefix of any applicable start tag
            for (const tagPair of this.tagPairs) {
                // Skip onlyAtStart tags if we've already emitted text
                if (tagPair.onlyAtStart && this.hasEmittedText) {
                    continue;
                }
                // Skip requireNoThinking tags if thinking content has already been received
                if (tagPair.requireNoThinking && this.hasReceivedThinking) {
                    continue;
                }
                if (tagPair.startTag.startsWith(suffix)) {
                    return {
                        emit: text.slice(0, text.length - k),
                        carry: text.slice(text.length - k)
                    };
                }
            }
        }

        return { emit: text, carry: '' };
    }

    private splitKeepingPossibleTagPrefix(text: string, tagLower: string): { emit: string; carry: string } {
        const lower = text.toLowerCase();
        const max = Math.min(tagLower.length - 1, text.length);

        for (let k = max; k > 0; k--) {
            if (tagLower.startsWith(lower.slice(-k))) {
                return {
                    emit: text.slice(0, text.length - k),
                    carry: text.slice(text.length - k)
                };
            }
        }

        return { emit: text, carry: '' };
    }
}

export interface StreamOptions {
    onChunk?: (chunk: string) => void;
    /**
     * Called when a thinking/reasoning content chunk is received.
     * Some models (e.g., DeepSeek) return chain-of-thought reasoning in a separate field.
     */
    onThinkingChunk?: (chunk: string, metadata?: Record<string, unknown>) => void;
    /**
     * @deprecated Use onToolCallsComplete for batch reporting of all tool calls
     */
    onToolCall?: (toolCall: ToolCallChunk) => void;
    /**
     * Called once when streaming is complete with all tool calls from this response.
     * This is the preferred way to handle tool calls as it ensures all tool calls
     * are reported together in a single batch.
     */
    onToolCallsComplete?: (toolCalls: CompletedToolCall[]) => void;
    /** Called with token usage statistics after the API response completes. */
    onUsage?: (usage: OpenAIUsage) => void;
    /** Called with response metadata surfaced by OpenAI-compatible streaming chunks. */
    onResponseMetadata?: (metadata: OpenAIResponseMetadata) => void;
    signal?: AbortSignal;
    tools?: ToolDefinition[];
    toolChoice?: ToolChoice;
    /** Optional max tokens for completion generation (mapped to OpenAI `max_tokens`). */
    maxTokens?: number;
    /** Optional OpenAI-compatible request parameters derived from model metadata/options. */
    requestOptions?: OpenAIRequestOptions;
    /**
     * When true, use the OpenAI Responses API instead of Chat Completions.
     */
    useResponsesApi?: boolean;
    /**
     * When enabled, suppress chain-of-thought transmission:
     * - strips leading `<think>...</think>` blocks from visible output (does not forward them)
     * - does NOT forward `reasoning_content`/`reasoning`/`thinking` fields
     *
     * Note: `<thinking>...</thinking>` blocks are still forwarded as thinking content.
     */
    suppressChainOfThought?: boolean;
}

export class OpenAIClient {
    private client: OpenAI;
    private config: OpenAIConfig;

    private coerceThinkingText(value: unknown): string | undefined {
        if (typeof value === 'string') {
            return value;
        }
        if (Array.isArray(value) && value.every(v => typeof v === 'string')) {
            return (value as string[]).join('');
        }
        return undefined;
    }

    private getRecord(value: unknown): Record<string, unknown> | undefined {
        return typeof value === 'object' && value !== null
            ? value as Record<string, unknown>
            : undefined;
    }

    private getFiniteNumber(value: unknown): number | undefined {
        return typeof value === 'number' && Number.isFinite(value)
            ? value
            : undefined;
    }

    private readNumber(value: unknown, key: string): number | undefined {
        return this.getFiniteNumber(this.getRecord(value)?.[key]);
    }

    private readNumberFromKeys(value: unknown, keys: string[]): number | undefined {
        for (const key of keys) {
            const numberValue = this.readNumber(value, key);
            if (numberValue !== undefined) {
                return numberValue;
            }
        }
        return undefined;
    }

    private readRecordFromKeys(value: unknown, keys: string[]): Record<string, unknown> | undefined {
        const record = this.getRecord(value);
        if (!record) {
            return undefined;
        }
        for (const key of keys) {
            const candidate = this.getRecord(record[key]);
            if (candidate) {
                return candidate;
            }
        }
        return undefined;
    }

    private buildPromptTokenDetails(value: unknown): OpenAIUsage['prompt_tokens_details'] | undefined {
        const details = this.getRecord(value);
        if (!details) {
            return undefined;
        }

        const cachedTokens = this.getFiniteNumber(details['cached_tokens'])
            ?? this.getFiniteNumber(details['cached_input_tokens'])
            ?? 0;
        const cacheCreationInputTokens = this.getFiniteNumber(details['cache_creation_input_tokens']);
        const cacheCreationTokens = this.getFiniteNumber(details['cache_creation_tokens']);

        return {
            cached_tokens: cachedTokens,
            ...(cacheCreationTokens !== undefined ? { cache_creation_tokens: cacheCreationTokens } : {}),
            ...(cacheCreationInputTokens !== undefined ? { cache_creation_input_tokens: cacheCreationInputTokens } : {})
        };
    }

    private buildCompletionTokenDetails(value: unknown): OpenAIUsage['completion_tokens_details'] | undefined {
        const details = this.getRecord(value);
        if (!details) {
            return undefined;
        }

        const reasoningTokens = this.getFiniteNumber(details['reasoning_tokens']);
        const acceptedPredictionTokens = this.getFiniteNumber(details['accepted_prediction_tokens']);
        const rejectedPredictionTokens = this.getFiniteNumber(details['rejected_prediction_tokens']);

        if (
            reasoningTokens === undefined &&
            acceptedPredictionTokens === undefined &&
            rejectedPredictionTokens === undefined
        ) {
            return undefined;
        }

        return {
            reasoning_tokens: reasoningTokens ?? 0,
            accepted_prediction_tokens: acceptedPredictionTokens ?? 0,
            rejected_prediction_tokens: rejectedPredictionTokens ?? 0
        };
    }

    private buildUsage(
        promptTokens: number | undefined,
        completionTokens: number | undefined,
        totalTokens: number | undefined,
        promptDetails?: unknown,
        completionDetails?: unknown
    ): OpenAIUsage | undefined {
        if (promptTokens === undefined || completionTokens === undefined) {
            return undefined;
        }

        const usage: OpenAIUsage = {
            prompt_tokens: promptTokens,
            completion_tokens: completionTokens,
            total_tokens: totalTokens ?? promptTokens + completionTokens
        };

        const normalizedPromptDetails = this.buildPromptTokenDetails(promptDetails);
        if (normalizedPromptDetails) {
            usage.prompt_tokens_details = normalizedPromptDetails;
        }

        const normalizedCompletionDetails = this.buildCompletionTokenDetails(completionDetails);
        if (normalizedCompletionDetails) {
            usage.completion_tokens_details = normalizedCompletionDetails;
        }

        return usage;
    }

    private buildChatUsage(value: unknown): OpenAIUsage | undefined {
        return this.buildUsage(
            this.readNumberFromKeys(value, ['prompt_tokens', 'input_tokens', 'promptTokens', 'inputTokens']),
            this.readNumberFromKeys(value, ['completion_tokens', 'output_tokens', 'completionTokens', 'outputTokens']),
            this.readNumberFromKeys(value, ['total_tokens', 'totalTokens']),
            this.readRecordFromKeys(value, ['prompt_tokens_details', 'input_tokens_details', 'promptTokensDetails', 'inputTokensDetails']),
            this.readRecordFromKeys(value, ['completion_tokens_details', 'output_tokens_details', 'completionTokensDetails', 'outputTokensDetails'])
        );
    }

    private buildResponsesUsage(value: unknown): OpenAIUsage | undefined {
        return this.buildUsage(
            this.readNumberFromKeys(value, ['input_tokens', 'prompt_tokens', 'inputTokens', 'promptTokens']),
            this.readNumberFromKeys(value, ['output_tokens', 'completion_tokens', 'outputTokens', 'completionTokens']),
            this.readNumberFromKeys(value, ['total_tokens', 'totalTokens']),
            this.readRecordFromKeys(value, ['input_tokens_details', 'prompt_tokens_details', 'inputTokensDetails', 'promptTokensDetails']),
            this.readRecordFromKeys(value, ['output_tokens_details', 'completion_tokens_details', 'outputTokensDetails', 'completionTokensDetails'])
        );
    }

    private extractReasoningMetadata(value: unknown): Record<string, unknown> | undefined {
        const record = this.getRecord(value);
        if (!record) {
            return undefined;
        }

        const metadata: Record<string, unknown> = {};
        for (const key of ['reasoning_details', 'thinking_details', 'reasoning_signature', 'signature']) {
            if (record[key] !== undefined) {
                metadata[key] = record[key];
            }
        }
        return Object.keys(metadata).length > 0 ? metadata : undefined;
    }

    private extractResponsesEventMetadata(value: unknown): Record<string, unknown> | undefined {
        const record = this.getRecord(value);
        if (!record) {
            return undefined;
        }

        const metadata: Record<string, unknown> = {};
        for (const key of ['type', 'item_id', 'output_index', 'content_index', 'sequence_number']) {
            if (record[key] !== undefined) {
                metadata[key] = record[key];
            }
        }
        return Object.keys(metadata).length > 0 ? metadata : undefined;
    }

    private extractChatResponseMetadata(value: unknown): OpenAIResponseMetadata {
        const record = this.getRecord(value);
        if (!record) {
            return {};
        }

        const choice0 = Array.isArray(record['choices']) ? this.getRecord(record['choices'][0]) : undefined;
        const metadata: OpenAIResponseMetadata = {};
        if (typeof record['id'] === 'string') {
            metadata.id = record['id'];
        }
        if (typeof record['object'] === 'string') {
            metadata.object = record['object'];
        }
        if (typeof record['model'] === 'string') {
            metadata.model = record['model'];
        }
        const created = this.getFiniteNumber(record['created']);
        if (created !== undefined) {
            metadata.created = created;
        }
        if (typeof record['system_fingerprint'] === 'string') {
            metadata.system_fingerprint = record['system_fingerprint'];
        }
        if (typeof record['service_tier'] === 'string') {
            metadata.service_tier = record['service_tier'];
        }
        if (choice0 && typeof choice0['finish_reason'] === 'string') {
            metadata.finish_reason = choice0['finish_reason'];
        }
        return metadata;
    }

    private extractResponsesResponseMetadata(value: unknown): OpenAIResponseMetadata {
        const record = this.getRecord(value);
        if (!record) {
            return {};
        }

        const metadata: OpenAIResponseMetadata = {};
        if (typeof record['id'] === 'string') {
            metadata.id = record['id'];
        }
        if (typeof record['object'] === 'string') {
            metadata.object = record['object'];
        }
        if (typeof record['model'] === 'string') {
            metadata.model = record['model'];
        }
        const created = this.getFiniteNumber(record['created_at']) ?? this.getFiniteNumber(record['created']);
        if (created !== undefined) {
            metadata.created = created;
        }
        if (typeof record['service_tier'] === 'string') {
            metadata.service_tier = record['service_tier'];
        }
        if (typeof record['status'] === 'string') {
            metadata.status = record['status'];
        }
        if (record['error'] !== undefined && record['error'] !== null) {
            metadata.error = record['error'];
        }
        if (record['incomplete_details'] !== undefined && record['incomplete_details'] !== null) {
            metadata.incomplete_details = record['incomplete_details'];
        }
        return metadata;
    }

    private mergeResponseMetadata(
        current: OpenAIResponseMetadata,
        next: OpenAIResponseMetadata
    ): OpenAIResponseMetadata {
        return {
            ...current,
            ...Object.fromEntries(
                Object.entries(next).filter(([, value]) => value !== undefined)
            )
        };
    }

    constructor(config: OpenAIConfig) {
        this.config = config;
        const normalizedEndpoint = normalizeApiEndpoint(config.apiEndpoint);
        this.client = new OpenAI({
            apiKey: config.apiKey,
            baseURL: normalizedEndpoint,
            dangerouslyAllowBrowser: false
        });
    }

    async listModels(): Promise<APIModelInfo[]> {
        try {
            const response = await this.client.models.list();
            // Cast to APIModelInfo to preserve extended fields from some providers
            const models = response.data.map(model => model as unknown as APIModelInfo);
            return models;
        } catch (error) {
            const e = error as Record<string, unknown>;
            logger.error('Failed to list models', error, 'OpenAI');
            logger.debug('listModels error details', {
                status: e?.status ?? (e?.response as Record<string, unknown>)?.status,
                code: e?.code ?? (e?.error as Record<string, unknown>)?.code,
                name: e?.name,
                message: (e?.error as Record<string, unknown>)?.message ?? e?.message
            }, 'OpenAI');
            throw new Error(`Failed to fetch models from API: ${error}`);
        }
    }

    async createChatCompletion(
        messages: ChatMessage[],
        model: string,
        options?: {
            maxTokens?: number;
            temperature?: number;
            stream?: boolean;
        }
    ): Promise<string> {
        // Convert to OpenAI message format
        const openaiMessages = this.convertMessagesToOpenAIFormat(messages);

        try {
            const response = await this.client.chat.completions.create({
                model: model,
                messages: openaiMessages,
                max_tokens: options?.maxTokens,
                temperature: options?.temperature ?? 0.7,
                stream: false
            });

            return response.choices[0]?.message?.content || '';
        } catch (error) {
            const e = error as Record<string, unknown>;
            logger.error('Failed to create chat completion', error, 'OpenAI');
            logger.debug('createChatCompletion error details', {
                model,
                status: e?.status ?? (e?.response as Record<string, unknown>)?.status,
                code: e?.code ?? (e?.error as Record<string, unknown>)?.code,
                name: e?.name,
                message: (e?.error as Record<string, unknown>)?.message ?? e?.message
            }, 'OpenAI');
            throw new Error(`Failed to create chat completion: ${error}`);
        }
    }

    async streamChatCompletion(
        messages: ChatMessage[],
        model: string,
        streamOptions: StreamOptions
    ): Promise<string> {
        if (streamOptions.useResponsesApi) {
            return this.streamResponsesCompletion(messages, model, streamOptions);
        }

        let fullContent = '';
        let thinkingChars = 0;
        let sawAnyModelOutput = false;

        const thinkTagParser = new ThinkTagStreamParser(
            {
            onText: (chunk) => {
                fullContent += chunk;
                streamOptions.onChunk?.(chunk);
            },
            onThinking: streamOptions.onThinkingChunk
                ? (chunk) => {
                    thinkingChars += chunk.length;
                    streamOptions.onThinkingChunk?.(chunk);
                }
                : undefined
            },
            {
                thinkTagHandling: streamOptions.suppressChainOfThought ? 'drop' : 'thinking'
            }
        );

        // Convert to OpenAI message format
        const openaiMessages = this.convertMessagesToOpenAIFormat(messages);

        try {
            const maxTokens = (typeof streamOptions.maxTokens === 'number' && streamOptions.maxTokens > 0)
                ? streamOptions.maxTokens
                : 2048;
            const requestOptionsFromCaller = streamOptions.requestOptions ?? {};

            // Build request options
            const requestOptions: OpenAI.Chat.ChatCompletionCreateParamsStreaming = {
                model: model,
                messages: openaiMessages,
                stream: true,
                stream_options: { include_usage: true },
                temperature: requestOptionsFromCaller.temperature ?? 1.0,
                max_tokens: maxTokens
            };
            if (requestOptionsFromCaller.topP !== undefined) {
                requestOptions.top_p = requestOptionsFromCaller.topP;
            }
            if (requestOptionsFromCaller.stop !== undefined) {
                requestOptions.stop = requestOptionsFromCaller.stop;
            }
            this.applyOpenAICompatibleRequestExtras(requestOptions, requestOptionsFromCaller);

            // Add tools if provided
            if (streamOptions.tools && streamOptions.tools.length > 0) {
                requestOptions.tools = streamOptions.tools;
                if (streamOptions.toolChoice) {
                    requestOptions.tool_choice = streamOptions.toolChoice;
                }
            }

            const stream = await this.client.chat.completions.create(requestOptions);

            // Track tool calls being assembled from streamed chunks
            const toolCallsInProgress: Map<number, { id: string; name: string; arguments: string }> = new Map();

            let chunkCount = 0;
            let finishReason: string | null = null;
            let streamUsage: OpenAIUsage | undefined;
            let responseMetadata: OpenAIResponseMetadata = {};

            for await (const chunk of stream) {
                chunkCount++;
                if (streamOptions.signal?.aborted) {
                    break;
                }

                responseMetadata = this.mergeResponseMetadata(
                    responseMetadata,
                    this.extractChatResponseMetadata(chunk)
                );

                if (chunk.usage) {
                    streamUsage = this.buildChatUsage(chunk.usage);
                }

                const choice0 = chunk.choices[0];
                finishReason = (choice0 as any)?.finish_reason ?? finishReason;
                const delta = choice0?.delta;

                // Some gateways put fields on `choices[0].message` instead of `delta`.
                const messageAny = (choice0 as any)?.message as Record<string, unknown> | undefined;
                const messageContent = messageAny?.content;
                if (typeof messageContent === 'string' && messageContent.length > 0) {
                    sawAnyModelOutput = true;
                    thinkTagParser.ingest(messageContent);
                }
                const messageRefusal = messageAny?.refusal;
                if (typeof messageRefusal === 'string' && messageRefusal.length > 0) {
                    sawAnyModelOutput = true;
                    thinkTagParser.ingest(messageRefusal);
                }

                const messageReasoningRaw = (messageAny as any)?.reasoning_content ?? (messageAny as any)?.reasoning ?? (messageAny as any)?.thinking;
                const messageReasoning = this.coerceThinkingText(messageReasoningRaw);
                if (messageReasoning && messageReasoning.length > 0) {
                    sawAnyModelOutput = true;
                    if (!streamOptions.suppressChainOfThought) {
                        thinkingChars += messageReasoning.length;
                        thinkTagParser.notifyThinkingReceived();
                        streamOptions.onThinkingChunk?.(messageReasoning, this.extractReasoningMetadata(messageAny));
                    }
                }

                const messageToolCalls = messageAny?.tool_calls;
                if (Array.isArray(messageToolCalls) && messageToolCalls.length > 0) {
                    sawAnyModelOutput = true;
                    for (let i = 0; i < messageToolCalls.length; i++) {
                        const tc: any = messageToolCalls[i];
                        const index = i;
                        let toolCall = toolCallsInProgress.get(index);
                        if (!toolCall) {
                            toolCall = {
                                id: tc?.id || '',
                                name: tc?.function?.name || '',
                                arguments: ''
                            };
                            toolCallsInProgress.set(index, toolCall);
                        }
                        if (tc?.id) {
                            toolCall.id = tc.id;
                        }
                        if (tc?.function?.name) {
                            toolCall.name = tc.function.name;
                        }
                        if (typeof tc?.function?.arguments === 'string') {
                            toolCall.arguments = tc.function.arguments;
                        }
                    }
                }

                // Handle thinking/reasoning content (chain-of-thought)
                // Some models (e.g., DeepSeek) return reasoning in a separate `reasoning_content` field
                const deltaAny = delta as Record<string, unknown> | undefined;
                const reasoningRaw = (deltaAny as any)?.reasoning_content ?? (deltaAny as any)?.reasoning ?? (deltaAny as any)?.thinking;
                const reasoningContent = this.coerceThinkingText(reasoningRaw);
                if (reasoningContent && reasoningContent.length > 0) {
                    sawAnyModelOutput = true;
                    if (!streamOptions.suppressChainOfThought) {
                        thinkingChars += reasoningContent.length;
                        thinkTagParser.notifyThinkingReceived();
                        streamOptions.onThinkingChunk?.(reasoningContent, this.extractReasoningMetadata(deltaAny));
                    }
                }

                // Handle text content
                const content = delta?.content || '';
                if (content) {
                    sawAnyModelOutput = true;
                    // Some models embed thinking in <think>...</think> inside the normal content stream.
                    // Parse and route those parts to onThinkingChunk when available.
                    thinkTagParser.ingest(content);
                }
                const refusal = (delta as { refusal?: unknown } | undefined)?.refusal;
                if (typeof refusal === 'string' && refusal.length > 0) {
                    sawAnyModelOutput = true;
                    thinkTagParser.ingest(refusal);
                }

                // Handle tool calls in streaming response
                if (delta?.tool_calls) {
                    if (delta.tool_calls.length > 0) {
                        sawAnyModelOutput = true;
                    }
                    for (const toolCallDelta of delta.tool_calls) {
                        const index = toolCallDelta.index;

                        // Get or create the tool call being assembled
                        let toolCall = toolCallsInProgress.get(index);
                        if (!toolCall) {
                            toolCall = {
                                id: toolCallDelta.id || '',
                                name: toolCallDelta.function?.name || '',
                                arguments: ''
                            };
                            toolCallsInProgress.set(index, toolCall);
                        }

                        // Update with new data from this chunk
                        if (toolCallDelta.id) {
                            toolCall.id = toolCallDelta.id;
                        }
                        if (toolCallDelta.function?.name) {
                            toolCall.name = toolCallDelta.function.name;
                        }
                        if (toolCallDelta.function?.arguments) {
                            toolCall.arguments += toolCallDelta.function.arguments;
                        }

                        // Legacy: Report incremental updates if onToolCall is provided
                        if (streamOptions.onToolCall) {
                            streamOptions.onToolCall({
                                id: toolCall.id,
                                name: toolCall.name,
                                arguments: toolCall.arguments
                            });
                        }
                    }
                }

                // Flush tool calls as soon as the stream signals completion (finish_reason === 'tool_calls').
                // Reporting during streaming (before VSCode finalizes the response) ensures the
                // LanguageModelToolCallPart parts are persisted into the response rather than
                // being dropped as late-arriving parts after stream finalize.
                if (finishReason === 'tool_calls' && toolCallsInProgress.size > 0 && streamOptions.onToolCallsComplete) {
                    const flushed: CompletedToolCall[] = [];
                    const flushedSeen = new Set<string>();
                    const sortedFlushed = Array.from(toolCallsInProgress.entries()).sort((a, b) => a[0] - b[0]);
                    for (const [, toolCall] of sortedFlushed) {
                        if (toolCall.id && toolCall.name && !flushedSeen.has(toolCall.id)) {
                            flushedSeen.add(toolCall.id);
                            flushed.push({
                                id: toolCall.id,
                                name: toolCall.name,
                                arguments: toolCall.arguments
                            });
                        }
                    }
                    if (flushed.length > 0) {
                        streamOptions.onToolCallsComplete(flushed);
                    }
                    toolCallsInProgress.clear();
                }
            }

            // Flush any pending partial tag/text at end of stream.
            thinkTagParser.flush();

            // Report all completed tool calls at once after streaming is done.
            // NOTE: if tool calls were already flushed during streaming (finish_reason === 'tool_calls'),
            // toolCallsInProgress has been cleared and this block is a no-op.
            const completedToolCalls: CompletedToolCall[] = [];
            if (toolCallsInProgress.size > 0) {
                const seenIds = new Set<string>();
                // Sort by index to maintain order
                const sortedEntries = Array.from(toolCallsInProgress.entries()).sort((a, b) => a[0] - b[0]);
                for (const [, toolCall] of sortedEntries) {
                    // Deduplicate by tool call ID to prevent duplicate reporting
                    if (toolCall.id && toolCall.name && !seenIds.has(toolCall.id)) {
                        seenIds.add(toolCall.id);
                        completedToolCalls.push({
                            id: toolCall.id,
                            name: toolCall.name,
                            arguments: toolCall.arguments
                        });
                    }
                }
            }
            if (streamOptions.onToolCallsComplete && completedToolCalls.length > 0) {
                streamOptions.onToolCallsComplete(completedToolCalls);
            }

            // If the stream produced nothing at all (no text/thinking/tool calls), fall back to non-streaming once.
            if (!sawAnyModelOutput && completedToolCalls.length === 0 && !streamOptions.signal?.aborted) {
                logger.warn('Empty streaming response detected; falling back to non-streaming', 'OpenAI');
                logger.debug('Empty streaming response details', {
                    model,
                    chunkCount,
                    finishReason
                }, 'OpenAI');

                const fallbackRequest: OpenAI.Chat.ChatCompletionCreateParamsNonStreaming = {
                    model: model,
                    messages: openaiMessages,
                    temperature: requestOptionsFromCaller.temperature ?? 0.7,
                    max_tokens: maxTokens,
                    stream: false,
                    ...(streamOptions.tools && streamOptions.tools.length > 0 ? { tools: streamOptions.tools } : {}),
                    ...(streamOptions.toolChoice ? { tool_choice: streamOptions.toolChoice } : {})
                };
                if (requestOptionsFromCaller.topP !== undefined) {
                    fallbackRequest.top_p = requestOptionsFromCaller.topP;
                }
                if (requestOptionsFromCaller.stop !== undefined) {
                    fallbackRequest.stop = requestOptionsFromCaller.stop;
                }
                this.applyOpenAICompatibleRequestExtras(fallbackRequest, requestOptionsFromCaller);

                const response = await this.client.chat.completions.create(fallbackRequest) as OpenAI.Chat.ChatCompletion;

                responseMetadata = this.mergeResponseMetadata(
                    responseMetadata,
                    this.extractChatResponseMetadata(response)
                );

                if (response.usage && streamOptions.onUsage) {
                    streamUsage = this.buildChatUsage(response.usage);
                }

                const msgAny = response.choices?.[0]?.message as unknown as Record<string, unknown> | undefined;
                const nonStreamContent = msgAny?.content;
                if (typeof nonStreamContent === 'string' && nonStreamContent.length > 0) {
                    sawAnyModelOutput = true;
                    thinkTagParser.ingest(nonStreamContent);
                }
                const nonStreamRefusal = msgAny?.refusal;
                if (typeof nonStreamRefusal === 'string' && nonStreamRefusal.length > 0) {
                    sawAnyModelOutput = true;
                    thinkTagParser.ingest(nonStreamRefusal);
                }

                const nonStreamReasoningRaw = msgAny?.reasoning_content ?? msgAny?.reasoning ?? msgAny?.thinking;
                const nonStreamReasoning = this.coerceThinkingText(nonStreamReasoningRaw);
                if (nonStreamReasoning && nonStreamReasoning.length > 0) {
                    sawAnyModelOutput = true;
                    if (!streamOptions.suppressChainOfThought) {
                        thinkingChars += nonStreamReasoning.length;
                        streamOptions.onThinkingChunk?.(nonStreamReasoning, this.extractReasoningMetadata(msgAny));
                    }
                }

                const nonStreamToolCalls = msgAny?.tool_calls;
                if (Array.isArray(nonStreamToolCalls) && nonStreamToolCalls.length > 0 && streamOptions.onToolCallsComplete) {
                    sawAnyModelOutput = true;
                    const mapped: CompletedToolCall[] = nonStreamToolCalls
                        .map((tc: Record<string, unknown>) => {
                            const tcFunction = tc?.function as Record<string, unknown> | undefined;
                            return {
                                id: (tc?.id as string) || '',
                                name: (tcFunction?.name as string) || '',
                                arguments: typeof tcFunction?.arguments === 'string' ? tcFunction.arguments : ''
                            };
                        })
                        .filter((tc: CompletedToolCall) => tc.id && tc.name);
                    if (mapped.length > 0) {
                        streamOptions.onToolCallsComplete(mapped);
                    }
                }

                thinkTagParser.flush();
            }

            if (streamUsage && streamOptions.onUsage) {
                streamOptions.onUsage(streamUsage);
            }
            if (Object.keys(responseMetadata).length > 0) {
                streamOptions.onResponseMetadata?.(responseMetadata);
            }

            return fullContent;
        } catch (error: unknown) {
            const err = error as Record<string, unknown>;
            if (err?.name === 'AbortError' || streamOptions.signal?.aborted) {
                thinkTagParser.flush();
                return fullContent;
            }

            logger.error('streamChatCompletion failed', error, 'OpenAI');
            logger.debug('streamChatCompletion error details', {
                model,
                messageCount: openaiMessages.length,
                toolsCount: streamOptions.tools?.length ?? 0,
                toolChoice: streamOptions.toolChoice ?? undefined,
                status: err?.status ?? (err?.response as Record<string, unknown>)?.status,
                code: err?.code ?? (err?.error as Record<string, unknown>)?.code,
                name: err?.name,
                message: (err?.error as Record<string, unknown>)?.message ?? err?.message,
                responseData: (err?.response as Record<string, unknown> | undefined)?.data,
                errorDetails: err?.error as Record<string, unknown> | undefined
            }, 'OpenAI');

            throw new Error(`Failed to stream chat completion: ${error}`);
        }
    }

    private async streamResponsesCompletion(
        messages: ChatMessage[],
        model: string,
        streamOptions: StreamOptions
    ): Promise<string> {
        let fullContent = '';
        let sawAnyModelOutput = false;

        const thinkTagParser = new ThinkTagStreamParser(
            {
                onText: (chunk) => {
                    fullContent += chunk;
                    streamOptions.onChunk?.(chunk);
                },
                onThinking: streamOptions.onThinkingChunk
                    ? (chunk) => {
                        streamOptions.onThinkingChunk?.(chunk);
                    }
                    : undefined
            },
            {
                thinkTagHandling: streamOptions.suppressChainOfThought ? 'drop' : 'thinking'
            }
        );

        const responseInput = this.convertMessagesToResponsesInput(messages);
        const responseTools = this.convertToolsToResponsesTools(streamOptions.tools);
        const responseToolChoice = responseTools && responseTools.length > 0
            ? this.convertToolChoiceToResponsesToolChoice(streamOptions.toolChoice)
            : undefined;

        const maxTokens = (typeof streamOptions.maxTokens === 'number' && streamOptions.maxTokens > 0)
            ? streamOptions.maxTokens
            : 2048;
        const requestOptionsFromCaller = streamOptions.requestOptions ?? {};

        const requestOptions: OpenAI.Responses.ResponseCreateParamsStreaming = {
            model,
            input: responseInput,
            stream: true,
            temperature: requestOptionsFromCaller.temperature ?? 1.0,
            max_output_tokens: maxTokens
        };
        if (requestOptionsFromCaller.topP !== undefined) {
            requestOptions.top_p = requestOptionsFromCaller.topP;
        }
        this.applyOpenAICompatibleRequestExtras(requestOptions, requestOptionsFromCaller);

        if (responseTools && responseTools.length > 0) {
            requestOptions.tools = responseTools;
            if (responseToolChoice) {
                requestOptions.tool_choice = responseToolChoice;
            }
        }

        const toolCallsById = new Map<string, { id: string; name: string; arguments: string }>();
        const itemIdToCallId = new Map<string, string>();
        const textDeltaItemIds = new Set<string>();
        const refusalDeltaItemIds = new Set<string>();
        const reasoningDeltaItemIds = new Set<string>();

        const getOrCreateToolCall = (callId: string) => {
            let toolCall = toolCallsById.get(callId);
            if (!toolCall) {
                toolCall = { id: callId, name: '', arguments: '' };
                toolCallsById.set(callId, toolCall);
            }
            return toolCall;
        };

        const moveToolCall = (fromId: string, toId: string) => {
            if (fromId === toId) {
                return;
            }
            const existing = toolCallsById.get(fromId);
            if (!existing) {
                return;
            }
            toolCallsById.delete(fromId);
            toolCallsById.set(toId, { ...existing, id: toId });
        };

        const updateToolCallFromItem = (item: OpenAI.Responses.ResponseFunctionToolCall) => {
            const callId = item.call_id || item.id || '';
            if (!callId) {
                return;
            }

            if (item.id) {
                const previous = itemIdToCallId.get(item.id);
                itemIdToCallId.set(item.id, callId);
                if (previous && previous !== callId) {
                    moveToolCall(previous, callId);
                } else if (item.id !== callId) {
                    moveToolCall(item.id, callId);
                }
            }

            const toolCall = getOrCreateToolCall(callId);
            if (item.name) {
                toolCall.name = item.name;
            }
            if (typeof item.arguments === 'string') {
                toolCall.arguments = item.arguments;
            }

            if (streamOptions.onToolCall && toolCall.name) {
                streamOptions.onToolCall({
                    id: toolCall.id,
                    name: toolCall.name,
                    arguments: toolCall.arguments
                });
            }
            sawAnyModelOutput = true;
        };

        let chunkCount = 0;
        let responseMetadata: OpenAIResponseMetadata = {};

        try {
            const stream = await this.client.responses.create(requestOptions);

            for await (const event of stream as AsyncIterable<OpenAI.Responses.ResponseStreamEvent>) {
                chunkCount++;
                if (streamOptions.signal?.aborted) {
                    break;
                }

                switch (event.type) {
                    case 'response.output_text.delta': {
                        if (event.delta) {
                            sawAnyModelOutput = true;
                            textDeltaItemIds.add(event.item_id);
                            thinkTagParser.ingest(event.delta);
                        }
                        break;
                    }
                    case 'response.output_text.done': {
                        if (!textDeltaItemIds.has(event.item_id) && event.text) {
                            sawAnyModelOutput = true;
                            thinkTagParser.ingest(event.text);
                        }
                        break;
                    }
                    case 'response.refusal.delta': {
                        if (event.delta) {
                            sawAnyModelOutput = true;
                            refusalDeltaItemIds.add(event.item_id);
                            thinkTagParser.ingest(event.delta);
                        }
                        break;
                    }
                    case 'response.refusal.done': {
                        if (!refusalDeltaItemIds.has(event.item_id) && event.refusal) {
                            sawAnyModelOutput = true;
                            thinkTagParser.ingest(event.refusal);
                        }
                        break;
                    }
                    case 'response.reasoning_text.delta': {
                        sawAnyModelOutput = true;
                        if (!streamOptions.suppressChainOfThought && event.delta) {
                            reasoningDeltaItemIds.add(event.item_id);
                            thinkTagParser.notifyThinkingReceived();
                            streamOptions.onThinkingChunk?.(event.delta, this.extractResponsesEventMetadata(event));
                        }
                        break;
                    }
                    case 'response.reasoning_text.done': {
                        sawAnyModelOutput = true;
                        if (!streamOptions.suppressChainOfThought && event.text && !reasoningDeltaItemIds.has(event.item_id)) {
                            thinkTagParser.notifyThinkingReceived();
                            streamOptions.onThinkingChunk?.(event.text, this.extractResponsesEventMetadata(event));
                        }
                        break;
                    }
                    case 'response.output_item.added':
                    case 'response.output_item.done': {
                        const outputItem = event.item;
                        if (outputItem?.type === 'function_call') {
                            updateToolCallFromItem(outputItem as OpenAI.Responses.ResponseFunctionToolCall);
                        }
                        break;
                    }
                    case 'response.function_call_arguments.delta': {
                        sawAnyModelOutput = true;
                        const callId = itemIdToCallId.get(event.item_id) ?? event.item_id;
                        if (callId !== event.item_id) {
                            moveToolCall(event.item_id, callId);
                        }
                        const toolCall = getOrCreateToolCall(callId);
                        toolCall.arguments += event.delta ?? '';
                        if (streamOptions.onToolCall && toolCall.name) {
                            streamOptions.onToolCall({
                                id: toolCall.id,
                                name: toolCall.name,
                                arguments: toolCall.arguments
                            });
                        }
                        break;
                    }
                    case 'response.function_call_arguments.done': {
                        sawAnyModelOutput = true;
                        const callId = itemIdToCallId.get(event.item_id) ?? event.item_id;
                        if (callId !== event.item_id) {
                            moveToolCall(event.item_id, callId);
                        }
                        const toolCall = getOrCreateToolCall(callId);
                        if (event.name) {
                            toolCall.name = event.name;
                        }
                        if (typeof event.arguments === 'string') {
                            toolCall.arguments = event.arguments;
                        }
                        if (streamOptions.onToolCall && toolCall.name) {
                            streamOptions.onToolCall({
                                id: toolCall.id,
                                name: toolCall.name,
                                arguments: toolCall.arguments
                            });
                        }
                        break;
                    }
                    case 'response.completed': {
                        const completedResponse = event.response;
                        responseMetadata = this.mergeResponseMetadata(
                            responseMetadata,
                            this.extractResponsesResponseMetadata(completedResponse)
                        );
                        if (completedResponse?.usage && streamOptions.onUsage) {
                            const usage = this.buildResponsesUsage(completedResponse.usage);
                            if (usage) {
                                streamOptions.onUsage(usage);
                            }
                        }
                        break;
                    }
                }
            }

            thinkTagParser.flush();

            const completedToolCalls: CompletedToolCall[] = Array.from(toolCallsById.values())
                .filter(tc => tc.id && tc.name)
                .map(tc => ({
                    id: tc.id,
                    name: tc.name,
                    arguments: tc.arguments
                }));

            if (streamOptions.onToolCallsComplete && completedToolCalls.length > 0) {
                streamOptions.onToolCallsComplete(completedToolCalls);
            }

            if (!sawAnyModelOutput && completedToolCalls.length === 0 && !streamOptions.signal?.aborted) {
                logger.warn('Empty responses stream detected; falling back to non-streaming', 'OpenAI');
                logger.debug('Empty responses stream details', {
                    model,
                    chunkCount
                }, 'OpenAI');

                const fallbackRequest: OpenAI.Responses.ResponseCreateParamsNonStreaming = {
                    model,
                    input: responseInput,
                    stream: false,
                    temperature: requestOptionsFromCaller.temperature ?? 0.7,
                    max_output_tokens: maxTokens
                };
                if (requestOptionsFromCaller.topP !== undefined) {
                    fallbackRequest.top_p = requestOptionsFromCaller.topP;
                }
                this.applyOpenAICompatibleRequestExtras(fallbackRequest, requestOptionsFromCaller);

                if (responseTools && responseTools.length > 0) {
                    fallbackRequest.tools = responseTools;
                    if (responseToolChoice) {
                        fallbackRequest.tool_choice = responseToolChoice;
                    }
                }

                const response = await this.client.responses.create(fallbackRequest) as OpenAI.Responses.Response;

                responseMetadata = this.mergeResponseMetadata(
                    responseMetadata,
                    this.extractResponsesResponseMetadata(response)
                );

                if (response?.usage && streamOptions.onUsage) {
                    const usage = this.buildResponsesUsage(response.usage);
                    if (usage) {
                        streamOptions.onUsage(usage);
                    }
                }

                const nonStreamContent = response?.output_text;
                if (typeof nonStreamContent === 'string' && nonStreamContent.length > 0) {
                    sawAnyModelOutput = true;
                    thinkTagParser.ingest(nonStreamContent);
                }

                const nonStreamToolCalls: CompletedToolCall[] = [];
                const seenToolCallIds = new Set<string>();
                if (Array.isArray(response?.output)) {
                    for (const item of response.output) {
                        if (item?.type === 'function_call') {
                            const toolItem = item as OpenAI.Responses.ResponseFunctionToolCall;
                            const callId = toolItem.call_id || toolItem.id || '';
                            if (!callId || !toolItem.name || seenToolCallIds.has(callId)) {
                                continue;
                            }
                            seenToolCallIds.add(callId);
                            nonStreamToolCalls.push({
                                id: callId,
                                name: toolItem.name,
                                arguments: typeof toolItem.arguments === 'string' ? toolItem.arguments : ''
                            });
                        }
                    }
                }

                if (nonStreamToolCalls.length > 0) {
                    sawAnyModelOutput = true;
                    if (streamOptions.onToolCallsComplete) {
                        streamOptions.onToolCallsComplete(nonStreamToolCalls);
                    }
                }

                thinkTagParser.flush();
            }

            if (Object.keys(responseMetadata).length > 0) {
                streamOptions.onResponseMetadata?.(responseMetadata);
            }

            return fullContent;
        } catch (error: unknown) {
            const err = error as Record<string, unknown>;
            if (err?.name === 'AbortError' || streamOptions.signal?.aborted) {
                thinkTagParser.flush();
                return fullContent;
            }

            logger.error('streamResponsesCompletion failed', error, 'OpenAI');
            logger.debug('streamResponsesCompletion error details', {
                model,
                messageCount: responseInput.length,
                toolsCount: responseTools?.length ?? 0,
                toolChoice: responseToolChoice ?? undefined,
                status: err?.status ?? (err?.response as Record<string, unknown>)?.status,
                code: err?.code ?? (err?.error as Record<string, unknown>)?.code,
                name: err?.name,
                message: (err?.error as Record<string, unknown>)?.message ?? err?.message,
                responseData: (err?.response as Record<string, unknown> | undefined)?.data,
                errorDetails: err?.error as Record<string, unknown> | undefined
            }, 'OpenAI');

            throw new Error(`Failed to stream responses completion: ${error}`);
        }
    }

    updateConfig(config: OpenAIConfig) {
        this.config = config;
        const normalizedEndpoint = normalizeApiEndpoint(config.apiEndpoint);
        this.client = new OpenAI({
            apiKey: config.apiKey,
            baseURL: normalizedEndpoint,
            dangerouslyAllowBrowser: false
        });
    }

    private applyOpenAICompatibleRequestExtras(
        request:
            | OpenAI.Chat.ChatCompletionCreateParamsStreaming
            | OpenAI.Chat.ChatCompletionCreateParamsNonStreaming
            | OpenAI.Responses.ResponseCreateParamsStreaming
            | OpenAI.Responses.ResponseCreateParamsNonStreaming,
        options: OpenAIRequestOptions
    ): void {
        const extras = request as unknown as Record<string, unknown>;
        if (options.stop !== undefined) {
            extras['stop'] = options.stop;
        }
        if (options.reasoning) {
            this.applyReasoningExtras(extras, options.reasoning, options.reasoningEffortFormat);
        }
        if (typeof options.includeReasoning === 'boolean') {
            extras['include_reasoning'] = options.includeReasoning;
        }
        if (options.responseFormat) {
            extras['response_format'] = options.responseFormat;
        }
        if (options.serviceTier) {
            extras['service_tier'] = options.serviceTier;
        }
    }

    /**
     * Writes reasoning-related extras onto the request body according to the
     * model's `reasoningEffortFormat`.
     *
     * - "chat": emits a top-level `reasoning_effort` string when an effort is
     *   present (OpenAI Chat Completions, OpenRouter). Any other keys on the
     *   reasoning object are forwarded under `reasoning` as a fallback so
     *   providers that accept the OpenRouter-style `reasoning` object still work.
     * - "responses": emits a nested `reasoning` object with `{ effort }`
     *   (OpenAI Responses API style).
     * - unset: forwards the entire `reasoning` object verbatim (OpenRouter
     *   passthrough — the historical behavior).
     */
    private applyReasoningExtras(
        extras: Record<string, unknown>,
        reasoning: Record<string, unknown>,
        format: 'chat' | 'responses' | undefined
    ): void {
        const effort = typeof reasoning['effort'] === 'string' ? reasoning['effort'] : undefined;
        const enabled = reasoning['enabled'];
        const maxTokens = reasoning['max_tokens'];

        if (format === 'responses') {
            const nested: Record<string, unknown> = {};
            if (effort !== undefined) {
                nested['effort'] = effort;
            }
            if (enabled !== undefined) {
                nested['enabled'] = enabled;
            }
            if (typeof maxTokens === 'number' && Number.isFinite(maxTokens)) {
                nested['max_tokens'] = maxTokens;
            }
            // Merge any extra keys the provider sent through `reasoning`.
            for (const [key, value] of Object.entries(reasoning)) {
                if (key !== 'effort' && key !== 'enabled' && key !== 'max_tokens') {
                    nested[key] = value;
                }
            }
            if (Object.keys(nested).length > 0) {
                extras['reasoning'] = nested;
            }
            return;
        }

        if (format === 'chat') {
            if (effort !== undefined) {
                extras['reasoning_effort'] = effort;
            }
            // Forward non-effort keys as a top-level `reasoning` object for
            // providers that accept the OpenRouter-style shape (e.g. enabled, max_tokens).
            const passthrough: Record<string, unknown> = {};
            for (const [key, value] of Object.entries(reasoning)) {
                if (key !== 'effort') {
                    passthrough[key] = value;
                }
            }
            if (Object.keys(passthrough).length > 0) {
                extras['reasoning'] = passthrough;
            }
            return;
        }

        // Unset: passthrough the whole object (OpenRouter style).
        extras['reasoning'] = reasoning;
    }

    /**
     * Converts ChatMessage array to OpenAI ChatCompletionMessageParam format.
     * Handles different message roles with their specific type requirements.
     */
    private convertMessagesToOpenAIFormat(messages: ChatMessage[]): OpenAI.Chat.ChatCompletionMessageParam[] {
        let fallbackIdCounter = 0;
        return messages.map(msg => {
            const textContent = this.contentToText(msg.content);
            switch (msg.role) {
                case 'system':
                    return {
                        role: 'system' as const,
                        content: textContent
                    };
                case 'user':
                    return {
                        role: 'user' as const,
                        content: this.toUserMessageContent(msg.content)
                    };
                case 'assistant':
                    // Assistant messages can have tool_calls
                    if (msg.tool_calls && msg.tool_calls.length > 0) {
                        // Filter out tool calls without valid IDs to prevent API errors
                        const validToolCalls = msg.tool_calls.filter(tc => 
                            tc.id && typeof tc.id === 'string' && tc.id.trim().length > 0
                        );
                        if (validToolCalls.length > 0) {
                            return {
                                role: 'assistant' as const,
                                content: textContent || null,
                                tool_calls: validToolCalls.map(tc => ({
                                    id: tc.id,
                                    type: 'function' as const,
                                    function: {
                                        name: tc.function.name,
                                        arguments: tc.function.arguments
                                    }
                                }))
                            };
                        }
                    }
                    return {
                        role: 'assistant' as const,
                        content: textContent
                    };
                case 'tool':
                    // Tool messages must have content (not null) and a valid tool_call_id
                    // Some APIs (e.g., Claude via OpenAI-compatible proxies) require non-empty tool_call_id
                    let toolCallId = msg.tool_call_id;
                    if (!toolCallId || typeof toolCallId !== 'string' || toolCallId.trim().length === 0) {
                        // Log warning as this may cause issues with some API providers
                        logger.warn('Tool message missing valid tool_call_id, using fallback', 'OpenAI');
                        // Generate a fallback ID using counter + timestamp + random component for uniqueness
                        // Random component ensures uniqueness even if called multiple times in same millisecond
                        toolCallId = `call_fallback_${Date.now()}_${fallbackIdCounter++}_${Math.random().toString(36).slice(2, 9)}`;
                    }
                    return {
                        role: 'tool' as const,
                        content: textContent,
                        tool_call_id: toolCallId
                    };
                default:
                    // Fallback to user role
                    return {
                        role: 'user' as const,
                        content: this.toUserMessageContent(msg.content)
                    };
            }
        });
    }

    private contentToText(content: ChatMessageContent): string {
        if (typeof content === 'string') {
            return content;
        }
        if (!Array.isArray(content)) {
            return '';
        }
        return content
            .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
            .map(part => part.text)
            .join('');
    }

    private toUserMessageContent(content: ChatMessageContent): string | OpenAI.Chat.ChatCompletionContentPart[] {
        if (!Array.isArray(content)) {
            return content ?? '';
        }

        const parts: OpenAI.Chat.ChatCompletionContentPart[] = [];
        for (const part of content) {
            if (part.type === 'text') {
                parts.push({
                    type: 'text',
                    text: part.text
                });
            } else if (part.type === 'image_url') {
                parts.push({
                    type: 'image_url',
                    image_url: part.image_url
                });
            }
        }
        return parts.length > 0 ? parts : '';
    }

    private convertMessagesToResponsesInput(messages: ChatMessage[]): OpenAI.Responses.ResponseInputItem[] {
        const inputItems: OpenAI.Responses.ResponseInputItem[] = [];
        let fallbackIdCounter = 0;

        const ensureCallId = (value: unknown, context: string): string => {
            if (typeof value === 'string' && value.trim().length > 0) {
                return value;
            }
            logger.warn(`${context} missing valid call_id, using fallback`, 'OpenAI');
            return `call_fallback_${Date.now()}_${fallbackIdCounter++}_${Math.random().toString(36).slice(2, 9)}`;
        };

        for (const msg of messages) {
            if (msg.role === 'tool') {
                const callId = ensureCallId(msg.tool_call_id, 'Tool message');
                inputItems.push({
                    type: 'function_call_output',
                    call_id: callId,
                    output: this.contentToText(msg.content)
                });
                continue;
            }

            if (msg.role === 'assistant' && msg.tool_calls && msg.tool_calls.length > 0) {
                const assistantContent = this.contentToText(msg.content);
                if (assistantContent.length > 0) {
                    inputItems.push({
                        role: 'assistant',
                        content: assistantContent,
                        type: 'message'
                    });
                }

                for (const toolCall of msg.tool_calls) {
                    const callId = ensureCallId(toolCall.id, `Tool call (${toolCall.function?.name ?? 'unknown'})`);
                    const name = toolCall.function?.name;
                    if (!name) {
                        continue;
                    }
                    inputItems.push({
                        type: 'function_call',
                        call_id: callId,
                        name,
                        arguments: toolCall.function.arguments ?? ''
                    });
                }
                continue;
            }

            const content = this.contentToText(msg.content);
            if (!content) {
                continue;
            }

            const role = msg.role === 'system'
                ? 'system'
                : msg.role === 'assistant'
                    ? 'assistant'
                    : 'user';

            inputItems.push({
                role,
                content,
                type: 'message'
            });
        }

        return inputItems;
    }

    private convertToolsToResponsesTools(tools: ToolDefinition[] | undefined): OpenAI.Responses.Tool[] | undefined {
        if (!tools || tools.length === 0) {
            return undefined;
        }

        const converted: OpenAI.Responses.Tool[] = [];
        for (const tool of tools) {
            if (tool.type !== 'function') {
                continue;
            }
            const name = tool.function?.name?.trim();
            if (!name) {
                continue;
            }
            const parameters = tool.function.parameters ?? { type: 'object', properties: {} };
            converted.push({
                type: 'function',
                name,
                description: tool.function.description ?? undefined,
                parameters,
                strict: true
            });
        }

        return converted.length > 0 ? converted : undefined;
    }

    private convertToolChoiceToResponsesToolChoice(toolChoice: ToolChoice | undefined): OpenAI.Responses.ToolChoiceOptions | OpenAI.Responses.ToolChoiceFunction | undefined {
        if (!toolChoice) {
            return undefined;
        }
        if (toolChoice === 'none' || toolChoice === 'auto' || toolChoice === 'required') {
            return toolChoice;
        }
        const name = toolChoice.function?.name;
        if (!name) {
            return undefined;
        }
        return {
            type: 'function',
            name
        };
    }
}
