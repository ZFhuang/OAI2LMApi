import { createHash } from 'node:crypto';
import * as vscode from 'vscode';
import { OpenAIClient, ChatMessage, APIModelInfo, ToolDefinition, ToolChoice, ToolCallChunk, CompletedToolCall, OpenAIUsage, OpenAIResponseMetadata, ChatMessageContentPart, OpenAIRequestOptions } from './openaiClient';
import { API_KEY_SECRET_KEY, CACHED_MODELS_KEY } from './constants';
import { getModelMetadata, isLLMModel, supportsToolCalling, ModelMetadata } from './modelMetadata';
import { generateXmlToolPrompt, formatToolCallAsXml, formatToolResultAsText, XmlToolCallStreamParser, XmlToolParseOptions } from './xmlToolPrompt';
import { getModelOverride, getEditTools } from './configUtils';
import { logger } from './logger';
import { modelsDevRegistry } from './modelsDevClient';

interface ModelInformation extends vscode.LanguageModelChatInformation {
    modelId: string;
    supportedParameters?: string[];
    defaultParameters?: Record<string, unknown>;
    supportsReasoning?: boolean;
    /** Reasoning effort levels supported by the model (e.g. ["low","medium","high"]). */
    supportedReasoningEfforts?: string[];
    /** Default reasoning effort for the model. */
    defaultReasoningEffort?: string;
    /** Body shape for reasoning effort: "chat" (top-level) or "responses" (nested). */
    reasoningEffortFormat?: 'chat' | 'responses';
}

export class OpenAILanguageModelProvider implements vscode.LanguageModelChatProvider<ModelInformation>, vscode.Disposable {
    private client: OpenAIClient | undefined;
    private disposables: vscode.Disposable[] = [];
    private modelList: ModelInformation[] = [];
    private _onDidChangeLanguageModelChatInformation = new vscode.EventEmitter<void>();
    private tokensPerChar = 0.25;
    private readonly tokenCountCache = new Map<string, number>();
    private readonly tokenCountCacheMaxEntries = 256;
    
    readonly onDidChangeLanguageModelChatInformation = this._onDidChangeLanguageModelChatInformation.event;

    constructor(private context: vscode.ExtensionContext) {}

    async initialize() {
        const config = vscode.workspace.getConfiguration('oai2lmapi');
        const apiEndpoint = config.get<string>('apiEndpoint', 'https://api.openai.com/v1');
        
        // Retrieve API key from SecretStorage
        const apiKey = await this.context.secrets.get(API_KEY_SECRET_KEY);

        logger.info(`Initializing with endpoint: ${apiEndpoint}`, 'OpenAI');

        if (!apiKey) {
            logger.warn('API key not configured', 'OpenAI');
            vscode.window.showWarningMessage('OAI2LMApi: API key not configured. Use command "OAI2LMApi: Set API Key" to configure.');
            return;
        }

        this.client = new OpenAIClient({
            apiEndpoint,
            apiKey
        });

        // Register the provider
        logger.info('Registering language model provider', 'OpenAI');
        const disposable = vscode.lm.registerLanguageModelChatProvider('oai2lmapi', this);
        this.disposables.push(disposable);

        // Try to load cached models first
        const cachedModels = this.context.globalState.get<APIModelInfo[]>(CACHED_MODELS_KEY);
        if (cachedModels && cachedModels.length > 0) {
            logger.info(`Loading ${cachedModels.length} models from cache`, 'OpenAI');
            this.updateModelList(cachedModels);
        }

        // Auto-load models if enabled
        const autoLoadModels = config.get<boolean>('autoLoadModels', true);
        if (autoLoadModels) {
            logger.info('Auto-loading models from API', 'OpenAI');
            await this.loadModels();
        } else {
            logger.warn('autoLoadModels is disabled; no models have been loaded automatically', 'OpenAI');
            if (!cachedModels || cachedModels.length === 0) {
                vscode.window.showWarningMessage(
                    'OAI2LMApi: autoLoadModels is disabled. No models have been loaded automatically; enable autoLoadModels in settings or manually refresh models to use this provider.'
                );
            }
        }
    }

    private updateModelList(apiModels: APIModelInfo[]) {
        const config = vscode.workspace.getConfiguration('oai2lmapi');
        const showModelsWithoutToolCalling = config.get<boolean>('showModelsWithoutToolCalling', false);

        // Clear existing models
        this.modelList = [];

        // Filter and add models
        let addedCount = 0;
        let filteredCount = 0;
        for (const apiModel of apiModels) {
            // Filter out non-LLM models (embedding, rerank, image, audio, etc.)
            if (!isLLMModel(apiModel.id)) {
                filteredCount++;
                logger.debug(`Filtered out non-LLM model: ${apiModel.id}`, undefined, 'OpenAI');
                continue;
            }

            // Filter out models without tool calling support unless setting is enabled
            if (!showModelsWithoutToolCalling && !this.modelSupportsToolCalling(apiModel)) {
                filteredCount++;
                logger.debug(`Filtered out model without tool calling: ${apiModel.id}`, undefined, 'OpenAI');
                continue;
            }

            this.addModel(apiModel);
            addedCount++;
        }

        logger.info(`Added ${addedCount} models, filtered ${filteredCount} models`, 'OpenAI');

        // Notify listeners that models changed
        this._onDidChangeLanguageModelChatInformation.fire();
    }

    async loadModels() {
        if (!this.client) {
            logger.warn('OpenAI client not initialized', 'OpenAI');
            return;
        }

        try {
            const apiModels = await this.client.listModels();
            logger.info(`Loaded ${apiModels.length} models from API`, 'OpenAI');

            this.updateModelList(apiModels);

            // Cache the models
            await this.context.globalState.update(CACHED_MODELS_KEY, apiModels);

            // Notify models.dev registry of loaded model IDs for new-model detection
            modelsDevRegistry.onModelsLoaded(apiModels.map(m => m.id));
        } catch (error) {
            logger.error('Failed to load models from API', error, 'OpenAI');
            vscode.window.showErrorMessage(`OAI2LMApi: Failed to load models from API. Please check your endpoint and API key.`);
            this._onDidChangeLanguageModelChatInformation.fire();
        }
    }

    /**
     * Checks if a model supports tool calling.
     * First checks API response, then falls back to pre-fetched metadata.
     */
    private modelSupportsToolCalling(apiModel: APIModelInfo): boolean {
        // Check if API provides capability information
        if (apiModel.capabilities?.tool_calling !== undefined) {
            return apiModel.capabilities.tool_calling;
        }
        if (apiModel.capabilities?.tools !== undefined) {
            return apiModel.capabilities.tools;
        }
        if (apiModel.capabilities?.tool_use !== undefined) {
            return apiModel.capabilities.tool_use;
        }
        if (apiModel.capabilities?.function_calling !== undefined) {
            return apiModel.capabilities.function_calling;
        }
        if (apiModel.supports_tools !== undefined) {
            return apiModel.supports_tools;
        }
        if (apiModel.supports_tool_use !== undefined) {
            return apiModel.supports_tool_use;
        }
        if (apiModel.supports_function_calling !== undefined) {
            return apiModel.supports_function_calling;
        }
        if (apiModel.supportsToolCall !== undefined) {
            return apiModel.supportsToolCall;
        }
        if (apiModel.supported_parameters?.includes('tools')) {
            return true;
        }
        // Fall back to pre-fetched metadata
        return supportsToolCalling(apiModel.id);
    }

    /**
     * Gets model metadata, preferring API response over pre-fetched data.
     */
    private getModelInfo(apiModel: APIModelInfo): { metadata: ModelMetadata; fromApi: boolean } {
        const registryMetadata = getModelMetadata(apiModel.id);
        
        // Start with registry metadata as base
        const metadata: ModelMetadata = { ...registryMetadata };
        let fromApi = false;

        // Override with API-provided values if available
        const apiMaxInputTokens = this.getValidNumber(
            apiModel.context_length,
            apiModel.top_provider?.context_length,
            apiModel.maxInputTokens,
            apiModel.maxAllowedSize
        );
        if (apiMaxInputTokens !== undefined) {
            metadata.maxInputTokens = apiMaxInputTokens;
            fromApi = true;
        }
        const apiMaxOutputTokens = this.getValidNumber(
            apiModel.max_completion_tokens,
            apiModel.top_provider?.max_completion_tokens,
            apiModel.maxOutputTokens
        );
        if (apiMaxOutputTokens !== undefined) {
            metadata.maxOutputTokens = apiMaxOutputTokens;
            fromApi = true;
        }
        const apiSupportsToolCalling = this.getApiToolCallingSupport(apiModel);
        if (apiSupportsToolCalling !== undefined) {
            metadata.supportsToolCalling = apiSupportsToolCalling;
            fromApi = true;
        }
        const apiSupportsImageInput = this.getApiImageInputSupport(apiModel);
        if (apiSupportsImageInput !== undefined) {
            metadata.supportsImageInput = apiSupportsImageInput;
            fromApi = true;
        }

        return { metadata, fromApi };
    }

    private getValidNumber(...values: Array<number | undefined>): number | undefined {
        for (const value of values) {
            if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
                return value;
            }
        }
        return undefined;
    }

    private getApiToolCallingSupport(apiModel: APIModelInfo): boolean | undefined {
        if (apiModel.capabilities?.tool_calling !== undefined) {
            return apiModel.capabilities.tool_calling;
        }
        if (apiModel.capabilities?.tools !== undefined) {
            return apiModel.capabilities.tools;
        }
        if (apiModel.capabilities?.tool_use !== undefined) {
            return apiModel.capabilities.tool_use;
        }
        if (apiModel.capabilities?.function_calling !== undefined) {
            return apiModel.capabilities.function_calling;
        }
        if (apiModel.supports_tools !== undefined) {
            return apiModel.supports_tools;
        }
        if (apiModel.supports_tool_use !== undefined) {
            return apiModel.supports_tool_use;
        }
        if (apiModel.supports_function_calling !== undefined) {
            return apiModel.supports_function_calling;
        }
        if (apiModel.supportsToolCall !== undefined) {
            return apiModel.supportsToolCall;
        }
        if (apiModel.supported_parameters?.includes('tools')) {
            return true;
        }
        return undefined;
    }

    private getApiImageInputSupport(apiModel: APIModelInfo): boolean | undefined {
        if (apiModel.capabilities?.vision !== undefined) {
            return apiModel.capabilities.vision;
        }
        if (apiModel.supports_vision !== undefined) {
            return apiModel.supports_vision;
        }
        if (apiModel.supportsImages !== undefined) {
            return apiModel.disabledMultimodal ? false : apiModel.supportsImages;
        }
        const inputModalities = apiModel.architecture?.input_modalities;
        if (inputModalities) {
            return inputModalities.includes('image');
        }
        return undefined;
    }

    /**
     * Extracts model family from model ID.
     * Examples: 'gpt-4o-mini' -> 'gpt-4o', 'claude-3.5-sonnet' -> 'claude-3.5'
     */
    private extractModelFamily(modelId: string): string {
        // Remove provider prefix if present
        const nameWithoutPrefix = modelId.replace(/^[^/]+\//, '');
        
        // Common patterns for model families
        const patterns = [
            // OpenAI patterns
            /^(gpt-4\.1|gpt-4o|gpt-4-turbo|gpt-4|gpt-3\.5-turbo|o1|o3|o4)(?=$|\b|[-_])/i,
            // Anthropic patterns
            /^(claude-sonnet-4|claude-3\.7|claude-3\.5|claude-3|claude-2\.1|claude-2|claude-instant)/i,
            // Google patterns
            /^(gemini-2\.5|gemini-2\.0|gemini-1\.5|gemini)/i,
            // Meta patterns
            /^(llama-4|llama-3\.3|llama-3\.2|llama-3\.1|llama-3|llama-2)/i,
            // Mistral patterns
            /^(mistral-large|mistral-medium|mistral-small|mixtral-8x22b|mixtral-8x7b|mistral|codestral|pixtral)/i,
            // Qwen patterns
            /^(qwq|qvq|qwen-3|qwen-2\.5|qwen-2|qwen-1\.5|qwen)/i,
            // DeepSeek patterns
            /^(deepseek-r1|deepseek-v3|deepseek-v2\.5|deepseek-v2|deepseek)/i,
        ];

        for (const pattern of patterns) {
            const match = nameWithoutPrefix.match(pattern);
            if (match) {
                return match[1].toLowerCase();
            }
        }

        // Fallback: use the model ID as-is for the family
        // This ensures unknown models are at least grouped consistently
        return nameWithoutPrefix.toLowerCase();
    }

    private getCreditMultiplier(apiModel: APIModelInfo): number | undefined {
        if (typeof apiModel.credit_multiplier === 'number' && Number.isFinite(apiModel.credit_multiplier)) {
            return apiModel.credit_multiplier;
        }
        if (typeof apiModel.credits !== 'string') {
            return undefined;
        }
        const match = apiModel.credits.match(/x\s*([0-9]+(?:\.[0-9]+)?)/i);
        if (!match) {
            return undefined;
        }
        const parsed = Number(match[1]);
        return Number.isFinite(parsed) ? parsed : undefined;
    }

    private getModelVersion(apiModel: APIModelInfo): string {
        if (typeof apiModel.created_at === 'string' && apiModel.created_at.trim()) {
            return apiModel.created_at;
        }
        if (typeof apiModel.created_at === 'number' && Number.isFinite(apiModel.created_at)) {
            return String(apiModel.created_at);
        }
        if (typeof apiModel.created === 'number' && Number.isFinite(apiModel.created) && apiModel.created > 0) {
            return String(apiModel.created);
        }
        return '1.0';
    }

    private getModelDetail(apiModel: APIModelInfo): string | undefined {
        const parts = [
            apiModel.vendor,
            apiModel.credits,
            apiModel.architecture?.modality
        ].filter((part): part is string => typeof part === 'string' && part.trim().length > 0);
        if (parts.length > 0) {
            return parts.join(' | ');
        }
        // Fallback: use the first line of description (truncated) when no short
        // labels are available, so the model picker still shows a meaningful subtitle.
        if (typeof apiModel.description === 'string') {
            const firstLine = apiModel.description.split(/\r?\n/)[0]?.trim();
            if (firstLine) {
                return firstLine.length > 80 ? `${firstLine.slice(0, 77)}...` : firstLine;
            }
        }
        return undefined;
    }

    private getModelTooltip(
        apiModel: APIModelInfo,
        maxInputTokens: number,
        maxOutputTokens: number,
        supportsToolCalling: boolean,
        supportsImageInput: boolean
    ): string {
        const lines: string[] = [];
        const displayName = apiModel.display_name || apiModel.name || apiModel.id;
        lines.push(displayName);
        if (apiModel.description) {
            lines.push(apiModel.description);
        }
        lines.push(`Context: ${maxInputTokens.toLocaleString()} input / ${maxOutputTokens.toLocaleString()} output tokens`);

        const capabilities: string[] = [];
        if (supportsToolCalling) {
            capabilities.push('tools');
        }
        if (supportsImageInput) {
            capabilities.push('images');
        }
        if (apiModel.supports_reasoning || apiModel.supportsReasoning || apiModel.capabilities?.reasoning) {
            capabilities.push('reasoning');
        }
        if (capabilities.length > 0) {
            lines.push(`Capabilities: ${capabilities.join(', ')}`);
        }
        if (apiModel.supported_parameters && apiModel.supported_parameters.length > 0) {
            lines.push(`Parameters: ${apiModel.supported_parameters.join(', ')}`);
        }
        return lines.join('\n');
    }

    private addModel(apiModel: APIModelInfo) {
        const { metadata, fromApi } = this.getModelInfo(apiModel);
        const family = this.extractModelFamily(apiModel.id);

        let maxInputTokens = metadata.maxInputTokens;
        let maxOutputTokens = metadata.maxOutputTokens;
        let supportsToolCalling = metadata.supportsToolCalling;
        let supportsImageInput = metadata.supportsImageInput;
        let multiplierNumeric = this.getCreditMultiplier(apiModel);
        const supportsReasoning = apiModel.supports_reasoning === true
            || apiModel.supportsReasoning === true
            || apiModel.capabilities?.reasoning === true
            || typeof apiModel.default_parameters?.['reasoning'] === 'object';

        // Pricing / cost fields — prefer API-provided values, fall back to overrides.
        let pricing = apiModel.pricing;
        let inputCost = apiModel.inputCost;
        let outputCost = apiModel.outputCost;
        let cacheCost = apiModel.cacheCost;
        let cacheWriteCost = apiModel.cacheWriteCost;
        let longContextInputCost = apiModel.longContextInputCost;
        let longContextOutputCost = apiModel.longContextOutputCost;
        let longContextCacheCost = apiModel.longContextCacheCost;
        let longContextCacheWriteCost = apiModel.longContextCacheWriteCost;
        let priceCategory = apiModel.priceCategory;
        let category = apiModel.category;

        // Reasoning effort configuration — prefer API, fall back to overrides.
        let supportedReasoningEfforts = apiModel.supportedReasoningEfforts;
        let defaultReasoningEffort = apiModel.defaultReasoningEffort;
        let reasoningEffortFormat = apiModel.reasoningEffortFormat;

        const override = getModelOverride(apiModel.id, 'openai');
        if (override) {
            if (typeof override.maxInputTokens === 'number' && Number.isFinite(override.maxInputTokens)) {
                maxInputTokens = override.maxInputTokens;
            }
            if (typeof override.maxOutputTokens === 'number' && Number.isFinite(override.maxOutputTokens)) {
                maxOutputTokens = override.maxOutputTokens;
            }
            if (typeof override.supportsToolCalling === 'boolean') {
                supportsToolCalling = override.supportsToolCalling;
            }
            if (typeof override.supportsImageInput === 'boolean') {
                supportsImageInput = override.supportsImageInput;
            }
            if (typeof override.multiplierNumeric === 'number' && Number.isFinite(override.multiplierNumeric)) {
                multiplierNumeric = override.multiplierNumeric;
            }
            if (typeof override.pricing === 'string') {
                pricing = override.pricing;
            }
            if (typeof override.inputCost === 'number') {
                inputCost = override.inputCost;
            }
            if (typeof override.outputCost === 'number') {
                outputCost = override.outputCost;
            }
            if (typeof override.cacheCost === 'number') {
                cacheCost = override.cacheCost;
            }
            if (typeof override.cacheWriteCost === 'number') {
                cacheWriteCost = override.cacheWriteCost;
            }
            if (typeof override.longContextInputCost === 'number') {
                longContextInputCost = override.longContextInputCost;
            }
            if (typeof override.longContextOutputCost === 'number') {
                longContextOutputCost = override.longContextOutputCost;
            }
            if (typeof override.longContextCacheCost === 'number') {
                longContextCacheCost = override.longContextCacheCost;
            }
            if (typeof override.longContextCacheWriteCost === 'number') {
                longContextCacheWriteCost = override.longContextCacheWriteCost;
            }
            if (typeof override.priceCategory === 'string') {
                priceCategory = override.priceCategory;
            }
            if (typeof override.category === 'string') {
                category = override.category;
            }
            if (Array.isArray(override.supportedReasoningEfforts) && override.supportedReasoningEfforts.length > 0) {
                supportedReasoningEfforts = override.supportedReasoningEfforts;
            }
            if (typeof override.defaultReasoningEffort === 'string') {
                defaultReasoningEffort = override.defaultReasoningEffort;
            }
            if (override.reasoningEffortFormat === 'chat' || override.reasoningEffortFormat === 'responses') {
                reasoningEffortFormat = override.reasoningEffortFormat;
            }
        }

        // Build a display pricing label when one wasn't provided but a multiplier is known.
        if (!pricing && typeof multiplierNumeric === 'number' && Number.isFinite(multiplierNumeric)) {
            pricing = `${multiplierNumeric}x`;
        }

        // Build a configuration schema so VS Code can surface reasoning effort in the picker.
        const configurationSchema = this.buildConfigurationSchema(
            supportsReasoning,
            supportedReasoningEfforts,
            defaultReasoningEffort
        );

        const modelInfo: ModelInformation = {
            modelId: apiModel.id,
            id: `oai2lmapi-${apiModel.id}`,
            family: family,
            name: apiModel.display_name || apiModel.name || apiModel.id,
            version: this.getModelVersion(apiModel),
            detail: this.getModelDetail(apiModel),
            tooltip: this.getModelTooltip(apiModel, maxInputTokens, maxOutputTokens, supportsToolCalling, supportsImageInput),
            maxInputTokens,
            maxOutputTokens,
            multiplierNumeric,
            isUserSelectable: true,
            isBYOK: true,
            ...(pricing !== undefined ? { pricing } : {}),
            ...(inputCost !== undefined ? { inputCost } : {}),
            ...(outputCost !== undefined ? { outputCost } : {}),
            ...(cacheCost !== undefined ? { cacheCost } : {}),
            ...(cacheWriteCost !== undefined ? { cacheWriteCost } : {}),
            ...(longContextInputCost !== undefined ? { longContextInputCost } : {}),
            ...(longContextOutputCost !== undefined ? { longContextOutputCost } : {}),
            ...(longContextCacheCost !== undefined ? { longContextCacheCost } : {}),
            ...(longContextCacheWriteCost !== undefined ? { longContextCacheWriteCost } : {}),
            ...(priceCategory !== undefined ? { priceCategory } : {}),
            ...(category !== undefined ? { category } : {}),
            ...(configurationSchema ? { configurationSchema } : {}),
            supportedParameters: apiModel.supported_parameters,
            defaultParameters: apiModel.default_parameters,
            supportsReasoning,
            ...(supportedReasoningEfforts ? { supportedReasoningEfforts } : {}),
            ...(defaultReasoningEffort ? { defaultReasoningEffort } : {}),
            ...(reasoningEffortFormat ? { reasoningEffortFormat } : {}),
            capabilities: {
                toolCalling: supportsToolCalling,
                imageInput: supportsImageInput,
                editTools: getEditTools(supportsToolCalling),
                editToolsHint: getEditTools(supportsToolCalling)
            }
        };

        this.modelList.push(modelInfo);
        const source = fromApi ? 'API' : 'registry';
        const hasOverride = override ? ' (with overrides)' : '';
        logger.debug(`Added model: ${modelInfo.id} (family: ${family}, source: ${source})${hasOverride}`, undefined, 'OpenAI');
    }

    /**
     * Builds a `LanguageModelConfigurationSchema` exposing reasoning effort as a
     * primary picker action (group: 'navigation') for models that support reasoning.
     * Mirrors the schema built by VS Code's built-in BYOK providers
     * (see `claudeCodeModels.ts` / `copilotCli.ts`).
     */
    private buildConfigurationSchema(
        supportsReasoning: boolean,
        supportedReasoningEfforts: string[] | undefined,
        defaultReasoningEffort: string | undefined
    ): vscode.LanguageModelConfigurationSchema | undefined {
        if (!supportsReasoning) {
            return undefined;
        }

        // Default effort levels when the provider didn't enumerate them.
        const effortLevels = supportedReasoningEfforts && supportedReasoningEfforts.length > 0
            ? supportedReasoningEfforts
            : ['low', 'medium', 'high'];
        const defaultEffort = defaultReasoningEffort
            ?? (effortLevels.includes('high') ? 'high' : effortLevels[effortLevels.length - 1]);

        return {
            properties: {
                reasoningEffort: {
                    type: 'string',
                    title: 'Thinking Effort',
                    enum: effortLevels,
                    enumItemLabels: effortLevels.map(level => level.charAt(0).toUpperCase() + level.slice(1)),
                    default: defaultEffort,
                    group: 'navigation'
                }
            }
        };
    }

    async provideLanguageModelChatInformation(
        options: vscode.PrepareLanguageModelChatModelOptions,
        token: vscode.CancellationToken
    ): Promise<ModelInformation[]> {
        logger.debug(`Providing ${this.modelList.length} models to VSCode`, undefined, 'OpenAI');
        // Surface only stable LanguageModelChatInformation fields plus our own
        // `modelId`. Proposed `chatProvider` fields (pricing, configurationSchema,
        // isBYOK, multiplierNumeric, supportedParameters, supportsReasoning, etc.)
        // stay on the internal objects for our own use but must not be returned
        // here, otherwise VS Code gates the provider behind the `chatProvider`
        // proposed API and refuses to register models.
        return this.modelList.map(model => ({
            id: model.id,
            modelId: model.modelId,
            name: model.name,
            family: model.family,
            version: model.version,
            maxInputTokens: model.maxInputTokens,
            maxOutputTokens: model.maxOutputTokens,
            ...(model.tooltip ? { tooltip: model.tooltip } : {}),
            ...(model.detail ? { detail: model.detail } : {}),
            capabilities: {
                toolCalling: typeof model.capabilities?.toolCalling === 'boolean' ? model.capabilities.toolCalling : Boolean(model.capabilities?.toolCalling),
                imageInput: typeof model.capabilities?.imageInput === 'boolean' ? model.capabilities.imageInput : Boolean(model.capabilities?.imageInput)
            }
        }));
    }

    async provideLanguageModelChatResponse(
        model: ModelInformation,
        messages: readonly vscode.LanguageModelChatRequestMessage[],
        options: vscode.ProvideLanguageModelChatResponseOptions,
        progress: vscode.Progress<vscode.ExLanguageModelResponsePart>,
        token: vscode.CancellationToken
    ): Promise<void> {
        if (!this.client) {
            throw new Error('OpenAI client not initialized');
        }

        // VS Code passes back the stable-only object we returned from
        // provideLanguageModelChatInformation, which omits proposed chatProvider
        // fields (supportsReasoning, defaultParameters, reasoningEffortFormat,
        // etc.). Resolve the full internal model entry by modelId so request
        // option building still has access to those fields.
        const internalModel = this.modelList.find(m => m.modelId === model.modelId) ?? model;

        // Check if prompt-based tool calling is enabled for this model
        const modelOverride = getModelOverride(internalModel.modelId, 'openai');
        const usePromptBasedToolCalling = modelOverride?.usePromptBasedToolCalling === true;

        // Determine whether to use the OpenAI Responses API for this model
        const config = vscode.workspace.getConfiguration('oai2lmapi');
        const openaiResponsesApiMode = config.get<string>('openaiResponsesApiMode', 'off');
        const normalizedModelId = model.modelId.replace(/^[^/]+\//, '').toLowerCase();
        const isGptModel = normalizedModelId.startsWith('gpt-');
        const useResponsesApi = typeof modelOverride?.useResponsesApi === 'boolean'
            ? modelOverride.useResponsesApi
            : (openaiResponsesApiMode === 'all' || (openaiResponsesApiMode === 'gpt-only' && isGptModel));

        // Chain-of-thought suppression: per-model override takes precedence over global.
        const globalSuppressChainOfThought = config.get<boolean>('suppressChainOfThought', false);
        const suppressChainOfThought = modelOverride?.suppressChainOfThought ?? globalSuppressChainOfThought;

        // XML tool parameter whitespace handling: per-model override takes precedence over global.
        const globalTrimXmlToolParameterWhitespace = config.get<boolean>('trimXmlToolParameterWhitespace', false);
        const trimXmlToolParameterWhitespace = modelOverride?.trimXmlToolParameterWhitespace ?? globalTrimXmlToolParameterWhitespace;
        const xmlParseOptions: XmlToolParseOptions = {
            trimParameterWhitespace: trimXmlToolParameterWhitespace
        };

        // Convert VSCode messages to OpenAI format
        let chatMessages: ChatMessage[] = this.convertMessages(messages, usePromptBasedToolCalling);

        // Get available tool names for XML parsing
        const availableToolNames = options.tools?.map(t => t.name).filter((n): n is string => !!n) ?? [];

        // Handle prompt-based tool calling
        let tools: ToolDefinition[] | undefined;
        let toolChoice: ToolChoice | undefined;

        if (usePromptBasedToolCalling && options.tools && options.tools.length > 0) {
            // Generate XML tool prompt and prepend to system message
            const xmlToolPrompt = generateXmlToolPrompt(options.tools);
            
            // Find or create system message
            const systemMsgIndex = chatMessages.findIndex(m => m.role === 'system');
            if (systemMsgIndex >= 0) {
                chatMessages[systemMsgIndex] = {
                    ...chatMessages[systemMsgIndex],
                    content: (chatMessages[systemMsgIndex].content || '') + '\n\n' + xmlToolPrompt
                };
            } else {
                // Prepend a new system message
                chatMessages = [{ role: 'system', content: xmlToolPrompt }, ...chatMessages];
            }
            
            // Don't pass native tools when using prompt-based tool calling
            tools = undefined;
            toolChoice = undefined;
            logger.debug(`Using prompt-based tool calling for model ${model.modelId}`, undefined, 'OpenAI');
        } else {
            // Use native function calling
            tools = this.convertTools(options.tools, model.modelId);
            toolChoice = this.convertToolMode(options.toolMode);
        }

        // Derive a reasonable maxTokens for providers that require an explicit budget.
        // VS Code's API surface may vary; probe common fields defensively.
        const optionsAny = options as any;
        const budgetFromOptions: unknown = optionsAny?.tokenBudget ?? optionsAny?.maxTokens ?? optionsAny?.maxOutputTokens;
        const budgetNumber = typeof budgetFromOptions === 'number' && Number.isFinite(budgetFromOptions) ? budgetFromOptions : undefined;
        const modelBudget = typeof model.maxOutputTokens === 'number' && Number.isFinite(model.maxOutputTokens) ? model.maxOutputTokens : 2048;
        // Cap to avoid proxies rejecting very large max_tokens.
        const maxTokens = Math.max(1, Math.min(budgetNumber ?? modelBudget, 8192));
        const requestOptions = this.buildRequestOptions(internalModel, options, modelOverride);

        // Create abort controller from cancellation token
        const abortController = new AbortController();
        if (token) {
            token.onCancellationRequested(() => {
                abortController.abort();
            });
        }

        // Track reported tool call IDs to prevent duplicates
        const reportedToolCallIds = new Set<string>();
        let responseUsage: OpenAIUsage | undefined;
        let responseMetadata: OpenAIResponseMetadata | undefined;

        // For prompt-based tool calling, use streaming parser to detect tool calls incrementally
        const streamParser = usePromptBasedToolCalling && availableToolNames.length > 0 
            ? new XmlToolCallStreamParser(availableToolNames, xmlParseOptions) 
            : null;

        // Stream the response
        const responseText = await this.client.streamChatCompletion(
            chatMessages,
            model.modelId,
            {
                onChunk: (chunk) => {
                    if (streamParser) {
                        // Add chunk to parser and emit any newly detected tool calls immediately
                        const newToolCalls = streamParser.addChunk(chunk);
                        for (const toolCall of newToolCalls) {
                            if (reportedToolCallIds.has(toolCall.id)) {
                                continue;
                            }
                            reportedToolCallIds.add(toolCall.id);
                            
                            progress.report(new vscode.LanguageModelToolCallPart(
                                toolCall.id,
                                toolCall.name,
                                toolCall.arguments
                            ));
                            logger.debug(`Streaming XML tool call detected: ${toolCall.name}`, undefined, 'OpenAI');
                        }
                    } else {
                        progress.report(new vscode.LanguageModelTextPart(chunk));
                    }
                },
                onThinkingChunk: (chunk, metadata) => {
                    this.tryReportThinkingPart(chunk, this.getThinkingIdFromMetadata(metadata), metadata, progress);
                },
                onToolCallStarted: (toolCall) => {
                    logger.debug(`Streaming native tool call started: ${toolCall.name}`, {
                        id: toolCall.id
                    }, 'OpenAI');
                },
                onToolCall: (toolCall) => {
                    this.tryReportNativeToolCall(toolCall, reportedToolCallIds, progress);
                },
                suppressChainOfThought,
                useResponsesApi,
                onToolCallsComplete: (toolCalls: CompletedToolCall[]) => {
                    // Report all tool calls at once after streaming is complete
                    for (const toolCall of toolCalls) {
                        // Skip if already reported (prevent duplicates)
                        if (reportedToolCallIds.has(toolCall.id)) {
                            continue;
                        }
                        reportedToolCallIds.add(toolCall.id);

                        try {
                            const parsedArgs = JSON.parse(toolCall.arguments);
                            progress.report(new vscode.LanguageModelToolCallPart(
                                toolCall.id,
                                toolCall.name,
                                parsedArgs
                            ));
                        } catch {
                            // If arguments are not valid JSON, report with empty object
                            logger.debug(`Failed to parse tool call arguments for ${toolCall.name}: ${toolCall.arguments}`, undefined, 'OpenAI');
                            progress.report(new vscode.LanguageModelToolCallPart(
                                toolCall.id,
                                toolCall.name,
                                {}
                            ));
                        }
                    }
                },
                onUsage: (usage) => {
                    responseUsage = usage;
                },
                onResponseMetadata: (metadata) => {
                    responseMetadata = metadata;
                },
                signal: abortController.signal,
                tools,
                toolChoice,
                maxTokens,
                requestOptions
            }
        );

        // After streaming, finalize the parser to catch any remaining tool calls
        if (streamParser) {
            const remainingToolCalls = streamParser.finalize();
            for (const toolCall of remainingToolCalls) {
                if (reportedToolCallIds.has(toolCall.id)) {
                    continue;
                }
                reportedToolCallIds.add(toolCall.id);
                
                progress.report(new vscode.LanguageModelToolCallPart(
                    toolCall.id,
                    toolCall.name,
                    toolCall.arguments
                ));
                logger.debug(`Finalized XML tool call: ${toolCall.name}`, undefined, 'OpenAI');
            }
            
            // Report any non-tool-call text content to the user
            const nonToolCallText = streamParser.getNonToolCallText();
            if (nonToolCallText) {
                progress.report(new vscode.LanguageModelTextPart(nonToolCallText));
            }
        }

        if (responseUsage) {
            this.rememberApiUsageTokenCounts(chatMessages, responseText, responseUsage);
            progress.report(this.createUsageDataPart(responseUsage));
        }
        if (responseMetadata) {
            progress.report(this.createResponseMetadataDataPart(responseMetadata));
        }
    }

    private tryReportNativeToolCall(
        toolCall: ToolCallChunk | CompletedToolCall,
        reportedToolCallIds: Set<string>,
        progress: vscode.Progress<vscode.LanguageModelTextPart | vscode.LanguageModelToolCallPart | vscode.LanguageModelDataPart | vscode.LanguageModelThinkingPart>
    ): boolean {
        if (!toolCall.id || reportedToolCallIds.has(toolCall.id)) {
            return false;
        }

        const parsedArgs = this.tryParseToolCallArguments(toolCall.arguments);
        if (!parsedArgs) {
            return false;
        }

        reportedToolCallIds.add(toolCall.id);
        progress.report(new vscode.LanguageModelToolCallPart(
            toolCall.id,
            toolCall.name,
            parsedArgs
        ));
        logger.debug(`Streaming native tool call emitted early: ${toolCall.name}`, undefined, 'OpenAI');
        return true;
    }

    private tryParseToolCallArguments(argumentsJson: string): Record<string, unknown> | undefined {
        try {
            const parsed = JSON.parse(argumentsJson || '{}');
            if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
                return parsed as Record<string, unknown>;
            }
        } catch {
            return undefined;
        }
        return undefined;
    }

    private createUsageDataPart(usage: OpenAIUsage): vscode.LanguageModelDataPart {
        return new vscode.LanguageModelDataPart(
            new TextEncoder().encode(JSON.stringify(usage)),
            'usage'
        );
    }

    private createResponseMetadataDataPart(metadata: OpenAIResponseMetadata): vscode.LanguageModelDataPart {
        return new vscode.LanguageModelDataPart(
            new TextEncoder().encode(JSON.stringify(metadata)),
            'openai.response_metadata'
        );
    }

    private getThinkingIdFromMetadata(metadata: Record<string, unknown> | undefined): string | undefined {
        if (!metadata) {
            return undefined;
        }
        for (const key of ['cot_id', 'reasoning_opaque', 'reasoning_signature', 'signature', 'id']) {
            const value = metadata[key];
            if (typeof value === 'string' && value.length > 0) {
                return value;
            }
        }
        return undefined;
    }

    private buildRequestOptions(
        model: ModelInformation,
        options: vscode.ProvideLanguageModelChatResponseOptions,
        override: ReturnType<typeof getModelOverride>
    ): OpenAIRequestOptions {
        const modelOptions = this.getRecord((options as { modelOptions?: unknown }).modelOptions);
        const modelConfiguration = this.getRecord((options as { modelConfiguration?: unknown }).modelConfiguration);
        const defaultParameters = this.getRecord(model.defaultParameters);

        const temperature = this.getFiniteNumber(modelOptions?.['temperature'])
            ?? this.getFiniteNumber(defaultParameters?.['temperature'])
            ?? (typeof override?.temperature === 'number' && Number.isFinite(override.temperature) ? override.temperature : undefined);
        const topP = this.getFiniteNumber(modelOptions?.['top_p'])
            ?? this.getFiniteNumber(modelOptions?.['topP'])
            ?? this.getFiniteNumber(defaultParameters?.['top_p']);
        const stop = this.getStopValue(modelOptions?.['stop']);
        const reasoning = this.getReasoningValue(modelOptions, modelConfiguration, defaultParameters, override, model);
        const includeReasoning = typeof modelOptions?.['include_reasoning'] === 'boolean'
            ? modelOptions['include_reasoning'] as boolean
            : typeof modelOptions?.['includeReasoning'] === 'boolean'
                ? modelOptions['includeReasoning'] as boolean
                : undefined;
        const responseFormat = this.getRecord(modelOptions?.['response_format']) ?? this.getRecord(modelOptions?.['responseFormat']);
        const serviceTier = typeof modelOptions?.['service_tier'] === 'string'
            ? modelOptions['service_tier']
            : typeof modelOptions?.['serviceTier'] === 'string'
                ? modelOptions['serviceTier']
                : undefined;

        return {
            ...(temperature !== undefined ? { temperature } : {}),
            ...(topP !== undefined ? { topP } : {}),
            ...(stop !== undefined ? { stop } : {}),
            ...(reasoning !== undefined ? { reasoning } : {}),
            ...(includeReasoning !== undefined ? { includeReasoning } : {}),
            ...(responseFormat !== undefined ? { responseFormat } : {}),
            ...(serviceTier !== undefined ? { serviceTier } : {}),
            ...(model.reasoningEffortFormat ? { reasoningEffortFormat: model.reasoningEffortFormat } : {})
        };
    }

    private getRecord(value: unknown): Record<string, unknown> | undefined {
        return typeof value === 'object' && value !== null && !Array.isArray(value)
            ? value as Record<string, unknown>
            : undefined;
    }

    private getFiniteNumber(value: unknown): number | undefined {
        return typeof value === 'number' && Number.isFinite(value)
            ? value
            : undefined;
    }

    private getStopValue(value: unknown): string | string[] | undefined {
        if (typeof value === 'string') {
            return value;
        }
        if (Array.isArray(value) && value.every(item => typeof item === 'string')) {
            return value;
        }
        return undefined;
    }

    private getReasoningValue(
        modelOptions: Record<string, unknown> | undefined,
        modelConfiguration: Record<string, unknown> | undefined,
        defaultParameters: Record<string, unknown> | undefined,
        override: ReturnType<typeof getModelOverride>,
        model: ModelInformation
    ): Record<string, unknown> | undefined {
        const explicitReasoning = this.getRecord(modelOptions?.['reasoning']);
        if (explicitReasoning) {
            return explicitReasoning;
        }

        const defaultReasoning = this.getRecord(defaultParameters?.['reasoning']);
        const effort = typeof modelOptions?.['reasoning_effort'] === 'string'
            ? modelOptions['reasoning_effort']
            : typeof modelOptions?.['reasoningEffort'] === 'string'
                ? modelOptions['reasoningEffort']
                : typeof modelConfiguration?.['reasoningEffort'] === 'string'
                    ? modelConfiguration['reasoningEffort']
                    : undefined;
        if (effort) {
            return { ...(defaultReasoning ?? {}), effort };
        }

        if (typeof override?.thinkingLevel === 'string' && override.thinkingLevel !== 'auto') {
            if (override.thinkingLevel === 'none') {
                return { ...(defaultReasoning ?? {}), enabled: false, effort: 'none' };
            }
            return { ...(defaultReasoning ?? {}), effort: override.thinkingLevel };
        }
        if (typeof override?.thinkingLevel === 'number' && Number.isFinite(override.thinkingLevel)) {
            return { ...(defaultReasoning ?? {}), enabled: true, max_tokens: override.thinkingLevel };
        }

        if (defaultReasoning && model.supportsReasoning) {
            return defaultReasoning;
        }
        return undefined;
    }

    private getContentText(content: ChatMessage['content']): string {
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

    private getContentTextLength(content: ChatMessage['content']): number {
        return this.getContentText(content).length;
    }

    private getTokenCountCacheKey(textContent: string): string | undefined {
        if (!textContent) {
            return undefined;
        }
        return createHash('sha256').update(textContent).digest('hex');
    }

    private rememberTokenCount(textContent: string, tokenCount: number): void {
        const key = this.getTokenCountCacheKey(textContent);
        if (!key || !Number.isFinite(tokenCount) || tokenCount < 0) {
            return;
        }

        if (this.tokenCountCache.has(key)) {
            this.tokenCountCache.delete(key);
        }
        this.tokenCountCache.set(key, Math.ceil(tokenCount));

        while (this.tokenCountCache.size > this.tokenCountCacheMaxEntries) {
            const oldestKey = this.tokenCountCache.keys().next().value;
            if (oldestKey === undefined) {
                break;
            }
            this.tokenCountCache.delete(oldestKey);
        }
    }

    private getCachedTokenCount(textContent: string): number | undefined {
        const key = this.getTokenCountCacheKey(textContent);
        if (!key) {
            return undefined;
        }

        const tokenCount = this.tokenCountCache.get(key);
        if (tokenCount === undefined) {
            return undefined;
        }

        this.tokenCountCache.delete(key);
        this.tokenCountCache.set(key, tokenCount);
        return tokenCount;
    }

    private getVisibleCompletionTokenCount(usage: OpenAIUsage): number | undefined {
        const reasoningTokens = usage.completion_tokens_details?.reasoning_tokens ?? 0;
        const visibleCompletionTokens = usage.completion_tokens - reasoningTokens;
        return visibleCompletionTokens > 0 ? visibleCompletionTokens : undefined;
    }

    private rememberApiUsageTokenCounts(chatMessages: ChatMessage[], responseText: string, usage: OpenAIUsage): void {
        const promptTexts = chatMessages
            .map(msg => this.getContentText(msg.content))
            .filter(text => text.length > 0);
        const promptText = promptTexts.join('');

        if (promptText && usage.prompt_tokens > 0) {
            this.rememberTokenCount(promptText, usage.prompt_tokens);
        }
        if (promptTexts.length === 1 && usage.prompt_tokens > 0) {
            this.rememberTokenCount(promptTexts[0], usage.prompt_tokens);
        }

        const inputChars = promptText.length;
        if (inputChars > 0 && usage.prompt_tokens > 0) {
            this.tokensPerChar = usage.prompt_tokens / inputChars;
        }

        const visibleCompletionTokens = this.getVisibleCompletionTokenCount(usage);
        if (responseText && visibleCompletionTokens !== undefined) {
            this.rememberTokenCount(responseText, visibleCompletionTokens);
        }
    }

    /**
     * Converts VSCode messages to OpenAI ChatMessage format.
     * Handles tool calls and tool results in message history.
     * 
     * @param messages - VSCode messages to convert
     * @param usePromptBasedToolCalling - If true, convert tool calls/results to plain text format
     */
    private convertMessages(messages: readonly vscode.LanguageModelChatRequestMessage[], usePromptBasedToolCalling = false): ChatMessage[] {
        const result: ChatMessage[] = [];
        const processedToolCallIds = new Set<string>();
        let toolCallIndex = 0;

        for (const msg of messages) {
            const role = this.mapRole(msg.role);
            
            if (Array.isArray(msg.content)) {
                // Check if this message contains tool calls or tool results
                const toolCalls: Array<{ id: string; type: 'function'; function: { name: string; arguments: string } }> = [];
                const toolResults: Array<{ tool_call_id: string; content: string; toolName?: string }> = [];
                let textContent = '';
                let thinkingContent = '';
                let thinkingId: string | undefined;
                let thinkingMetadata: Record<string, unknown> | undefined;
                const contentParts: ChatMessageContentPart[] = [];

                for (const part of msg.content) {
                    if (this.isThinkingPart(part)) {
                        const thinkingText = this.extractThinkingContent(part);
                        if (thinkingText) {
                            thinkingContent += thinkingText;
                        }
                        thinkingId ??= this.getThinkingId(part);
                        thinkingMetadata = {
                            ...(thinkingMetadata ?? {}),
                            ...(part.metadata ?? {})
                        };
                    } else if (this.isToolCallPart(part)) {
                        // Ensure we have a valid tool call ID
                        const toolCallId = this.ensureToolCallId(part.callId, part.name, toolCallIndex++);
                        
                        // Skip duplicate tool calls in message history
                        if (processedToolCallIds.has(toolCallId)) {
                            continue;
                        }
                        processedToolCallIds.add(toolCallId);
                        
                        if (usePromptBasedToolCalling) {
                            // For prompt-based tool calling, convert tool call to XML text
                            const args = typeof part.input === 'object' && part.input !== null 
                                ? part.input as Record<string, unknown>
                                : {};
                            textContent += formatToolCallAsXml(part.name, args) + '\n';
                        } else {
                            // Native function calling format
                            toolCalls.push({
                                id: toolCallId,
                                type: 'function',
                                function: {
                                    name: part.name,
                                    arguments: JSON.stringify(part.input)
                                }
                            });
                        }
                    } else if (this.isToolResultPart(part)) {
                        // This is a tool result
                        const resultContent = this.extractToolResultContent(part);
                        // Ensure we have a valid tool call ID, consistent with tool calls
                        const toolCallId = this.ensureToolCallId(part.callId, 'result', toolCallIndex++);
                        
                        // Try to get tool name from the part
                        const toolName = this.getToolNameFromResult(part);
                        
                        toolResults.push({
                            tool_call_id: toolCallId,
                            content: resultContent,
                            toolName
                        });
                    } else {
                        // Regular text content
                        const extractedParts = this.extractContentPartsFromPart(part);
                        if (extractedParts.length > 0) {
                            contentParts.push(...extractedParts);
                            textContent += extractedParts
                                .filter((contentPart): contentPart is { type: 'text'; text: string } => contentPart.type === 'text')
                                .map(contentPart => contentPart.text)
                                .join('');
                        } else {
                            textContent += this.extractTextFromPart(part);
                        }
                    }
                }

                const reasoningFields = role === 'assistant'
                    ? this.buildReasoningChatFields(thinkingContent, thinkingId, thinkingMetadata)
                    : {};
                const hasReasoningFields = Object.keys(reasoningFields).length > 0;

                // Handle tool calls and results based on mode
                if (usePromptBasedToolCalling) {
                    // For prompt-based tool calling, everything is text.
                    // Tool results are pushed as user messages first, followed by assistant
                    // content with tool calls. This ordering may differ from the original
                    // message content array order, but is consistent with the prompt-based
                    // tool calling convention where results precede the next assistant turn.
                    if (toolResults.length > 0) {
                        // Convert tool results to user message with formatted text
                        const formattedResults = toolResults.map(tr => 
                            formatToolResultAsText(tr.toolName || 'Tool', tr.content)
                        ).join('\n\n');
                        result.push({
                            role: 'user',
                            content: formattedResults
                        });
                    }
                    // Tool calls were already added to textContent
                    if (textContent.trim()) {
                        result.push({
                            role,
                            content: textContent.trim(),
                            ...reasoningFields
                        });
                    } else if (role === 'assistant' && hasReasoningFields) {
                        result.push({
                            role: 'assistant',
                            content: null,
                            ...reasoningFields
                        });
                    }
                } else {
                    // Native function calling mode
                    // If we have tool calls, this is an assistant message with tool calls
                    if (toolCalls.length > 0) {
                        result.push({
                            role: 'assistant',
                            content: textContent || null,
                            tool_calls: toolCalls,
                            ...reasoningFields
                        });
                    }
                    // If we have tool results, add them as separate tool messages
                    else if (toolResults.length > 0) {
                        for (const toolResult of toolResults) {
                            result.push({
                                role: 'tool',
                                content: toolResult.content,
                                tool_call_id: toolResult.tool_call_id
                            });
                        }
                    }
                    // Regular message with text content
                    else if (textContent || contentParts.length > 0) {
                        result.push({
                            role,
                            content: role === 'user' && contentParts.some(part => part.type === 'image_url')
                                ? this.mergeAdjacentTextParts(contentParts)
                                : textContent,
                            ...reasoningFields
                        });
                    } else if (role === 'assistant' && hasReasoningFields) {
                        result.push({
                            role: 'assistant',
                            content: null,
                            ...reasoningFields
                        });
                    }
                }
            } else if (typeof msg.content === 'string') {
                result.push({
                    role,
                    content: msg.content
                });
            } else if (msg.content && typeof msg.content === 'object') {
                const contentParts = this.extractContentPartsFromPart(msg.content);
                result.push({
                    role,
                    content: role === 'user' && contentParts.some(part => part.type === 'image_url')
                        ? this.mergeAdjacentTextParts(contentParts)
                        : this.extractTextFromPart(msg.content)
                });
            }
        }

        return result;
    }

    private mergeAdjacentTextParts(parts: ChatMessageContentPart[]): ChatMessageContentPart[] {
        const merged: ChatMessageContentPart[] = [];
        for (const part of parts) {
            const previous = merged[merged.length - 1];
            if (part.type === 'text' && previous?.type === 'text') {
                previous.text += part.text;
            } else {
                merged.push(part);
            }
        }
        return merged;
    }

    private isThinkingPart(part: unknown): part is { value: string | string[]; id?: string; metadata?: Record<string, unknown> } {
        return typeof part === 'object'
            && part !== null
            && (part as { constructor?: { name?: string } }).constructor?.name === 'LanguageModelThinkingPart';
    }

    private tryReportThinkingPart(
        value: string,
        id: string | undefined,
        metadata: Record<string, unknown> | undefined,
        progress: vscode.Progress<vscode.LanguageModelTextPart | vscode.LanguageModelToolCallPart | vscode.LanguageModelDataPart | vscode.LanguageModelThinkingPart>
    ): void {
        void value;
        void id;
        void metadata;
        void progress;
    }

    private extractThinkingContent(part: { value: string | string[] }): string {
        return Array.isArray(part.value) ? part.value.join('') : part.value;
    }

    private getThinkingId(part: { id?: string; metadata?: Record<string, unknown> }): string | undefined {
        if (typeof part.id === 'string' && part.id.length > 0) {
            return part.id;
        }
        const metadata = part.metadata;
        if (!metadata) {
            return undefined;
        }
        for (const key of ['cot_id', 'reasoning_opaque', 'reasoning_signature', 'signature', 'id']) {
            const value = metadata[key];
            if (typeof value === 'string' && value.length > 0) {
                return value;
            }
        }
        return undefined;
    }

    private buildReasoningChatFields(
        thinkingContent: string,
        thinkingId: string | undefined,
        metadata: Record<string, unknown> | undefined
    ): Partial<ChatMessage> {
        if (!thinkingContent) {
            return {};
        }
        const reasoningOpaque = typeof metadata?.['reasoning_opaque'] === 'string' ? metadata['reasoning_opaque'] : undefined;
        const signature = typeof metadata?.['signature'] === 'string'
            ? metadata['signature']
            : typeof metadata?.['reasoning_signature'] === 'string'
                ? metadata['reasoning_signature']
                : undefined;
        return {
            ...(thinkingId ? { cot_id: thinkingId } : {}),
            cot_summary: thinkingContent,
            reasoning_content: thinkingContent,
            reasoning: thinkingContent,
            ...(reasoningOpaque ? { reasoning_opaque: reasoningOpaque } : {}),
            ...(signature ? { signature } : {})
        };
    }

    private extractContentPartsFromPart(part: unknown): ChatMessageContentPart[] {
        if (!part || typeof part !== 'object') {
            return [];
        }

        if ('mimeType' in part && 'data' in part) {
            const dataPart = part as { mimeType?: unknown; data?: unknown };
            if (typeof dataPart.mimeType === 'string' && dataPart.data instanceof Uint8Array) {
                if (dataPart.mimeType.startsWith('image/')) {
                    return [{
                        type: 'image_url',
                        image_url: {
                            url: `data:${dataPart.mimeType};base64,${Buffer.from(dataPart.data).toString('base64')}`
                        }
                    }];
                }
                if (dataPart.mimeType.startsWith('text/')) {
                    return [{
                        type: 'text',
                        text: new TextDecoder().decode(dataPart.data)
                    }];
                }
            }
        }

        const text = this.extractTextFromPart(part);
        return text ? [{ type: 'text', text }] : [];
    }

    /**
     * Checks if a content part is a LanguageModelToolCallPart
     */
    private isToolCallPart(part: unknown): part is { callId: string; name: string; input: unknown } {
        return part !== null &&
               typeof part === 'object' &&
               'callId' in part &&
               'name' in part &&
               'input' in part;
    }

    /**
     * Checks if a content part is a LanguageModelToolResultPart
     */
    private isToolResultPart(part: unknown): part is { callId: string; content: unknown[] } {
        return part !== null &&
               typeof part === 'object' &&
               'callId' in part &&
               'content' in part &&
               !('name' in part); // Distinguish from ToolCallPart
    }

    /**
     * Generates a unique tool call ID if one is missing or invalid.
     */
    private ensureToolCallId(callId: unknown, name: string, index: number): string {
        if (typeof callId === 'string' && callId.trim().length > 0) {
            return callId;
        }
        // Generate a unique ID using timestamp + index + random component for uniqueness
        // Random component ensures uniqueness even if called multiple times in same millisecond
        return `call_fallback_${Date.now()}_${index}_${Math.random().toString(36).slice(2, 9)}`;
    }

    /**
     * Extracts content from a tool result part
     */
    private extractToolResultContent(part: { callId: string; content: unknown[] }): string {
        if (Array.isArray(part.content)) {
            return part.content.map(c => this.extractTextFromPart(c)).join('');
        }
        return '';
    }

    /**
     * Gets the tool name from a tool result part if available.
     * 
     * Note: LanguageModelToolResultPart does not currently define a `toolName` property
     * in the VS Code API. This method is implemented as a forward-compatibility hook
     * in case future versions add such a property.
     */
    private getToolNameFromResult(part: unknown): string | undefined {
        if (!part || typeof part !== 'object') {
            return undefined;
        }

        // Forward-compatibility: check for an optional `toolName` property in a
        // type-safe way without using `any`. If present and a string, return it.
        const candidate = (part as { toolName?: unknown }).toolName;
        if (typeof candidate === 'string') {
            return candidate;
        }
        return undefined;
    }

    /**
     * Converts VSCode tools to OpenAI ToolDefinition format
     * @param modelId - The model ID, used for special handling of certain models
     */
    private convertTools(tools: readonly vscode.LanguageModelChatTool[] | undefined, modelId?: string): ToolDefinition[] | undefined {
        if (!tools || tools.length === 0) {
            return undefined;
        }

        let dropped = 0;
        let sanitized = 0;
        const converted: ToolDefinition[] = [];

        // WORKAROUND: Some API gateways (e.g., new-api converting to Gemini/Claude format)
        // fail when a tool has no parameters (empty properties object).
        // For models starting with 'gemini-claude' or 'claude', we inject a dummy parameter
        // to work around this issue. This is a temporary fix until we find a better solution.
        const needsDummyParameter = modelId && 
            (modelId.toLowerCase().startsWith('gemini-claude') || modelId.toLowerCase().startsWith('claude'));

        for (const tool of tools) {
            const name = (tool.name ?? '').trim();
            if (!name) {
                dropped++;
                continue;
            }

            // Some OpenAI-compatible gateways reject missing/empty `parameters`.
            // Ensure we always send at least a minimal JSON schema.
            let parameters: Record<string, unknown> | undefined = tool.inputSchema as Record<string, unknown> | undefined;
            const isPlainObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);

            if (!isPlainObject(parameters) || Object.keys(parameters).length === 0) {
                // Tool has no parameters
                if (needsDummyParameter) {
                    // Inject a dummy parameter for gateway compatibility
                    parameters = {
                        type: 'object',
                        properties: {
                            _placeholder: {
                                type: 'string',
                                description: 'This tool has no parameters. Do not pass any value for this parameter.'
                            }
                        },
                        required: []
                    };
                } else {
                    parameters = { type: 'object', properties: {} };
                }
                sanitized++;
            } else {
                // Ensure schema has a top-level type/properties when it's intended to be an object.
                if (!('type' in parameters)) {
                    parameters = { ...parameters, type: 'object' };
                    sanitized++;
                }
                if ((parameters as any).type === 'object' && !('properties' in parameters)) {
                    parameters = { ...parameters, properties: {} };
                    sanitized++;
                }
            }

            const description = (tool.description ?? '').trim();
            converted.push({
                type: 'function' as const,
                function: {
                    name,
                    description: description || undefined,
                    parameters
                }
            });
        }

        // Warn on dropped tools (empty name). Sanitizing schemas is expected for compatibility.
        if (dropped > 0) {
            logger.warn(`Dropped ${dropped} invalid tools with empty name`, 'OpenAI');
            logger.debug('Dropped tools details', {
                original: tools.length,
                converted: converted.length,
                dropped
            }, 'OpenAI');
        }

        return converted.length > 0 ? converted : undefined;
    }

    /**
     * Converts VSCode toolMode to OpenAI tool_choice format
     */
    private convertToolMode(toolMode: vscode.LanguageModelChatToolMode | undefined): ToolChoice | undefined {
        if (!toolMode) {
            return undefined;
        }

        switch (toolMode) {
            case vscode.LanguageModelChatToolMode.Auto:
                return 'auto';
            case vscode.LanguageModelChatToolMode.Required:
                return 'required';
            default:
                return 'auto';
        }
    }

    async provideTokenCount(
        model: ModelInformation,
        text: string | vscode.LanguageModelChatRequestMessage,
        token: vscode.CancellationToken
    ): Promise<number> {
        let textContent: string;
        
        if (typeof text === 'string') {
            textContent = text;
        } else {
            if (typeof text.content === 'string') {
                textContent = text.content;
            } else if (Array.isArray(text.content)) {
                textContent = text.content.map(part => this.extractTextFromPart(part)).join('');
            } else if (text.content && typeof text.content === 'object') {
                textContent = this.extractTextFromPart(text.content);
            } else {
                textContent = '';
            }
        }
        
        const cachedTokenCount = this.getCachedTokenCount(textContent);
        if (cachedTokenCount !== undefined) {
            return cachedTokenCount;
        }

        return Math.ceil(textContent.length * this.tokensPerChar);
    }

    /**
     * Extracts text content from a LanguageModel content part.
     * Handles LanguageModelTextPart, LanguageModelToolResultPart, and other part types.
     */
    private extractTextFromPart(part: unknown): string {
        if (!part || typeof part !== 'object') {
            return '';
        }
        
        // LanguageModelTextPart uses 'value' property in VSCode's API
        if ('value' in part && typeof (part as { value: unknown }).value === 'string') {
            return (part as { value: string }).value;
        }
        
        // Some implementations may use 'text' property
        if ('text' in part && typeof (part as { text: unknown }).text === 'string') {
            return (part as { text: string }).text;
        }
        
        // LanguageModelToolResultPart has a 'content' property which is an array
        if ('content' in part && Array.isArray((part as { content: unknown }).content)) {
            const contentArray = (part as { content: unknown[] }).content;
            return contentArray.map(subPart => this.extractTextFromPart(subPart)).join('');
        }
        
        // LanguageModelToolCallPart - serialize tool call info for context
        if ('toolName' in part && 'parameters' in part) {
            const toolPart = part as { toolName: string; parameters: unknown };
            try {
                return `[Tool Call: ${toolPart.toolName}(${JSON.stringify(toolPart.parameters)})]`;
            } catch {
                return `[Tool Call: ${toolPart.toolName}]`;
            }
        }
        
        return '';
    }

    private mapRole(role: vscode.LanguageModelChatMessageRole): 'system' | 'user' | 'assistant' {
        switch (role) {
            case vscode.LanguageModelChatMessageRole.User:
                return 'user';
            case vscode.LanguageModelChatMessageRole.Assistant:
                return 'assistant';
            default:
                if (role === 0 || String(role).toLowerCase() === 'system') {
                    return 'system';
                }
                return 'user';
        }
    }

    dispose() {
        this._onDidChangeLanguageModelChatInformation.dispose();
        for (const disposable of this.disposables) {
            disposable.dispose();
        }
        this.disposables = [];
        this.modelList = [];
    }
}
