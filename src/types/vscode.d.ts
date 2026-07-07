import {
    LanguageModelDataPart as _LanguageModelDataPart,
    LanguageModelResponsePart as _LanguageModelResponsePart
} from "vscode";

declare module 'vscode' {
    export type ExLanguageModelResponsePart = _LanguageModelResponsePart | _LanguageModelDataPart | LanguageModelThinkingPart;

    /**
     * A JSON Schema describing configurable options for a language model.
     * Mirrors `LanguageModelConfigurationSchema` from `vscode.proposed.chatProvider.d.ts`.
     * Each property in `properties` defines a configurable option (e.g. reasoning effort)
     * using standard JSON Schema fields plus display hints.
     */
    export interface LanguageModelConfigurationSchema {
        readonly type?: 'object';
        readonly properties?: {
            readonly [key: string]: Record<string, any> & {
                /** Human-readable labels for enum values, shown instead of the raw values. */
                readonly enumItemLabels?: string[];
                /** When set to `'navigation'`, the property is shown as a primary action in the model picker. */
                readonly group?: string;
            };
        };
    }

    export interface LanguageModelChatInformation {
        /**
         * Numeric cost/scale multiplier used by recent VS Code/Copilot model picker surfaces.
         */
        readonly multiplierNumeric?: number;

        /**
         * Whether or not the model will show up in the model picker immediately.
         */
        readonly isUserSelectable?: boolean;

        /**
         * Whether this is a "bring your own key" (BYOK) model served with user-supplied
         * credentials rather than through the built-in Copilot (CAPI) service.
         */
        readonly isBYOK?: boolean;

        /**
         * When present, gates `requestLanguageModelAccess` behind an authorization flow
         * where the user must approve another extension accessing these models.
         */
        readonly requiresAuthorization?: true | { label: string };

        /**
         * Optional pricing label, e.g. "Free", "2x", "$0.01/request". For display only.
         */
        readonly pricing?: string;

        /** Optional input cost in AI credits per 1M tokens. */
        readonly inputCost?: number;
        /** Optional output cost in AI credits per 1M tokens. */
        readonly outputCost?: number;
        /** Optional cached-input (read) cost in credits per 1M tokens. */
        readonly cacheCost?: number;
        /** Optional cache-write cost in credits per 1M tokens. */
        readonly cacheWriteCost?: number;

        /** Optional long-context input cost (when differs from default). */
        readonly longContextInputCost?: number;
        /** Optional long-context output cost (when differs from default). */
        readonly longContextOutputCost?: number;
        /** Optional long-context cache cost (when differs from default). */
        readonly longContextCacheCost?: number;
        /** Optional long-context cache write cost (when differs from default). */
        readonly longContextCacheWriteCost?: number;

        /** Optional relative pricing category: "low" | "medium" | "high" | "very_high". */
        readonly priceCategory?: string;
        /** Optional model tier: "lightweight" | "versatile" | "powerful". */
        readonly category?: string;

        /**
         * JSON Schema describing per-model configuration options (e.g. reasoning effort).
         * When present, VS Code renders a configuration action in the model picker.
         */
        readonly configurationSchema?: LanguageModelConfigurationSchema;

        /**
         * Optional warning text shown alongside the model in the picker.
         */
        readonly warningText?: string | Record<string, string>;
    }

    export interface LanguageModelChatCapabilities {
        /**
         * Preferred edit tools for Copilot's editing flows.
         */
        readonly editTools?: string[];
        readonly editToolsHint?: string[];
    }
}
