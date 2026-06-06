import {
    LanguageModelDataPart as _LanguageModelDataPart,
    LanguageModelResponsePart as _LanguageModelResponsePart
} from "vscode";

declare module 'vscode' {
    export type ExLanguageModelResponsePart = _LanguageModelResponsePart | _LanguageModelDataPart | LanguageModelThinkingPart;

    export interface LanguageModelChatInformation {
        /**
         * Numeric cost/scale multiplier used by recent VS Code/Copilot model picker surfaces.
         */
        readonly multiplierNumeric?: number;

        /**
         * Whether or not the model will show up in the model picker immediately.
         */
        readonly isUserSelectable?: boolean;
    }

    export interface LanguageModelChatCapabilities {
        /**
         * Preferred edit tools for Copilot's editing flows.
         */
        readonly editTools?: string[];
        readonly editToolsHint?: string[];
    }
}
