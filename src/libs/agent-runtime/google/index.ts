import type { VertexAI } from '@google-cloud/vertexai';
import {
  Content,
  FunctionCallPart,
  FunctionDeclaration,
  Tool as GoogleFunctionCallTool,
  GoogleGenerativeAI,
  GoogleSearchRetrievalTool,
  Part,
  SchemaType,
} from '@google/generative-ai';

import type { ChatModelCard } from '@/types/llm';
import { imageUrlToBase64 } from '@/utils/imageToBase64';
import { safeParseJSON } from '@/utils/safeParseJSON';

import { LobeRuntimeAI } from '../BaseAI';
import { AgentRuntimeErrorType, ILobeAgentRuntimeErrorType } from '../error';
import {
  ChatCompetitionOptions,
  ChatCompletionTool,
  ChatStreamPayload,
  OpenAIChatMessage,
  UserMessageContentPart,
} from '../types';
import { AgentRuntimeError } from '../utils/createError';
import { debugStream } from '../utils/debugStream';
import { StreamingResponse } from '../utils/response';
import {
  GoogleGenerativeAIStream,
  VertexAIStream,
  convertIterableToStream,
} from '../utils/streams';
import { parseDataUri } from '../utils/uriParser';

const modelsOffSafetySettings = new Set(['gemini-2.0-flash-exp']);

const modelsWithModalities = new Set([
  'gemini-2.0-flash-exp',
  'gemini-2.0-flash-exp-image-generation',
]);

const modelsDisableInstuction = new Set([
  'gemini-2.0-flash-exp',
  'gemini-2.0-flash-exp-image-generation',
]);

export interface GoogleModelCard {
  displayName: string;
  inputTokenLimit: number;
  name: string;
  outputTokenLimit: number;
}

enum HarmCategory {
  HARM_CATEGORY_DANGEROUS_CONTENT = 'HARM_CATEGORY_DANGEROUS_CONTENT',
  HARM_CATEGORY_HARASSMENT = 'HARM_CATEGORY_HARASSMENT',
  HARM_CATEGORY_HATE_SPEECH = 'HARM_CATEGORY_HATE_SPEECH',
  HARM_CATEGORY_SEXUALLY_EXPLICIT = 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
}

enum HarmBlockThreshold {
  BLOCK_NONE = 'BLOCK_NONE',
}

function getThreshold(model: string): HarmBlockThreshold {
  if (modelsOffSafetySettings.has(model)) {
    return 'OFF' as HarmBlockThreshold; // https://discuss.ai.google.dev/t/59352
  }
  return HarmBlockThreshold.BLOCK_NONE;
}

const DEFAULT_BASE_URL = 'https://generativelanguage.googleapis.com';

interface LobeGoogleAIParams {
  apiKey?: string;
  baseURL?: string;
  client?: GoogleGenerativeAI | VertexAI;
  id?: string;
  isVertexAi?: boolean;
}

export class LobeGoogleAI implements LobeRuntimeAI {
  private client: GoogleGenerativeAI;
  private isVertexAi: boolean;
  baseURL?: string;
  apiKey?: string;
  provider: string;

  constructor({ apiKey, baseURL, client, isVertexAi, id }: LobeGoogleAIParams = {}) {
    if (!apiKey) throw AgentRuntimeError.createError(AgentRuntimeErrorType.InvalidProviderAPIKey);

    this.client = new GoogleGenerativeAI(apiKey);
    this.apiKey = apiKey;
    this.client = client ? (client as GoogleGenerativeAI) : new GoogleGenerativeAI(apiKey);
    this.baseURL = client ? undefined : baseURL || DEFAULT_BASE_URL;
    this.isVertexAi = isVertexAi || false;

    this.provider = id || (isVertexAi ? 'vertexai' : 'google');
  }

  async chat(rawPayload: ChatStreamPayload, options?: ChatCompetitionOptions) {
    try {
      const payload = this.buildPayload(rawPayload);
      const model = payload.model;

      const contents = await this.buildGoogleMessages(payload.messages);

      const geminiStreamResult = await this.client
        .getGenerativeModel(
          {
            generationConfig: {
              maxOutputTokens: payload.max_tokens,
              // @ts-expect-error - Google SDK 0.24.0 doesn't have this property for now with
              response_modalities: modelsWithModalities.has(model) ? ['Text', 'Image'] : undefined,
              temperature: payload.temperature,
              topP: payload.top_p,
            },
            model,
            // avoid wide sensitive words
            // refs: https://github.com/lobehub/lobe-chat/pull/1418
            safetySettings: [
              {
                category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold: getThreshold(model),
              },
              {
                category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold: getThreshold(model),
              },
              {
                category: HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold: getThreshold(model),
              },
              {
                category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold: getThreshold(model),
              },
            ],
          },
          { apiVersion: 'v1beta', baseUrl: this.baseURL },
        )
        .generateContentStream({
          contents,
          systemInstruction: modelsDisableInstuction.has(model)
            ? undefined
            : (payload.system as string),
          tools: this.buildGoogleTools(payload.tools, payload),
        });

      const googleStream = convertIterableToStream(geminiStreamResult.stream);
      const [prod, useForDebug] = googleStream.tee();

      const key = this.isVertexAi
        ? 'DEBUG_VERTEX_AI_CHAT_COMPLETION'
        : 'DEBUG_GOOGLE_CHAT_COMPLETION';

      if (process.env[key] === '1') {
        debugStream(useForDebug).catch();
      }

      // Convert the response into a friendly text-stream
      const Stream = this.isVertexAi ? VertexAIStream : GoogleGenerativeAIStream;
      const stream = Stream(prod, options?.callback);

      // Respond with the stream
      return StreamingResponse(stream, { headers: options?.headers });
    } catch (e) {
      const err = e as Error;

      console.log(err);
      const { errorType, error } = this.parseErrorMessage(err.message);

      throw AgentRuntimeError.chat({ error, errorType, provider: this.provider });
    }
  }

  async models() {
    const { LOBE_DEFAULT_MODEL_LIST } = await import('@/config/aiModels');

    const url = `${this.baseURL}/v1beta/models?key=${this.apiKey}`;
    const response = await fetch(url, {
      method: 'GET',
    });
    const json = await response.json();

    const modelList: GoogleModelCard[] = json['models'];

    return modelList
      .map((model) => {
        const modelName = model.name.replace(/^models\//, '');

        const knownModel = LOBE_DEFAULT_MODEL_LIST.find(
          (m) => modelName.toLowerCase() === m.id.toLowerCase(),
        );

        return {
          contextWindowTokens: model.inputTokenLimit + model.outputTokenLimit,
          displayName: model.displayName,
          enabled: knownModel?.enabled || false,
          functionCall:
            (modelName.toLowerCase().includes('gemini') &&
              !modelName.toLowerCase().includes('thinking')) ||
            knownModel?.abilities?.functionCall ||
            false,
          id: modelName,
          reasoning:
            modelName.toLowerCase().includes('thinking') ||
            knownModel?.abilities?.reasoning ||
            false,
          vision:
            modelName.toLowerCase().includes('vision') ||
            (modelName.toLowerCase().includes('gemini') &&
              !modelName.toLowerCase().includes('gemini-1.0')) ||
            knownModel?.abilities?.vision ||
            false,
        };
      })
      .filter(Boolean) as ChatModelCard[];
  }

  private buildPayload(payload: ChatStreamPayload) {
    const system_message = payload.messages.find((m) => m.role === 'system');
    const user_messages = payload.messages.filter((m) => m.role !== 'system');

    return {
      ...payload,
      messages: user_messages,
      system: system_message?.content,
    };
  }
  private convertContentToGooglePart = async (
    content: UserMessageContentPart,
  ): Promise<Part | undefined> => {
    console.log(
      '<<<<<<<<<<<<<<<<<< EXECUTING MODIFIED google/index.ts convertContentToGooglePart >>>>>>>>>>>>>>>>>>',
    );
    switch (content.type) {
      default: {
        return undefined;
      }

      case 'text': {
        return { text: content.text };
      }

      case 'image_url': {
        const { mimeType, base64, type } = parseDataUri(content.image_url.url);

        if (type === 'base64') {
          if (!base64) {
            console.error(
              "LobeGoogleAI Error: Image Data URI doesn't contain base64 data",
              content.image_url.url,
            );
            // 可以選擇拋出錯誤或返回 undefined 跳過這個部分
            // throw new TypeError("Image URL (Data URI) doesn't contain base64 data");
            return undefined;
          }
          console.log('LobeGoogleAI: Using provided Base64 Data URI for image.');
          return {
            inlineData: {
              data: base64,
              mimeType: mimeType || 'image/png',
            },
          };
        }

        // 情況 2：輸入的是普通的 HTTP/HTTPS URL
        if (type === 'url') {
          // 檢查環境變數是否要求使用 Base64
          if (process.env.LLM_VISION_IMAGE_USE_BASE64 === '1') {
            console.log(
              'LobeGoogleAI: LLM_VISION_IMAGE_USE_BASE64=1 detected. Fetching and converting image URL to Base64...',
            );
            try {
              const { base64: fetchedBase64, mimeType: fetchedMimeType } = await imageUrlToBase64(
                content.image_url.url,
              );
              console.log(
                `LobeGoogleAI: Conversion complete. MimeType: ${fetchedMimeType}, Base64 length: ${fetchedBase64.length}`,
              );
              return {
                inlineData: {
                  data: fetchedBase64,
                  // 確保 MIME Type 是 Google 支援的，例如 image/png 或 image/jpeg
                  mimeType:
                    fetchedMimeType === 'image/webp'
                      ? 'image/jpeg'
                      : fetchedMimeType || 'image/jpeg',
                },
              };
            } catch (error) {
              console.error(
                'LobeGoogleAI Error: Failed to fetch or convert image URL to Base64.',
                error,
              );
              // 轉換失敗，可以選擇返回 undefined 或包含錯誤訊息的文本部分
              // return { text: `[Error processing image: ${content.image_url.url}]` };
              return undefined;
            }
          } else {
            // 如果環境變數沒有設定為 '1'
            console.warn(
              `LobeGoogleAI Warning: LLM_VISION_IMAGE_USE_BASE64 is not set to '1'. ` +
                `Directly sending image URLs to Google Gemini API is often problematic and may result in errors ` +
                `(like 'Unsupported MIME type'). It's recommended to set this env var to '1' ` +
                `when using Google Vision models with image URLs. Attempting Base64 conversion anyway.`,
            );
            // 即使未設定，鑑於直接傳 URL 的問題，仍然嘗試轉換為 Base64
            try {
              const { base64: fetchedBase64, mimeType: fetchedMimeType } = await imageUrlToBase64(
                content.image_url.url,
              );
              console.log(
                `LobeGoogleAI: Conversion complete (fallback). MimeType: ${fetchedMimeType}, Base64 length: ${fetchedBase64.length}`,
              );
              return {
                inlineData: {
                  data: fetchedBase64,
                  mimeType:
                    fetchedMimeType === 'image/webp'
                      ? 'image/jpeg'
                      : fetchedMimeType || 'image/jpeg',
                },
              };
            } catch (error) {
              console.error(
                'LobeGoogleAI Error: Failed to fetch or convert image URL to Base64 (fallback).',
                error,
              );
              return undefined;
            }

            // 如果你想在未設定 Base64 時嚴格禁止 URL，可以取消下面這行的註解
            // throw AgentRuntimeError.createError(AgentRuntimeErrorType.UnsupportedVisionModel, { provider: this.provider, model: 'Gemini (Vision URL requires LLM_VISION_IMAGE_USE_BASE64=1)' });
          }
        }

        // 如果 URI 類型無法識別
        console.error(
          'LobeGoogleAI Error: Unsupported image URL format or type:',
          content.image_url.url,
        );
        // throw new TypeError(`Unsupported image URL format or type: ${content.image_url.url}`);
        return undefined;
      }
    }
  };

  private convertOAIMessagesToGoogleMessage = async (
    message: OpenAIChatMessage,
  ): Promise<Content> => {
    const content = message.content as string | UserMessageContentPart[];
    if (!!message.tool_calls) {
      return {
        parts: message.tool_calls.map<FunctionCallPart>((tool) => ({
          functionCall: {
            args: safeParseJSON(tool.function.arguments)!,
            name: tool.function.name,
          },
        })),
        role: 'function',
      };
    }

    const getParts = async () => {
      if (typeof content === 'string') return [{ text: content }];

      const parts = await Promise.all(
        content.map(async (c) => await this.convertContentToGooglePart(c)),
      );
      return parts.filter(Boolean) as Part[];
    };

    return {
      parts: await getParts(),
      role: message.role === 'assistant' ? 'model' : 'user',
    };
  };

  // convert messages from the OpenAI format to Google GenAI SDK
  private buildGoogleMessages = async (messages: OpenAIChatMessage[]): Promise<Content[]> => {
    const pools = messages
      .filter((message) => message.role !== 'function')
      .map(async (msg) => await this.convertOAIMessagesToGoogleMessage(msg));

    return Promise.all(pools);
  };

  private parseErrorMessage(message: string): {
    error: any;
    errorType: ILobeAgentRuntimeErrorType;
  } {
    const defaultError = {
      error: { message },
      errorType: AgentRuntimeErrorType.ProviderBizError,
    };

    if (message.includes('location is not supported'))
      return { error: { message }, errorType: AgentRuntimeErrorType.LocationNotSupportError };

    const startIndex = message.lastIndexOf('[');
    if (startIndex === -1) {
      return defaultError;
    }

    try {
      // 从开始位置截取字符串到最后
      const jsonString = message.slice(startIndex);

      // 尝试解析 JSON 字符串
      const json: GoogleChatErrors = JSON.parse(jsonString);

      const bizError = json[0];

      switch (bizError.reason) {
        case 'API_KEY_INVALID': {
          return { ...defaultError, errorType: AgentRuntimeErrorType.InvalidProviderAPIKey };
        }

        default: {
          return { error: json, errorType: AgentRuntimeErrorType.ProviderBizError };
        }
      }
    } catch {
      //
    }

    const errorObj = this.extractErrorObjectFromError(message);

    const { errorDetails } = errorObj;

    if (errorDetails) {
      return { error: errorDetails, errorType: AgentRuntimeErrorType.ProviderBizError };
    }

    return defaultError;
  }

  private buildGoogleTools(
    tools: ChatCompletionTool[] | undefined,
    payload?: ChatStreamPayload,
  ): GoogleFunctionCallTool[] | undefined {
    // 目前 Tools (例如 googleSearch) 无法与其他 FunctionCall 同时使用
    if (payload?.enabledSearch) {
      return [{ googleSearch: {} } as GoogleSearchRetrievalTool];
    }

    if (!tools || tools.length === 0) return;

    return [
      {
        functionDeclarations: tools.map((tool) => this.convertToolToGoogleTool(tool)),
      },
    ];
  }

  private convertToolToGoogleTool = (tool: ChatCompletionTool): FunctionDeclaration => {
    const functionDeclaration = tool.function;
    const parameters = functionDeclaration.parameters;
    // refs: https://github.com/lobehub/lobe-chat/pull/5002
    const properties =
      parameters?.properties && Object.keys(parameters.properties).length > 0
        ? parameters.properties
        : { dummy: { type: 'string' } }; // dummy property to avoid empty object

    return {
      description: functionDeclaration.description,
      name: functionDeclaration.name,
      parameters: {
        description: parameters?.description,
        properties: properties,
        required: parameters?.required,
        type: SchemaType.OBJECT,
      },
    };
  };

  private extractErrorObjectFromError(message: string) {
    // 使用正则表达式匹配状态码部分 [数字 描述文本]
    const regex = /^(.*?)(\[\d+ [^\]]+])(.*)$/;
    const match = message.match(regex);

    if (match) {
      const prefix = match[1].trim();
      const statusCodeWithBrackets = match[2].trim();
      const message = match[3].trim();

      // 提取状态码数字
      const statusCodeMatch = statusCodeWithBrackets.match(/\[(\d+)/);
      const statusCode = statusCodeMatch ? parseInt(statusCodeMatch[1]) : null;

      // 创建包含状态码和消息的JSON
      const resultJson = {
        message: message,
        statusCode: statusCode,
        statusCodeText: statusCodeWithBrackets,
      };

      return {
        errorDetails: resultJson,
        prefix: prefix,
      };
    }

    // 如果无法匹配，返回原始消息
    return {
      errorDetails: null,
      prefix: message,
    };
  }
}

export default LobeGoogleAI;

type GoogleChatErrors = GoogleChatError[];

interface GoogleChatError {
  '@type': string;
  'domain': string;
  'metadata': {
    service: string;
  };
  'reason': string;
}
