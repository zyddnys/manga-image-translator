export type StatusKey =
  | "upload"
  | "pending"
  | "detection"
  | "ocr"
  | "mask-generation"
  | "inpainting"
  | "upscaling"
  | "translating"
  | "rendering"
  | "finished"
  | "error"
  | "error-upload"
  | "error-lang"
  | "error-translating"
  | "error-too-large"
  | "error-disconnect"
  | null;

export type TranslatorKey =
  | "youdao"
  | "baidu"
  | "deepl"
  | "papago"
  | "caiyun"
  | "sakura"
  | "offline"
  | "openai"
  | "deepseek"
  | "none";

export const validTranslators: TranslatorKey[] = [
  "youdao",
  "baidu",
  "deepl",
  "papago",
  "caiyun",
  "sakura",
  "offline",
  "openai",
  "deepseek",
  "none",
];
