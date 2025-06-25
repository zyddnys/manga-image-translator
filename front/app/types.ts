export type StatusKey =
  | "upload"
  | "pending"
  | "detection"
  | "ocr"
  | "textline_merge"
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

export interface ChunkProcessingResult {
  updatedBuffer: Uint8Array;
}

export const processingStatuses = [
  "upload",
  "pending",
  "detection",
  "ocr",
  "textline_merge",
  "mask-generation",
  "inpainting",
  "upscaling",
  "translating",
  "rendering",
];

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
  | "groq"  
  | "gemini"  
  | "custom_openai"  
  | "nllb"  
  | "nllb_big"  
  | "sugoi"  
  | "jparacrawl"  
  | "jparacrawl_big"  
  | "m2m100"  
  | "m2m100_big"  
  | "mbart50"  
  | "qwen2"  
  | "qwen2_big"  
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
  "groq",  
  "gemini",  
  "custom_openai",  
  "nllb",  
  "nllb_big",  
  "sugoi",  
  "jparacrawl",  
  "jparacrawl_big",  
  "m2m100",  
  "m2m100_big",  
  "mbart50",  
  "qwen2",  
  "qwen2_big",  
  "none",  
];  

export interface FileStatus {
  status: StatusKey | null;
  progress: string | null;
  queuePos: string | null;
  result: Blob | null;
  error: string | null;
}
