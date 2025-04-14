export const BASE_URI = "http://127.0.0.1:8000/";

export const languageOptions = [
  { value: "CHS", label: "简体中文" },
  { value: "CHT", label: "繁體中文" },
  { value: "JPN", label: "日本語" },
  { value: "ENG", label: "English" },
  { value: "KOR", label: "한국어" },
  { value: "VIN", label: "Tiếng Việt" },
  { value: "CSY", label: "čeština" },
  { value: "NLD", label: "Nederlands" },
  { value: "FRA", label: "français" },
  { value: "DEU", label: "Deutsch" },
  { value: "HUN", label: "magyar nyelv" },
  { value: "ITA", label: "italiano" },
  { value: "PLK", label: "polski" },
  { value: "PTB", label: "português" },
  { value: "ROM", label: "limba română" },
  { value: "RUS", label: "русский язык" },
  { value: "ESP", label: "español" },
  { value: "TRK", label: "Türk dili" },
  { value: "IND", label: "Indonesia" },
];

export const detectionResolutions = [1024, 1536, 2048, 2560];

export const inpaintingSizes = [516, 1024, 2048, 2560];

export const textDetectorOptions = [
  { value: "default", label: "Default" },
  { value: "ctd", label: "CTD" },
  { value: "paddle", label: "Paddle" },
];

export const inpainterOptions = [
  { value: "default", label: "Default" },
  { value: "lama_large", label: "Lama Large" },
  { value: "lama_mpe", label: "Lama MPE" },
  { value: "sd", label: "SD" },
  { value: "none", label: "None" },
  { value: "original", label: "Original" },
];

export const imageMimeTypes = [
  "image/png",
  "image/jpeg",
  "image/bmp",
  "image/webp",
];
