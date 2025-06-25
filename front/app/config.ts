export const languageOptions = [  
  { value: "CHS", label: "简体中文" },  
  { value: "CHT", label: "繁體中文" },  
  { value: "CSY", label: "čeština" },  
  { value: "NLD", label: "Nederlands" },  
  { value: "ENG", label: "English" },  
  { value: "FRA", label: "français" },  
  { value: "DEU", label: "Deutsch" },  
  { value: "HUN", label: "magyar nyelv" },  
  { value: "ITA", label: "italiano" },  
  { value: "JPN", label: "日本語" },  
  { value: "KOR", label: "한국어" },  
  { value: "PLK", label: "polski" },  
  { value: "PTB", label: "português" },  
  { value: "ROM", label: "limba română" },  
  { value: "RUS", label: "русский язык" },  
  { value: "ESP", label: "español" },  
  { value: "TRK", label: "Türk dili" },  
  { value: "UKR", label: "українська мова" },  
  { value: "VIN", label: "Tiếng Việt" },  
  { value: "ARA", label: "العربية" },  
  { value: "CNR", label: "crnogorski jezik" },  
  { value: "SRP", label: "српски језик" },  
  { value: "HRV", label: "hrvatski jezik" },  
  { value: "THA", label: "ภาษาไทย" },  
  { value: "IND", label: "Indonesia" },  
  { value: "FIL", label: "Wikang Filipino" }  
];  

export const detectionResolutions = [1024, 1536, 2048, 2560];

export const inpaintingSizes = [516, 1024, 2048, 2560];

export const textDetectorOptions = [
  { value: "default", label: "Default" },
  { value: "ctd", label: "CTD" },
  { value: "paddle", label: "Paddle" },
  { value: "switch", label: "Switch" },
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