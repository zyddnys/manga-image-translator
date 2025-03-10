import type { TranslatorKey } from "@/types";

export function getTranslatorName(key: TranslatorKey): string {
  if (key === "none") return "No Text";
  return key[0].toUpperCase() + key.slice(1);
}
