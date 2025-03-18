import type { StatusKey } from "@/types";

export const fetchStatusText = (
  status: StatusKey | null,
  progress: string | null,
  queuePos: string | null
) => {
  switch (status) {
    case "upload":
      return progress ? `Uploading (${progress})` : "Uploading";
    case "pending":
      return queuePos ? `Queuing, your position is ${queuePos}` : "Processing";
    case "detection":
      return "Detecting texts";
    case "ocr":
      return "Running OCR";
    case "mask-generation":
      return "Generating text mask";
    case "inpainting":
      return "Running inpainting";
    case "upscaling":
      return "Running upscaling";
    case "translating":
      return "Translating";
    case "rendering":
      return "Rendering translated texts";
    case "finished":
      return "Downloading image";
    case "error":
      return "Something went wrong, please try again";
    case "error-upload":
      return "Upload failed, please try again";
    case "error-lang":
      return "Your target language is not supported by the chosen translator";
    case "error-translating":
      return "No text returned from the text translation service";
    case "error-too-large":
      return "Image size too large (greater than 8000x8000 px)";
    case "error-disconnect":
      return "Lost connection to server";
    default:
      return "";
  }
};
