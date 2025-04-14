import React, { useState, useEffect, useCallback, useMemo } from "react";
import type { StatusKey, TranslatorKey } from "@/types";
import { BASE_URI, imageMimeTypes } from "@/config";
import { OptionsPanel } from "@/components/OptionsPanel";
import { UploadArea } from "@/components/UploadArea";
import { Header } from "@/components/Header";
import { fetchStatusText } from "@/utils/fetchStatusText";

export const App: React.FC = () => {
  // アップロードファイル/結果格納
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<Blob | null>(null);

  // ステータス管理
  const [status, setStatus] = useState<StatusKey>(null);
  const [queuePos, setQueuePos] = useState<string | null>(null);
  const [progress, setProgress] = useState<string | null>(null);

  // 翻訳オプション系
  const [detectionResolution, setDetectionResolution] = useState("1536");
  const [textDetector, setTextDetector] = useState("default");
  const [renderTextDirection, setRenderTextDirection] = useState("auto");
  const [translator, setTranslator] = useState<TranslatorKey>("youdao");
  const [targetLanguage, setTargetLanguage] = useState("CHS");

  const [inpaintingSize, setInpaintingSize] = useState("2048");
  const [customUnclipRatio, setCustomUnclipRatio] = useState<number>(2.3);
  const [customBoxThreshold, setCustomBoxThreshold] = useState<number>(0.7);
  const [maskDilationOffset, setMaskDilationOffset] = useState<number>(30);
  const [inpainter, setInpainter] = useState("default");

  // 画像プレビュー用 URL
  const fileUri = file ? URL.createObjectURL(file) : null;
  // 翻訳後の画像表示用 URL
  const resultUri = result ? URL.createObjectURL(result) : null;

  // エラー状態か判定
  const error = useMemo(() => !!status?.startsWith("error"), [status]);

  // ステータス文言のリアルタイム値
  const statusText = useMemo(
    () => fetchStatusText(status, progress, queuePos),
    [status, progress, queuePos]
  );

  /** フォーム再セット */
  const clearForm = useCallback(() => {
    setFile(null);
    setResult(null);
    setStatus(null);
    setProgress(null);
    setQueuePos(null);
  }, []);

  /** ドラッグ＆ドロップ対応 */
  const handleDrop = useCallback((e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer?.files?.[0];
    if (droppedFile && imageMimeTypes.includes(droppedFile.type)) {
      setFile(droppedFile);
    }
  }, []);

  /** ファイル選択時 */
  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selected = e.target.files?.[0];
      if (selected && imageMimeTypes.includes(selected.type)) {
        setFile(selected);
      }
    },
    []
  );

  /** クリップボード ペースト対応 */
  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items || [];
      for (const item of items) {
        if (item.kind === "file") {
          const pastedFile = item.getAsFile();
          if (pastedFile && imageMimeTypes.includes(pastedFile.type)) {
            setFile(pastedFile);
            break;
          }
        }
      }
    };

    window.addEventListener("paste", handlePaste as EventListener);
    return () =>
      window.removeEventListener("paste", handlePaste as EventListener);
  }, []);

  /**
   * フォーム送信 (翻訳リクエスト)
   */
  const handleSubmit = useCallback(
    async (e: React.FormEvent<HTMLFormElement>) => {
      e.preventDefault();
      if (!file) return;

      setStatus("upload");
      setProgress(null);
      setQueuePos(null);
      setResult(null);

      const formData = new FormData();
      formData.append("image", file);

      const config = JSON.stringify({
        detector: {
          detector: textDetector,
          detection_size: detectionResolution,
          box_threshold: customBoxThreshold,
          unclip_ratio: customUnclipRatio,
        },
        render: {
          direction: renderTextDirection,
        },
        translator: {
          translator: translator,
          target_lang: targetLanguage,
        },
        inpainter: {
          inpainter: inpainter,
          inpainting_size: inpaintingSize,
        },
        mask_dilation_offset: maskDilationOffset,
      });

      formData.append("config", config);

      let buffer = new Uint8Array();

      const processChunk = (value: Uint8Array) => {
        if (error) return; // 既にエラーの場合は続行しない

        const newBuffer = new Uint8Array(buffer.length + value.length);
        newBuffer.set(buffer);
        newBuffer.set(value, buffer.length);
        buffer = newBuffer;

        while (buffer.length >= 5) {
          const dataSize = new DataView(buffer.buffer).getUint32(1, false);
          const totalSize = 5 + dataSize;
          if (buffer.length < totalSize) {
            break;
          }

          const statusCode = buffer[0];
          const data = buffer.slice(5, totalSize);
          const decoder = new TextDecoder("utf-8");

          switch (statusCode) {
            case 0:
              // 結果画像 (PNG)
              setResult(new Blob([data], { type: "image/png" }));
              setStatus(null);
              break;
            case 1:
              // ステータス文字列
              setStatus(decoder.decode(data) as StatusKey);
              break;
            case 2:
              // エラー
              setStatus("error");
              console.error(decoder.decode(data));
              break;
            case 3:
              // キュー内順位
              setStatus("pending");
              setQueuePos(decoder.decode(data));
              break;
            case 4:
              // キュークリア
              setStatus("pending");
              setQueuePos(null);
              break;
            default:
              break;
          }

          buffer = buffer.slice(totalSize);
        }
      };

      try {
        const response = await fetch(
          `${BASE_URI}translate/with-form/image/stream`,
          {
            method: "POST",
            body: formData,
          }
        );

        if (response.status !== 200) {
          setStatus("error-upload");
          setStatus("pending");
          return;
        }

        const reader = response.body?.getReader();
        if (!reader) return;

        while (true) {
          const { done, value } = await reader.read();
          if (done || !value) break;
          processChunk(value);
        }
      } catch (err) {
        console.error(err);
        setStatus("error-disconnect");
      }
    },
    [
      file,
      textDetector,
      detectionResolution,
      customBoxThreshold,
      customUnclipRatio,
      renderTextDirection,
      translator,
      targetLanguage,
      inpainter,
      inpaintingSize,
      maskDilationOffset,
      error,
    ]
  );

  return (
    <div>
      <Header />
      <div className="bg-gray-100 min-h-screen flex flex-col pt-10 items-center">
        <div className="bg-white shadow-md rounded-lg p-6 w-full max-w-4xl space-y-6">
          <OptionsPanel
            detectionResolution={detectionResolution}
            textDetector={textDetector}
            renderTextDirection={renderTextDirection}
            translator={translator}
            targetLanguage={targetLanguage}
            inpaintingSize={inpaintingSize}
            customUnclipRatio={customUnclipRatio}
            customBoxThreshold={customBoxThreshold}
            maskDilationOffset={maskDilationOffset}
            inpainter={inpainter}
            setDetectionResolution={setDetectionResolution}
            setTextDetector={setTextDetector}
            setRenderTextDirection={setRenderTextDirection}
            setTranslator={setTranslator}
            setTargetLanguage={setTargetLanguage}
            setInpaintingSize={setInpaintingSize}
            setCustomUnclipRatio={setCustomUnclipRatio}
            setCustomBoxThreshold={setCustomBoxThreshold}
            setMaskDilationOffset={setMaskDilationOffset}
            setInpainter={setInpainter}
          />
          <UploadArea
            file={file}
            fileUri={fileUri}
            resultUri={resultUri}
            status={status}
            statusText={statusText}
            error={error}
            handleFileChange={handleFileChange}
            handleDrop={handleDrop}
            handleSubmit={handleSubmit}
            clearForm={clearForm}
          />
        </div>
      </div>
    </div>
  );
};

export default App;
