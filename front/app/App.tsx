import React, { useState, useEffect, useCallback, useMemo } from "react";
import {
  type StatusKey,
  processingStatuses,
  type TranslatorKey,
  type FileStatus,
} from "@/types";
import { BASE_URI, imageMimeTypes } from "@/config";
import { OptionsPanel } from "@/components/OptionsPanel";
import { UploadArea } from "@/components/UploadArea";
import { Header } from "@/components/Header";

export const App: React.FC = () => {
  // File statuses
  const [fileStatuses, setFileStatuses] = useState<Map<string, FileStatus>>(
    new Map()
  );
  const [files, setFiles] = useState<File[]>([]);
  const [results, setResults] = useState<Blob[]>([]);

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

  // Helper to update status for a specific file
  const updateFileStatus = useCallback(
    (fileId: string, update: Partial<FileStatus>) => {
      setFileStatuses((prev) => {
        const newStatuses = new Map(prev);
        const currentStatus = newStatuses.get(fileId) || {
          status: null,
          progress: null,
          queuePos: null,
          result: null,
          error: null,
        };
        const updatedStatus = { ...currentStatus, ...update };
        newStatuses.set(fileId, updatedStatus);
        return newStatuses;
      });
    },
    []
  );

  /** フォーム再セット */
  const clearForm = useCallback(() => {
    setFiles([]);
    setResults([]);
    setFileStatuses(new Map());
  }, []);

  const isProcessing = useMemo(() => {
    // If there are no files or no statuses, we're not processing
    if (files.length === 0 || fileStatuses.size === 0) return false;

    // Check if any file has a processing status
    return Array.from(fileStatuses.values()).some((fileStatus) => {
      if (!fileStatus || fileStatus.status === null) return false;
      return processingStatuses.includes(fileStatus.status);
    });
  }, [files, fileStatuses]);

  const isProcessingAllFinished = useMemo(() => {
    // If there are no files or no statuses, we're not finished
    if (files.length === 0 || fileStatuses.size === 0) return false;

    // Check if all files are finished
    return Array.from(fileStatuses.values()).every((status) => {
      if (!status || status.status === null) return false;
      return status.status === "finished";
    });
  }, [files, fileStatuses]);

  /** ドラッグ＆ドロップ対応 */
  const handleDrop = useCallback((e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    const droppedFiles = Array.from(e.dataTransfer?.files || []);
    const validFiles = droppedFiles.filter((file) =>
      imageMimeTypes.includes(file.type)
    );
    setFiles((prev) => [...prev, ...validFiles]);
  }, []);

  /** ファイル選択時 */
  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selectedFiles = Array.from(e.target.files || []);
      const validFiles = selectedFiles.filter((file) =>
        imageMimeTypes.includes(file.type)
      );
      setFiles((prev) => [...prev, ...validFiles]);
    },
    []
  );

  // Remove file handler
  const removeFile = useCallback((fileName: string) => {
    // Remove from files array
    setFiles((prev) => prev.filter((file) => file.name !== fileName));

    // Remove from status map
    setFileStatuses((prev) => {
      const newStatuses = new Map(prev);
      newStatuses.delete(fileName);
      return newStatuses;
    });
  }, []);

  /** クリップボード ペースト対応 */
  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items || [];
      for (const item of items) {
        if (item.kind === "file") {
          const pastedFile = item.getAsFile();
          if (pastedFile && imageMimeTypes.includes(pastedFile.type)) {
            setFiles((prev) => [...prev, pastedFile]);
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
  const handleSubmit = useCallback(async () => {
    // Track readers to ensure cleanup
    const readers: ReadableStreamDefaultReader<Uint8Array>[] = [];

    try {
      // If no files, do nothing
      if (files.length === 0) return;

      // Initialize status for all files
      const newStatuses = new Map();
      files.forEach((file) => {
        newStatuses.set(file.name, {
          status: null,
          progress: null,
          queuePos: null,
          result: null,
          error: null,
        });
      });
      setFileStatuses(newStatuses);

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

      // Process all files in parallel
      await Promise.all(
        files.map(async (file) => {
          const formData = new FormData();
          formData.append("image", file);
          formData.append("config", config);

          try {
            const response = await fetch(
              `${BASE_URI}translate/with-form/image/stream`,
              {
                method: "POST",
                body: formData,
              }
            );

            if (response.status !== 200) {
              throw new Error("Upload failed");
            }

            const reader = response.body?.getReader();
            if (!reader) return;
            readers.push(reader);

            let fileBuffer = new Uint8Array();
            let processingPromise = Promise.resolve(); // Track ongoing processing

            const processChunk = async (value: Uint8Array, fileId: string) => {
              try {
                // Wait for any previous processing to complete
                await processingPromise;

                // Create a new processing promise for this chunk
                processingPromise = (async () => {
                  if (fileStatuses.get(fileId)?.error) return;

                  const newBuffer = new Uint8Array(
                    fileBuffer.length + value.length
                  );
                  newBuffer.set(fileBuffer);
                  newBuffer.set(value, fileBuffer.length);
                  fileBuffer = newBuffer;

                  while (fileBuffer.length >= 5) {
                    const dataSize = new DataView(fileBuffer.buffer).getUint32(
                      1,
                      false
                    );
                    const totalSize = 5 + dataSize;
                    if (fileBuffer.length < totalSize) break;

                    const statusCode = fileBuffer[0];
                    const data = fileBuffer.slice(5, totalSize);
                    const decoder = new TextDecoder("utf-8");
                    const decodedData = decoder.decode(data);

                    processStatusUpdate(statusCode, decodedData, fileId, data);

                    fileBuffer = fileBuffer.slice(totalSize);
                  }
                })().catch((error) => {
                  console.error(`Error processing chunk for ${fileId}:`, error);
                  updateFileStatus(fileId, {
                    status: "error",
                    error: error.message || "Error processing chunk",
                  });
                });

                await processingPromise;
              } catch (error) {
                console.error(
                  `Fatal error processing chunk for ${fileId}:`,
                  error
                );
              }
            };

            while (true) {
              const { done, value } = await reader.read();
              if (done || !value) break;
              await processChunk(value, file.name);
            }
          } catch (err) {
            console.error("Error processing file: ", file.name, err);
            updateFileStatus(file.name, {
              status: "error",
              error: err instanceof Error ? err.message : "Unknown error",
            });
          }
        })
      );
    } finally {
      // Cleanup all readers
      await Promise.all(readers.map((reader) => reader.cancel()));
    }
  }, [
    files,
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
    updateFileStatus,
  ]);

  // Create a separate function for clarity
  const processStatusUpdate = (
    statusCode: number,
    decodedData: string,
    fileId: string,
    data: Uint8Array
  ) => {
    switch (statusCode) {
      case 0: // 結果が返ってきた
        updateFileStatus(fileId, {
          status: "finished",
          result: new Blob([data], { type: "image/png" }),
        });
        break;
      case 1: // 翻訳中
        const newStatus = decodedData as StatusKey;
        updateFileStatus(fileId, { status: newStatus });
        break;
      case 2: // エラー
        updateFileStatus(fileId, {
          status: "error",
          error: decodedData,
        });
        break;
      case 3: // キューに追加された
        updateFileStatus(fileId, {
          status: "pending",
          queuePos: decodedData,
        });
        break;
      case 4: // キューがクリアされた
        updateFileStatus(fileId, {
          status: "pending",
          queuePos: null,
        });
        break;
      default: // 未知のステータスコード
        console.warn(`Unknown status code ${statusCode} for file ${fileId}`);
        break;
    }
  };

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
            files={files}
            fileStatuses={fileStatuses}
            isProcessing={isProcessing}
            isProcessingAllFinished={isProcessingAllFinished}
            handleFileChange={handleFileChange}
            handleDrop={handleDrop}
            handleSubmit={handleSubmit}
            clearForm={clearForm}
            removeFile={removeFile}
          />
        </div>
      </div>
    </div>
  );
};

export default App;
