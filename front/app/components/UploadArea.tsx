import React from "react";
import { Icon } from "@iconify/react";
import { fetchStatusText } from "@/utils/fetchStatusText";
import type { FileStatus } from "@/types";

export interface UploadAreaProps {
  files: File[];
  fileStatuses: Map<string, FileStatus>;
  isProcessing: boolean;
  isProcessingAllFinished: boolean;

  handleFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  handleDrop: (e: React.DragEvent<HTMLLabelElement>) => void;
  handleSubmit: () => Promise<void>;
  clearForm: () => void;
  removeFile: (fileName: string) => void;
}

/**
 * ファイルのアップロードやプレビュー、翻訳開始ボタンをまとめたコンポーネント
 */
export const UploadArea: React.FC<UploadAreaProps> = ({
  files,
  fileStatuses,
  isProcessing,
  isProcessingAllFinished,
  handleFileChange,
  handleDrop,
  handleSubmit,
  clearForm,
  removeFile,
}) => {
  return (
    <div className="space-y-4 max-w-[1200px] mx-auto">
      {!isProcessing && !isProcessingAllFinished && (
        <form>
          {/* Upload area */}
          <label
            htmlFor="file"
            className="block p-4 border-2 border-dashed border-gray-300 rounded-lg"
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            onDragEnter={(e) => e.preventDefault()}
            onDragLeave={(e) => e.preventDefault()}
          >
            <div className="text-center p-8">
              <Icon
                icon="carbon:cloud-upload"
                className="w-8 h-8 mx-auto text-gray-500"
              />
              <div className="mt-2 text-gray-600">
                Drop images here or click to select and upload images
              </div>
            </div>
            <input
              id="file"
              type="file"
              multiple
              accept="image/png,image/jpeg,image/bmp,image/webp"
              className="hidden"
              onChange={handleFileChange}
            />
          </label>
        </form>
      )}
      {/* Image grid */}
      {files.length > 0 && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-8">
            {files.map((file) => {
              const status = fileStatuses.get(file.name);
              return (
                <div key={file.name} className="relative">
                  <div className="relative w-full min-h-[400px] max-h-[600px] group">
                    {/* Delete button - displayed when uploading */}
                    {!isProcessing && !isProcessingAllFinished && (
                      <button
                        type="button"
                        onClick={() => removeFile(file.name)}
                        className="absolute top-2 right-2 z-10 p-2 bg-red-500 rounded-lg text-white opacity-75 group-hover:opacity-100 transition-opacity hover:bg-red-600 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2"
                      >
                        <Icon icon="carbon:trash-can" className="w-5 h-5" />
                      </button>
                    )}

                    <img
                      src={
                        status?.result
                          ? URL.createObjectURL(status.result)
                          : URL.createObjectURL(file)
                      }
                      alt={file.name}
                      className="w-full h-full object-contain rounded-lg border border-gray-200"
                    />

                    {/* Status overlay */}
                    {status && status.status !== "finished" && (
                      <div className="absolute inset-0 bg-black/50 flex items-center justify-center rounded-lg">
                        <div className="text-white text-center px-6 py-3 text-lg">
                          {fetchStatusText(
                            status.status,
                            status.progress,
                            status.queuePos,
                            status.error
                          )}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* File info */}
                  <div className="mt-3 flex justify-between items-center px-2">
                    <div className="text-base truncate max-w-[80%] text-gray-700">
                      {file.name}
                    </div>
                    {status?.error ? (
                      <div className="text-red-500 text-base flex items-center">
                        <Icon icon="carbon:warning" className="w-5 h-5 mr-1" />
                        Error
                      </div>
                    ) : status?.status === "finished" ? (
                      <div className="text-green-500 flex items-center">
                        <Icon icon="carbon:checkmark" className="w-5 h-5" />
                      </div>
                    ) : null}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Submit button */}
          {!isProcessing && !isProcessingAllFinished && (
            <button
              type="button"
              className="w-full mt-8 py-4 px-6 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-lg"
              disabled={files.length === 0}
              onClick={handleSubmit}
            >
              Translate All Images
            </button>
          )}
          {isProcessingAllFinished && (
            <button
              onClick={clearForm}
              className="w-full mt-8 py-4 px-6 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-lg"
            >
              Start Over
            </button>
          )}
        </>
      )}
    </div>
  );
};
