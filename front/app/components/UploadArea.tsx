import React from "react";
import { Icon } from "@iconify/react";
import { StatusDisplay } from "@/components/StatusDisplay";
import type { StatusKey } from "@/types";

type Props = {
  file: File | null;
  fileUri: string | null;
  resultUri: string | null;
  status: StatusKey | null;
  statusText: string;
  error: boolean;

  handleFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  handleDrop: (e: React.DragEvent<HTMLLabelElement>) => void;
  handleSubmit: (e: React.FormEvent<HTMLFormElement>) => Promise<void>;
  clearForm: () => void;
};

/**
 * ファイルのアップロードやプレビュー、翻訳開始ボタンをまとめたコンポーネント
 */
export const UploadArea: React.FC<Props> = ({
  file,
  fileUri,
  resultUri,
  status,
  statusText,
  error,
  handleFileChange,
  handleDrop,
  handleSubmit,
  clearForm,
}) => {
  // 翻訳結果があるかどうか
  const hasResult = !!resultUri;

  return (
    <div>
      {hasResult ? (
        // 結果表示エリア
        <div className="flex flex-col items-center space-y-4">
          <img
            className="max-w-full max-h-[50vh] rounded-md"
            src={resultUri || ""}
            alt="Result"
          />
          <button
            type="button"
            onClick={clearForm}
            className="px-3 py-2 text-center rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            Upload another
          </button>
        </div>
      ) : status ? (
        // 処理中/ステータス表示エリア
        <div className="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-md p-8 h-72">
          <StatusDisplay
            status={status}
            statusText={statusText}
            error={error}
            onRetry={clearForm}
          />
        </div>
      ) : (
        // 画像未アップロード時のアップロードボックス
        <form onSubmit={handleSubmit}>
          <label
            htmlFor="file"
            className="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-md px-8 cursor-pointer hover:border-blue-400 transition-colors"
            onDragOver={(e) => e.preventDefault()}
            onDragEnter={(e) => e.preventDefault()}
            onDragLeave={(e) => e.preventDefault()}
            onDrop={handleDrop}
          >
            {file ? (
              // 選択済みファイルあり → プレビュー + 翻訳ボタン
              <div className="flex flex-col items-center gap-4 my-10 text-center">
                <div className="text-gray-700">
                  <Icon
                    icon="carbon:image-search"
                    className="inline-block mr-2 text-xl"
                  />
                  File Preview
                </div>
                <img
                  className="max-w-[18rem] max-h-[18rem] rounded-md border border-gray-200"
                  src={fileUri || ""}
                  alt="Preview"
                />
                <button
                  type="submit"
                  className="px-4 py-2 rounded-md bg-blue-600 text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  Translate
                </button>
                <div className="text-sm text-gray-500">
                  Click the empty space or paste/drag a new one to replace
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-2 my-20 text-center">
                <Icon
                  icon="carbon:cloud-upload"
                  className="w-8 h-8 text-gray-500"
                />
                <div className="text-gray-600">
                  Paste an image, click to select one, or drag and drop here
                </div>
              </div>
            )}
            <input
              id="file"
              type="file"
              accept="image/png,image/jpeg,image/bmp,image/webp"
              className="hidden"
              onChange={handleFileChange}
            />
          </label>
        </form>
      )}
    </div>
  );
};
