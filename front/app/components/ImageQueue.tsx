import React from 'react';
import { Icon } from '@iconify/react';
import type { QueuedImage } from '@/types';
import PreviewImage from './PreviewImage';

interface ImageQueueProps {
  queuedImages: QueuedImage[];
  onRemoveFromQueue: (id: string) => void;
  onAddToQueue: (files: File[]) => void;
  isProcessing: boolean;
}

export const ImageQueue: React.FC<ImageQueueProps> = ({
  queuedImages,
  onRemoveFromQueue,
  onAddToQueue,
  isProcessing,
}) => {
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      onAddToQueue(files);
      // Reset input value to allow selecting the same file again
      e.target.value = '';
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      onAddToQueue(files);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-800">
          Translation Queue ({queuedImages.length})
        </h3>
        {isProcessing && (
          <span className="text-sm text-blue-600 bg-blue-100 px-2 py-1 rounded-full">
            Processing...
          </span>
        )}
      </div>

      {/* Add images area - always visible */}
      <div
        className="border-2 border-dashed border-gray-300 rounded-lg p-4 hover:border-blue-400 transition-colors"
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        onDragEnter={(e) => e.preventDefault()}
        onDragLeave={(e) => e.preventDefault()}
      >
        <label htmlFor="queue-file" className="cursor-pointer">
          <div className="text-center">
            <Icon
              icon="carbon:add"
              className="w-6 h-6 mx-auto text-gray-500 mb-2"
            />
            <div className="text-gray-600 text-sm">
              {isProcessing 
                ? 'Add more images to queue while processing'
                : 'Add images to queue'
              }
            </div>
            <div className="text-gray-400 text-xs mt-1">
              Drag & drop or click to select
            </div>
          </div>
        </label>
        <input
          id="queue-file"
          type="file"
          multiple
          accept="image/png,image/jpeg,image/bmp,image/webp"
          className="hidden"
          onChange={handleFileChange}
        />
      </div>

      {/* Queue display */}
      {queuedImages.length > 0 && (
        <div className="space-y-3">
          {queuedImages.map((queuedImage) => (
            <div
              key={queuedImage.id}
              className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg border"
            >
              {/* Image preview */}
              <div className="w-16 h-16 flex-shrink-0">
                <PreviewImage file={queuedImage.file} result={null} />
              </div>

              {/* Image info */}
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium text-gray-900 truncate">
                  {queuedImage.file.name}
                </div>
                <div className="text-xs text-gray-500">
                  Added: {queuedImage.addedAt.toLocaleTimeString()}
                </div>
                <div className="text-xs text-gray-500">
                  Size: {(queuedImage.file.size / 1024 / 1024).toFixed(2)} MB
                </div>
              </div>

              {/* Status */}
              <div className="flex items-center space-x-2">
                <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                  queuedImage.status === 'queued' 
                    ? 'bg-yellow-100 text-yellow-800'
                    : queuedImage.status === 'processing'
                    ? 'bg-blue-100 text-blue-800'
                    : queuedImage.status === 'finished'
                    ? 'bg-green-100 text-green-800'
                    : 'bg-red-100 text-red-800'
                }`}>
                  {queuedImage.status === 'queued' && 'Queued'}
                  {queuedImage.status === 'processing' && 'Processing'}
                  {queuedImage.status === 'finished' && 'Finished'}
                  {queuedImage.status === 'error' && 'Error'}
                </div>

                {/* Remove button - only show for queued items */}
                {queuedImage.status === 'queued' && (
                  <button
                    onClick={() => onRemoveFromQueue(queuedImage.id)}
                    className="p-1 text-gray-400 hover:text-red-500 transition-colors"
                    title="Remove from queue"
                  >
                    <Icon icon="carbon:close" className="w-4 h-4" />
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}; 