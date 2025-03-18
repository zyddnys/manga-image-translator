import React from "react";
import { Icon } from "@iconify/react";
import type { StatusKey } from "@/types";

type Props = {
  status: StatusKey | null;
  statusText: string;
  error: boolean;
  onRetry: () => void;
};

export const StatusDisplay: React.FC<Props> = ({
  status,
  statusText,
  error,
  onRetry,
}) => {
  if (!status) {
    return null;
  }

  if (error) {
    return (
      <div className="flex flex-col items-center gap-4 text-center">
        <div className="text-red-600 font-semibold">{statusText}</div>
        <button
          type="button"
          onClick={onRetry}
          className="px-3 py-2 text-center rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          Upload another
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center gap-2 text-center text-gray-700">
      <Icon
        icon="carbon:chevron-down"
        className="absolute top-1 right-1 text-gray-500 pointer-events-none"
      />
      <div>{statusText}</div>
    </div>
  );
};
