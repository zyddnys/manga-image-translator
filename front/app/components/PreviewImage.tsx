import React, { useState, useEffect } from "react";

const ImagePreview = React.memo(
  ({ file, result }: { file: File; result: File | null }) => {
    // Create URL only when file or result changes
    const [imageUrl, setImageUrl] = useState<string | null>(null);
    useEffect(() => {
      const objectUrl = URL.createObjectURL(result || file);
      setImageUrl(objectUrl);
      return () => URL.revokeObjectURL(objectUrl);
    }, [file, result]);

    return (
      <img
        src={imageUrl ?? undefined}
        alt={file.name}
        className="w-full h-full object-contain rounded-lg border border-gray-200"
      />
    );
  }
);

export default ImagePreview;
