import React from "react";
import { Icon } from "@iconify/react";

export type LabeledInputProps = {
  id: string;
  label: string;
  icon: string;
  title?: string;
  type?: React.InputHTMLAttributes<unknown>["type"];
  step?: number | string;
  value: number;
  onChange: (value: number) => void;
};

export const LabeledInput: React.FC<LabeledInputProps> = ({
  id,
  label,
  icon,
  title,
  type = "number",
  step,
  value,
  onChange,
}) => {
  return (
    <div className="flex flex-col">
      <label htmlFor={id} className="mb-1 text-sm text-gray-700 font-medium">
        {label}
      </label>
      <div className="relative">
        <Icon
          icon={icon}
          className="absolute top-1/2 left-2 -translate-y-1/2 text-gray-400"
        />
        <input
          id={id}
          title={title}
          type={type}
          step={step}
          value={value}
          onChange={(e) => {
            onChange(Number(e.target.value));
          }}
          className="w-full appearance-none border border-gray-300 rounded pl-8 pr-3 py-1 text-sm text-gray-700 focus:border-blue-500 focus:outline-none"
        />
      </div>
    </div>
  );
};
