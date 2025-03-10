import React from "react";
import { Icon } from "@iconify/react";

type LabeledSelectOption = {
  label: string;
  value: string;
};

type LabeledSelectProps = {
  id: string;
  label: string;
  icon: string;
  title?: string;
  value: string;
  options: LabeledSelectOption[];
  onChange: (value: string) => void;
};

export const LabeledSelect: React.FC<LabeledSelectProps> = ({
  id,
  label,
  icon,
  title,
  value,
  options,
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
        <select
          id={id}
          title={title}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full appearance-none border border-gray-300 rounded pl-8 pr-6 py-1 text-sm text-gray-700 focus:border-blue-500 focus:outline-none"
        >
          {options.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
};
