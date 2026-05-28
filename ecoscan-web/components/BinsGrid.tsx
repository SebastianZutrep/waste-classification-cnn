"use client";

import { BIN_INFO, ALL_BINS } from "@/lib/bins";

interface Props { activeBin: string | null; }

export default function BinsGrid({ activeBin }: Props) {
  return (
    <div className="grid grid-cols-2 gap-2">
      {ALL_BINS.map((bin) => {
        const info   = BIN_INFO[bin];
        const isActive = activeBin === bin;
        return (
          <div
            key={bin}
            className={`bg-surface rounded-xl border p-3 text-center transition-all duration-300 ${
              isActive ? "bg-surface2" : "border-border"
            }`}
            style={isActive ? { borderColor: info.color + "80" } : {}}
          >
            <div className="text-xl mb-1">{info.emoji}</div>
            <p className={`text-[9px] font-mono leading-snug tracking-wide ${
              isActive ? "text-[#dff0e8]" : "text-muted"
            }`}>
              {info.label}
            </p>
          </div>
        );
      })}
    </div>
  );
}
