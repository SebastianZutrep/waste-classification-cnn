"use client";

import { PredictResult } from "@/lib/api";
import { BIN_INFO, confColor, confLevel } from "@/lib/bins";

interface Props {
  result: PredictResult | null;
  scanning: boolean;
}

export default function ResultCard({ result, scanning }: Props) {
  const conf     = result ? Math.round(result.confidence) : 0;
  const level    = confLevel(conf);
  const binInfo  = result ? BIN_INFO[result.bin] : null;
  const fillColor = confColor(conf);

  const borderClass = !result
    ? "border-border"
    : level === "high"   ? "border-green/50"
    : level === "medium" ? "border-amber/50"
    :                      "border-danger/50";

  const badgeClass = !result
    ? "text-muted bg-surface2"
    : level === "high"   ? "text-green bg-green-dim"
    : level === "medium" ? "text-amber bg-amber-dim"
    :                      "text-danger bg-danger-dim";

  return (
    <div className={`bg-surface rounded-2xl border p-5 transition-colors duration-300 ${borderClass}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <span className="text-[10px] font-mono text-muted uppercase tracking-widest">
          Material detectado
        </span>
        <span className={`font-mono text-[12px] font-bold px-2 py-1 rounded-md ${badgeClass}`}>
          {result ? `${conf}%` : "—"}
        </span>
      </div>

      {/* Material name */}
      <div className="text-xl font-medium mb-3 min-h-[1.75rem]">
        {scanning ? (
          <span className="text-muted text-sm font-mono animate-pulse">Analizando...</span>
        ) : result ? (
          result.material
        ) : (
          <span className="text-muted">—</span>
        )}
      </div>

      {/* Bin pill */}
      <div className="inline-flex items-center gap-2 bg-surface2 border border-border rounded-lg px-3 py-1.5 mb-4">
        {binInfo ? (
          <>
            <span
              className="w-2 h-2 rounded-full flex-shrink-0"
              style={{ background: binInfo.color }}
            />
            <span className="text-[11px] font-mono text-[#dff0e8]">{result!.bin}</span>
          </>
        ) : (
          <>
            <span className="w-2 h-2 rounded-full bg-muted flex-shrink-0" />
            <span className="text-[11px] font-mono text-muted">Sin clasificar</span>
          </>
        )}
      </div>

      {/* Confidence bar */}
      <div className="h-1 bg-surface2 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${conf}%`, background: result ? fillColor : "#182419" }}
        />
      </div>
    </div>
  );
}
