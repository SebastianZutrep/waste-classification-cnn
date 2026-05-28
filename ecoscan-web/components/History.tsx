"use client";

import { HistoryEntry } from "@/app/page";

interface Props {
  entries: HistoryEntry[];
  onClear: () => void;
}

export default function History({ entries, onClear }: Props) {
  return (
    <div className="bg-surface border border-border rounded-2xl overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-border">
        <span className="text-[10px] font-mono text-muted uppercase tracking-widest">Historial</span>
        <button
          onClick={onClear}
          className="text-[10px] font-mono text-muted uppercase tracking-wide hover:text-danger transition-colors"
        >
          Limpiar
        </button>
      </div>

      <div className="max-h-44 overflow-y-auto">
        {entries.length === 0 ? (
          <p className="text-center text-[11px] font-mono text-muted py-6">Sin escaneos aún</p>
        ) : (
          entries.map((e, i) => (
            <div
              key={i}
              className="slide-in flex items-center gap-2 px-4 py-2 border-b border-border last:border-0 text-[12px]"
            >
              <span
                className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                style={{ background: e.color }}
              />
              <span className="flex-1 text-[#dff0e8] truncate">{e.material}</span>
              <span className="font-mono text-[10px] text-muted">{e.confidence}%</span>
              <span className="font-mono text-[10px] text-muted ml-1">{e.time}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
