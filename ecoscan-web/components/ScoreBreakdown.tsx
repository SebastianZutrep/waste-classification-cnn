"use client";

interface Props {
  scores: Record<string, number>;
  topMaterial: string;
}

export default function ScoreBreakdown({ scores, topMaterial }: Props) {
  const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const max    = sorted[0]?.[1] ?? 1;

  return (
    <div className="bg-surface border border-border rounded-2xl p-4">
      <p className="text-[10px] font-mono text-muted uppercase tracking-widest mb-3">
        Todas las clases
      </p>
      <div className="flex flex-col gap-2">
        {sorted.map(([label, pct]) => {
          const isTop = label === topMaterial;
          const barW  = max > 0 ? Math.round((pct / max) * 100) : 0;
          return (
            <div key={label} className="flex items-center gap-2">
              <span
                className={`text-[11px] w-32 flex-shrink-0 truncate ${
                  isTop ? "text-[#dff0e8] font-medium" : "text-muted"
                }`}
                title={label}
              >
                {label}
              </span>
              <div className="flex-1 h-1 bg-surface2 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${barW}%`,
                    background: isTop ? "#1D9E75" : "rgba(29,158,117,0.25)",
                  }}
                />
              </div>
              <span className="text-[10px] font-mono text-muted w-9 text-right flex-shrink-0">
                {pct.toFixed(1)}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
