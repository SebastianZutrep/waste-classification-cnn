"use client";

import { useState, useCallback, useEffect } from "react";
import CameraView from "@/components/CameraView";
import ResultCard from "@/components/ResultCard";
import BinsGrid from "@/components/BinsGrid";
import ScoreBreakdown from "@/components/ScoreBreakdown";
import History from "@/components/History";
import ServerStatus from "@/components/ServerStatus";
import { predict, PredictResult } from "@/lib/api";

export interface HistoryEntry {
  material: string;
  bin: string;
  confidence: number;
  color: string;
  time: string;
}

export default function Home() {
  const [result, setResult] = useState<PredictResult | null>(null);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  const handleFrame = useCallback(async (base64: string) => {
    if (scanning) return;
    setScanning(true);
    setError(null);
    try {
      const data = await predict(base64);
      setResult(data);
      setHistory((prev) => {
        const entry: HistoryEntry = {
          material: data.material,
          bin: data.bin,
          confidence: Math.round(data.confidence),
          color: data.confidence >= 80 ? "#1D9E75" : data.confidence >= 50 ? "#BA7517" : "#A32D2D",
          time: new Date().toLocaleTimeString("es", { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
        };
        return [entry, ...prev].slice(0, 30);
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Error desconocido");
    } finally {
      setScanning(false);
    }
  }, [scanning]);

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      {/* Header */}
      <header className="w-full px-6 py-4 flex items-center gap-3 border-b border-border bg-bg/80 backdrop-blur sticky top-0 z-10">
        <div className="w-8 h-8 rounded-lg border-2 border-green flex items-center justify-center">
          <svg className="w-4 h-4 fill-green" viewBox="0 0 24 24">
            <path d="M12 2L4 6v6c0 5.55 3.84 10.74 8 12 4.16-1.26 8-6.45 8-12V6l-8-4z"/>
          </svg>
        </div>
        <span className="font-mono text-sm font-bold tracking-widest text-green uppercase">EcoScan</span>
        <div className="ml-auto">
          <ServerStatus />
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 w-full max-w-5xl mx-auto px-4 py-6 grid grid-cols-1 lg:grid-cols-[1fr_300px] gap-5 items-start">
        {/* Left — camera */}
        <div className="flex flex-col gap-4">
          <CameraView onFrame={handleFrame} scanning={scanning} />
          {error && (
            <div className="bg-danger-dim border border-danger/30 rounded-lg px-4 py-3 text-xs font-mono text-danger">
              ⚠ {error}
            </div>
          )}
        </div>

        {/* Right — results */}
        <div className="flex flex-col gap-4">
          <ResultCard result={result} scanning={scanning} />
          <BinsGrid activeBin={result?.bin ?? null} />
          {result?.all_scores && <ScoreBreakdown scores={result.all_scores} topMaterial={result.material} />}
          <History entries={history} onClear={() => setHistory([])} />
        </div>
      </main>
    </div>
  );
}
