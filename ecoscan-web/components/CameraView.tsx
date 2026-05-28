"use client";

import { useEffect, useRef, useState, useCallback } from "react";

interface Props {
  onFrame: (base64: string) => void;
  scanning: boolean;
}

interface CamDevice { deviceId: string; label: string; }

export default function CameraView({ onFrame, scanning }: Props) {
  const videoRef  = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const autoRef   = useRef<ReturnType<typeof setInterval> | null>(null);

  const [devices,    setDevices]    = useState<CamDevice[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [active,     setActive]     = useState(false);
  const [autoMode,   setAutoMode]   = useState(false);
  const [camError,   setCamError]   = useState<string | null>(null);

  // Enumerate cameras (need permission first so labels appear)
  useEffect(() => {
    (async () => {
      try {
        const tmp = await navigator.mediaDevices.getUserMedia({ video: true });
        tmp.getTracks().forEach((t) => t.stop());
        const all  = await navigator.mediaDevices.enumerateDevices();
        const cams = all
          .filter((d) => d.kind === "videoinput")
          .map((d, i) => ({ deviceId: d.deviceId, label: d.label || `Cámara ${i + 1}` }));
        setDevices(cams);
        if (cams.length) setSelectedId(cams[0].deviceId);
      } catch {
        setCamError("Sin permiso para acceder a las cámaras.");
      }
    })();
  }, []);

  const startCamera = useCallback(async (deviceId: string) => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 960 } },
      });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setActive(true);
      setCamError(null);
    } catch {
      setCamError("No se pudo abrir la cámara seleccionada.");
      setActive(false);
    }
  }, []);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setActive(false);
    if (autoRef.current) clearInterval(autoRef.current);
    setAutoMode(false);
  }, []);

  const capture = useCallback(() => {
    const v = videoRef.current;
    const c = canvasRef.current;
    if (!v || !c || !active) return;
    c.width  = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d")!.drawImage(v, 0, 0);
    onFrame(c.toDataURL("image/jpeg", 0.88));
  }, [active, onFrame]);

  const toggleAuto = () => {
    if (autoMode) {
      if (autoRef.current) clearInterval(autoRef.current);
      setAutoMode(false);
    } else {
      capture();
      autoRef.current = setInterval(capture, 4500);
      setAutoMode(true);
    }
  };

  // Change camera if already streaming
  const handleSelect = (id: string) => {
    setSelectedId(id);
    if (active) startCamera(id);
  };

  useEffect(() => () => { if (autoRef.current) clearInterval(autoRef.current); }, []);

  return (
    <div className="flex flex-col gap-3">

      {/* Camera selector */}
      <div className="flex items-center gap-2 bg-surface border border-border rounded-lg px-3 py-2">
        <span className="text-[10px] font-mono text-muted uppercase tracking-widest whitespace-nowrap">Cámara</span>
        <select
          value={selectedId}
          onChange={(e) => handleSelect(e.target.value)}
          className="flex-1 bg-transparent text-[#dff0e8] text-xs font-mono outline-none cursor-pointer min-w-0"
        >
          {devices.length === 0 && <option>Sin cámaras detectadas</option>}
          {devices.map((d) => (
            <option key={d.deviceId} value={d.deviceId} className="bg-surface">
              {d.label}
            </option>
          ))}
        </select>
      </div>

      {/* Video */}
      <div
        className="relative bg-black rounded-2xl overflow-hidden border border-border"
        style={{ aspectRatio: "4/3" }}
      >
        <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
        <canvas ref={canvasRef} className="hidden" />

        {/* Scan frame corners */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="relative w-52 h-52 border border-green/40 rounded-xl">
            <span className="absolute -top-px -left-px  w-5 h-5 border-t-2 border-l-2 border-green rounded-tl" />
            <span className="absolute -top-px -right-px w-5 h-5 border-t-2 border-r-2 border-green rounded-tr" />
            <span className="absolute -bottom-px -left-px  w-5 h-5 border-b-2 border-l-2 border-green rounded-bl" />
            <span className="absolute -bottom-px -right-px w-5 h-5 border-b-2 border-r-2 border-green rounded-br" />
            {active && (
              <div
                className="scan-line absolute left-2 right-2 h-px opacity-0"
                style={{ background: "linear-gradient(to right, transparent, #1D9E75, transparent)" }}
              />
            )}
          </div>
        </div>

        {/* Offline overlay */}
        {!active && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/60">
            <p className="text-xs font-mono text-muted">Cámara no iniciada</p>
          </div>
        )}

        {/* Scanning border flash */}
        {scanning && (
          <div className="absolute inset-0 rounded-2xl border-2 border-amber pointer-events-none animate-pulse" />
        )}
      </div>

      {/* Camera error */}
      {camError && <p className="text-[11px] font-mono text-danger">{camError}</p>}

      {/* Controls */}
      <div className="flex gap-2">
        <button
          onClick={() => (active ? stopCamera() : startCamera(selectedId))}
          disabled={devices.length === 0}
          className="flex-1 py-2.5 rounded-lg font-mono text-[11px] font-bold tracking-widest uppercase
            bg-green text-white border border-green
            hover:bg-green-dark hover:border-green-dark
            disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        >
          {active ? "Detener" : "Iniciar cámara"}
        </button>

        <button
          onClick={capture}
          disabled={!active || scanning}
          className="flex-1 py-2.5 rounded-lg font-mono text-[11px] font-bold tracking-widest uppercase
            bg-transparent text-green border border-green
            hover:bg-green-dim
            disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        >
          {scanning ? (
            <span className="flex items-center justify-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-amber dot-1 inline-block" />
              <span className="w-1.5 h-1.5 rounded-full bg-amber dot-2 inline-block" />
              <span className="w-1.5 h-1.5 rounded-full bg-amber dot-3 inline-block" />
            </span>
          ) : "Escanear"}
        </button>

        <button
          onClick={toggleAuto}
          disabled={!active}
          className={`px-4 py-2.5 rounded-lg font-mono text-[11px] font-bold tracking-widest uppercase border transition-colors
            disabled:opacity-30 disabled:cursor-not-allowed
            ${autoMode
              ? "text-amber border-amber bg-amber-dim"
              : "text-muted border-border bg-transparent hover:border-green/40"}`}
        >
          {autoMode ? "Auto ON" : "Auto"}
        </button>
      </div>
    </div>
  );
}
