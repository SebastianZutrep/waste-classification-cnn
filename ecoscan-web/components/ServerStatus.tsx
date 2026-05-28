"use client";

import { useEffect, useState } from "react";
import { checkHealth } from "@/lib/api";

type Status = "checking" | "ok" | "error";

export default function ServerStatus() {
  const [status, setStatus] = useState<Status>("checking");

  const check = async () => {
    setStatus("checking");
    const ok = await checkHealth();
    setStatus(ok ? "ok" : "error");
  };

  useEffect(() => {
    check();
    const id = setInterval(check, 15000);
    return () => clearInterval(id);
  }, []);

  const dot =
    status === "ok"
      ? "bg-green live-dot shadow-[0_0_6px_#1D9E75]"
      : status === "error"
      ? "bg-danger"
      : "bg-amber";

  const label =
    status === "ok" ? "Servidor activo" : status === "error" ? "Sin conexión" : "Verificando...";

  return (
    <button
      onClick={check}
      className="flex items-center gap-2 bg-surface border border-border rounded-lg px-3 py-1.5 text-[11px] font-mono text-muted hover:border-green/40 transition-colors"
    >
      <span className={`w-2 h-2 rounded-full flex-shrink-0 ${dot}`} />
      {label}
    </button>
  );
}
