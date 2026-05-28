export interface BinInfo {
  label: string;
  color: string;
  emoji: string;
  id: string;
}

export const BIN_INFO: Record<string, BinInfo> = {
  "Residuos aprovechables":           { label: "Aprovechables",           color: "#185FA5", emoji: "♻️", id: "aprovechable" },
  "Residuos organicos aprovechables": { label: "Orgánicos aprovechables", color: "#1D9E75", emoji: "🍃", id: "organico" },
  "Residuos no aprovechables":        { label: "No aprovechables",        color: "#5F5E5A", emoji: "🗑️", id: "no-aprovechable" },
  "Residuos peligrosos":              { label: "Peligrosos",              color: "#A32D2D", emoji: "⚠️", id: "peligroso" },
};

export const ALL_BINS = Object.keys(BIN_INFO);

export function confLevel(c: number): "high" | "medium" | "low" {
  return c >= 80 ? "high" : c >= 50 ? "medium" : "low";
}

export function confColor(c: number): string {
  return c >= 80 ? "#1D9E75" : c >= 50 ? "#BA7517" : "#A32D2D";
}