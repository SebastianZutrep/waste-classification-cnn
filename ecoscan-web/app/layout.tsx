import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "EcoScan — Clasificador de Residuos",
  description: "Clasificación inteligente de residuos con CNN",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  );
}
