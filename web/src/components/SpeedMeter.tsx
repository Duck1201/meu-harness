import { Zap } from "lucide-react";

import type { GenerationSpeed } from "../types";

export function tokensPerSecond(tokens: number, ms: number): number | null {
  return ms > 0 ? (tokens * 1000) / ms : null;
}

function format(value: number | null): string {
  return value === null ? "—" : value >= 100 ? value.toFixed(0) : value.toFixed(1);
}

/**
 * Tokens por segundo do passo que acabou de gerar, a leitura do prompt e a média
 * do turno. O harness não faz streaming token a token: o número chega quando o
 * passo termina, medido pelo runtime (Ollama ou llama.cpp), não estimado aqui.
 */
export function SpeedMeter({ speed }: { speed: GenerationSpeed }) {
  const last = tokensPerSecond(speed.lastOutputTokens, speed.lastEvalMs);
  const prefill =
    speed.lastPromptEvalMs !== undefined
      ? tokensPerSecond(speed.lastPromptTokens, speed.lastPromptEvalMs)
      : null;
  const turn = tokensPerSecond(speed.turnOutputTokens, speed.turnEvalMs);
  return (
    <div className="speed-meter" role="status" aria-label="Velocidade de geração">
      <Zap size={13} />
      <strong>{format(last)} tok/s</strong>
      <span>
        {speed.lastOutputTokens} tokens em {(speed.lastEvalMs / 1000).toFixed(1)} s
      </span>
      {prefill !== null && <span>prompt {format(prefill)} tok/s</span>}
      {speed.steps > 1 && <span>média do turno {format(turn)} tok/s</span>}
    </div>
  );
}
