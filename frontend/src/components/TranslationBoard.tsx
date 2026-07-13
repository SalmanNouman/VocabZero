import React from "react";
import { ChevronRight, Languages, Mic } from "lucide-react";

export interface StaticWordBubble {
  id: string;
  index: number;
  signature: string;
  translatedText: string | null;
  confidence: number | null;
  wordSamples: Float32Array;
}

interface TranslationBoardProps {
  reconstructedText: string;
  staticBubbles: StaticWordBubble[];
  onTeachOpenStatic: (
    signature: string,
    index: number,
    samples: Float32Array,
    currentValue: string,
  ) => void;
}

export const TranslationBoard: React.FC<TranslationBoardProps> = ({
  reconstructedText,
  staticBubbles,
  onTeachOpenStatic,
}) => {
  return (
    <div className="bg-card/80 backdrop-blur-md border border-border rounded-xl p-5 shadow-[var(--shadow-md)] flex flex-col gap-4 flex-1">
      <div className="flex items-center justify-between border-b border-border pb-3">
        <h3 className="text-lg font-semibold tracking-tight text-foreground flex items-center gap-2">
          <Languages className="h-5 w-5 text-accent-foreground" /> Translation
          Board
        </h3>
        <span className="text-[10px] uppercase tracking-wider font-bold text-text-dim bg-surface-sunken px-2 py-0.5 rounded border border-border">
          Phrase Translation
        </span>
      </div>

      {/* Reconstructed Translation Text */}
      <div className="bg-surface-sunken border border-border rounded-xl p-4 flex flex-col gap-1 shadow-[var(--shadow-sm)]">
        <span className="text-xs font-semibold text-text-dim flex items-center gap-1">
          <ChevronRight className="h-3.5 w-3.5" /> Reconstructed Translation:
        </span>
        <span className="text-lg font-bold text-foreground min-h-[28px] break-words">
          {reconstructedText || "..."}
        </span>
      </div>

      {/* Board Content */}
      <div className="flex-1 flex flex-col justify-center min-h-[160px]">
        {staticBubbles.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-3 py-8 text-center">
            <div className="h-12 w-12 rounded-full bg-surface-raised flex items-center justify-center">
              <Mic className="h-6 w-6 text-muted-foreground" />
            </div>
            <p className="text-sm text-muted-foreground">
              Click <strong className="text-foreground">Record Phrase</strong>,
              speak a sentence, then click{" "}
              <strong className="text-foreground">Stop Recording</strong>.
            </p>
          </div>
        ) : (
          <div className="flex flex-wrap gap-3 items-center justify-center p-3">
            {staticBubbles.map((bubble) => {
              const isKnown = bubble.translatedText !== null;
              return (
                <button
                  key={bubble.id}
                  onClick={() =>
                    onTeachOpenStatic(
                      bubble.signature,
                      bubble.index,
                      bubble.wordSamples,
                      bubble.translatedText || "",
                    )
                  }
                  className={`px-4 py-2.5 rounded-xl border font-medium text-sm flex flex-col items-center justify-center gap-0.5 cursor-pointer transition-all shadow-[var(--shadow-sm)] active:scale-[0.98] ${
                    isKnown
                      ? "bg-accent/12 border-accent/30 text-accent-foreground hover:bg-accent/20"
                      : "bg-destructive/12 border-destructive/30 text-destructive hover:bg-destructive/20"
                  }`}
                >
                  <span className="font-semibold">
                    {isKnown ? bubble.translatedText : `[${bubble.signature}]`}
                  </span>
                  <span className="text-[10px] text-muted-foreground font-mono">
                    {isKnown
                      ? `conf: ${bubble.confidence !== null ? bubble.confidence.toFixed(2) : "1.00"}`
                      : "Click to Teach"}
                  </span>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
