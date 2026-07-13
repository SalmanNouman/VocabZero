import React, { useEffect } from "react";
import { X } from "lucide-react";
import { LexiconTable } from "./LexiconTable";
import type { LexiconEntry } from "../types";

interface LexiconPanelProps {
  isOpen: boolean;
  onClose: () => void;
  lexiconData: LexiconEntry[];
  onDelete: (sourceTerm: string) => void;
  isLoading: boolean;
}

export const LexiconPanel: React.FC<LexiconPanelProps> = ({
  isOpen,
  onClose,
  lexiconData,
  onDelete,
  isLoading,
}) => {
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex justify-end bg-black/50 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-popover border-l border-border w-full max-w-2xl h-full shadow-[var(--shadow-lg)] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-border px-5 py-4 shrink-0">
          <h2 className="text-lg font-bold text-foreground">Lexicon Dictionary</h2>
          <button
            onClick={onClose}
            className="p-1.5 rounded-md text-muted-foreground hover:bg-surface-raised hover:text-foreground transition-all cursor-pointer"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5">
          <LexiconTable
            lexiconData={lexiconData}
            onDelete={onDelete}
            isLoading={isLoading}
            embedded
          />
        </div>
      </div>
    </div>
  );
};
