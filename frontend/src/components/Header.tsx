import React from "react";
import { Database } from "lucide-react";

interface HeaderProps {
  onViewLexicon: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onViewLexicon }) => {
  return (
    <header className="flex items-center justify-between border-b border-border bg-card/60 px-6 py-4 backdrop-blur-lg z-10">
      <div className="flex items-center gap-3">
        <svg
          className="h-6 w-6 text-accent"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M2 12h2M6 8v8M10 4v16M14 6v12M18 9v6M22 12h-2" />
        </svg>
        <h1 className="font-extrabold text-2xl tracking-wide text-foreground">
          VocabZero
        </h1>
        <span className="font-mono text-xs border border-border-strong rounded px-2 py-0.5 text-muted-foreground bg-surface-raised">
          Lexicon-Learning Translator
        </span>
      </div>
      <button
        onClick={onViewLexicon}
        className="flex items-center gap-2 px-3.5 py-2 rounded-lg border border-border-strong bg-surface-raised hover:bg-surface-raised text-sm font-medium text-muted-foreground hover:text-foreground transition-all cursor-pointer"
      >
        <Database className="h-4 w-4" />
        <span>View Lexicon</span>
      </button>
    </header>
  );
};
