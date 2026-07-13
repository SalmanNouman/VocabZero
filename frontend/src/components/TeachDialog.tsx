import React, { useState, useEffect, useRef } from "react";
import { HelpCircle, Check, X } from "lucide-react";

interface TeachDialogProps {
  isOpen: boolean;
  onClose: () => void;
  signature: string | null;
  initialValue: string;
  suggestions: string[];
  onSave: (translation: string) => Promise<void>;
}

export const TeachDialog: React.FC<TeachDialogProps> = ({
  isOpen,
  onClose,
  signature,
  initialValue,
  suggestions,
  onSave,
}) => {
  const [translation, setTranslation] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      setTranslation(initialValue);
      setTimeout(() => {
        inputRef.current?.focus();
      }, 50);
    }
  }, [isOpen, initialValue]);

  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!translation.trim() || !signature) return;
    setIsSubmitting(true);
    try {
      await onSave(translation.trim());
      onClose();
    } catch (err) {
      console.error(err);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <div
        className="bg-popover border border-border rounded-xl max-w-md w-full shadow-[var(--shadow-lg)] p-6 relative"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute right-4 top-4 p-1 rounded-md text-muted-foreground hover:bg-surface-raised hover:text-foreground transition-all cursor-pointer"
        >
          <X className="h-4.5 w-4.5" />
        </button>

        <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
          <HelpCircle className="h-5 w-5 text-accent-foreground" />
          Teach New Sound
        </h3>
        <p className="text-xs text-muted-foreground mt-1">
          This acoustic sound is unrecognized. Provide the English translation
          to save it:
        </p>

        <form onSubmit={handleSubmit} className="mt-4 flex flex-col gap-4">
          <div className="font-mono text-xs text-accent-foreground bg-surface-sunken border border-border p-2.5 rounded-lg break-all">
            Signature:{" "}
            <span className="text-foreground font-bold">
              {signature || "None"}
            </span>
          </div>

          {/* Autocomplete suggestions */}
          {suggestions.length > 0 && (
            <div className="flex flex-col gap-2">
              <label className="text-xs font-semibold text-muted-foreground">
                Context Autocomplete Suggestions:
              </label>
              <div className="flex flex-wrap gap-2">
                {suggestions.map((suggestion, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => setTranslation(suggestion)}
                    className="text-xs font-mono border border-accent/30 hover:border-accent/70 bg-accent/10 hover:bg-accent/25 px-2.5 py-1 rounded text-accent-foreground transition-all cursor-pointer"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Translation Input */}
          <div className="flex flex-col gap-1.5">
            <input
              ref={inputRef}
              type="text"
              placeholder="Type English translation..."
              value={translation}
              onChange={(e) => setTranslation(e.target.value)}
              className="bg-input border border-border rounded-lg px-3.5 py-2.5 text-sm text-foreground placeholder-text-dim focus:outline-none focus:ring-1 focus:ring-ring w-full"
              disabled={isSubmitting}
            />
          </div>

          {/* Buttons */}
          <div className="flex justify-end gap-3 mt-2">
            <button
              type="button"
              onClick={onClose}
              className="py-2.5 px-4 rounded-lg border border-border-strong text-muted-foreground hover:bg-surface-raised hover:text-foreground text-sm font-semibold transition-all cursor-pointer"
              disabled={isSubmitting}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="bg-primary hover:bg-primary text-primary-foreground py-2.5 px-4 rounded-lg text-sm font-semibold transition-all flex items-center gap-1.5 cursor-pointer active:scale-[0.98]"
              disabled={isSubmitting || !translation.trim()}
            >
              <Check className="h-4 w-4" />
              <span>{isSubmitting ? "Saving..." : "Teach Sound"}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
