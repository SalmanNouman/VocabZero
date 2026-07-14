import React, { useEffect, useRef, useState } from "react";
import { AlertCircle, Info, X } from "lucide-react";

interface ToastProps {
  message: string | null;
  variant: "error" | "info";
  onDismiss: () => void;
}

export const Toast: React.FC<ToastProps> = ({
  message,
  variant,
  onDismiss,
}) => {
  const [visible, setVisible] = useState(false);
  const hideTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dismissTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearTimers = () => {
    if (hideTimerRef.current) {
      clearTimeout(hideTimerRef.current);
      hideTimerRef.current = null;
    }
    if (dismissTimerRef.current) {
      clearTimeout(dismissTimerRef.current);
      dismissTimerRef.current = null;
    }
  };

  useEffect(() => {
    if (message) {
      clearTimers();
      setVisible(true);
      hideTimerRef.current = setTimeout(() => {
        setVisible(false);
        dismissTimerRef.current = setTimeout(onDismiss, 200);
      }, 4000);
    }
    return clearTimers;
  }, [message, onDismiss]);

  if (!message) return null;

  const isError = variant === "error";

  return (
    <div
      className={`fixed bottom-6 right-6 z-[60] flex items-center gap-3 px-4 py-3 rounded-lg border shadow-[var(--shadow-lg)] max-w-sm transition-transform duration-200 ${
        visible ? "translate-x-0" : "translate-x-[120%]"
      } ${
        isError
          ? "bg-popover border-destructive text-foreground"
          : "bg-popover border-accent text-foreground"
      }`}
    >
      {isError ? (
        <AlertCircle className="h-5 w-5 text-destructive flex-shrink-0" />
      ) : (
        <Info className="h-5 w-5 text-accent flex-shrink-0" />
      )}
      <span className="text-sm font-medium">{message}</span>
      <button
        onClick={() => {
          clearTimers();
          setVisible(false);
          dismissTimerRef.current = setTimeout(onDismiss, 200);
        }}
        className="ml-auto p-1 rounded-md text-muted-foreground hover:text-foreground hover:bg-surface-raised transition-colors cursor-pointer flex-shrink-0"
        aria-label="Dismiss"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  );
};
