import React from "react";
import { Mic, Activity, Sliders, Volume2, Database } from "lucide-react";

interface AcousticControlPanelProps {
  isRecording: boolean;
  onRecordToggle: () => void;
  peaksPerFrame: number;
  onPeaksChange: (peaks: number) => void;
  gateThreshold: number;
  onGateThresholdChange: (val: number) => void;
  wordsDetected: number;
  dtwThreshold36: number | null;
  dtwThreshold12: number | null;
  onCalibrateOpen: () => void;
  visualizerComponent: React.ReactNode;
}

export const AcousticControlPanel: React.FC<AcousticControlPanelProps> = ({
  isRecording,
  onRecordToggle,
  peaksPerFrame,
  onPeaksChange,
  gateThreshold,
  onGateThresholdChange,
  wordsDetected,
  dtwThreshold36,
  dtwThreshold12,
  onCalibrateOpen,
  visualizerComponent,
}) => {
  return (
    <div className="bg-card/80 backdrop-blur-md border border-border rounded-xl p-5 shadow-[var(--shadow-md)] flex flex-col gap-5">
      <div>
        <h3 className="text-lg font-semibold tracking-tight text-foreground flex items-center gap-2">
          <Activity className="h-5 w-5 text-accent-foreground" />
          Control Panel
        </h3>
        <p className="text-xs text-muted-foreground mt-1">
          Configure audio threshold and speak phrases
        </p>
      </div>

      {/* Waveform Visualizer */}
      <div className="flex flex-col gap-2">
        <label className="text-xs font-semibold text-muted-foreground flex items-center gap-1.5">
          <Volume2 className="h-3.5 w-3.5" />
          Audio Input Waveform
        </label>
        {visualizerComponent}
      </div>

      {/* Record Action Button */}
      <button
        onClick={onRecordToggle}
        className={`w-full py-3 px-4 rounded-lg font-semibold text-sm flex items-center justify-center gap-2 transition-all cursor-pointer active:scale-[0.98] ${
          isRecording
            ? "bg-destructive hover:bg-destructive text-destructive-foreground pulse-ring-active"
            : "bg-primary hover:bg-primary text-primary-foreground shadow-[var(--shadow-md)]"
        }`}
      >
        <Mic className={`h-4 w-4 ${isRecording ? "animate-pulse" : ""}`} />
        <span>{isRecording ? "Stop Recording" : "Record Phrase"}</span>
      </button>

      {/* Settings Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-start">
        {/* Peaks Dropdown */}
        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-medium text-muted-foreground flex items-center gap-1">
            <Sliders className="h-3 w-3" /> Peaks / Frame
          </label>
          <select
            value={peaksPerFrame}
            onChange={(e) => onPeaksChange(Number(e.target.value))}
            className="bg-input border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
          >
            <option value={1}>1 Peak</option>
            <option value={2}>2 Peaks (Standard)</option>
            <option value={3}>3 Peaks</option>
          </select>
        </div>

        {/* Gate Slider */}
        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-medium text-muted-foreground flex items-center justify-between">
            <span className="flex items-center gap-1">
              <Volume2 className="h-3 w-3" /> Gate Threshold
            </span>
            <span className="font-mono text-accent-foreground font-bold">
              {gateThreshold.toFixed(3)}
            </span>
          </label>
          <div className="flex items-center min-h-[34px]">
            <input
              type="range"
              min={0.002}
              max={0.05}
              step={0.001}
              value={gateThreshold}
              onChange={(e) => onGateThresholdChange(Number(e.target.value))}
              className="w-full h-1.5 bg-secondary rounded-lg appearance-none cursor-pointer accent-accent"
            />
          </div>
        </div>
      </div>

      {/* Mic Calibrate Trigger */}
      <button
        onClick={onCalibrateOpen}
        className="w-full border border-border-strong hover:bg-surface-raised bg-transparent text-muted-foreground py-2.5 px-4 rounded-lg font-medium text-sm transition-all flex items-center justify-center gap-2 cursor-pointer"
      >
        <Mic className="h-4 w-4" /> Calibrate Microphone
      </button>

      {/* Acoustic Info Card */}
      <div className="bg-surface-sunken border border-border rounded-lg p-3.5 flex flex-col gap-2">
        <label className="text-xs font-bold text-muted-foreground flex items-center gap-1.5">
          <Database className="h-3.5 w-3.5 text-muted-foreground" />
          Acoustic Translator Stats
        </label>
        <div className="grid grid-cols-2 gap-y-2 gap-x-4 text-xs font-mono text-text-dim">
          <div className="flex justify-between border-b border-border pb-1">
            <span>Sample Rate:</span>
            <span className="text-foreground">16.0 kHz</span>
          </div>
          <div className="flex justify-between border-b border-border pb-1">
            <span>Detected:</span>
            <span className="text-foreground font-bold">{wordsDetected}</span>
          </div>
          <div className="flex justify-between pb-1">
            <span>DTW Thresh:</span>
            <span className="text-foreground font-bold">
              {(dtwThreshold36 ?? dtwThreshold12) !== null
                ? (dtwThreshold36 ?? dtwThreshold12)!.toFixed(2)
                : "--"}
            </span>
          </div>
          <div className="flex justify-between pb-1">
            <span>Bands:</span>
            <span className="text-foreground">29 Mel</span>
          </div>
        </div>
      </div>
    </div>
  );
};
