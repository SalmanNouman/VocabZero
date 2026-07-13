import React, { useState, useEffect, useRef } from "react";

import { X, Mic, RefreshCw, CheckCircle, ChevronRight } from "lucide-react";

import { SAMPLE_RATE, vadSegmentSamples } from "../utils/audio";

interface CalibrationWizardProps {
  isOpen: boolean;

  onClose: () => void;

  gateThreshold: number;

  onCalibrationComplete: (dtwThreshold: number) => void;
}

interface ComputeData {
  intra_class: { mean: number; min: number; max: number };

  inter_class: { mean: number; min: number; max: number };

  separation_ratio: number;

  well_separated: boolean;

  suggested_threshold: number;
}

export const CalibrationWizard: React.FC<CalibrationWizardProps> = ({
  isOpen,

  onClose,

  gateThreshold,

  onCalibrationComplete,
}) => {
  const [step, setStep] = useState(1);

  const [currentPhrase, setCurrentPhrase] = useState(0); // 0 to 2 (3 phrases)

  const [currentRep, setCurrentRep] = useState(0); // 0 to 2 (3 reps per phrase)

  const [isRecording, setIsRecording] = useState(false);

  const [statusMsg, setStatusMsg] = useState("Press Record to start.");

  // Recording references

  const audioContextRef = useRef<AudioContext | null>(null);

  const mediaStreamRef = useRef<MediaStream | null>(null);

  const audioSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);

  const recordedChunksRef = useRef<Float32Array[]>([]);

  // Step 2 variables

  const [isComputing, setIsComputing] = useState(false);

  const [computeError, setComputeError] = useState<string | null>(null);

  const [computeData, setComputeData] = useState<ComputeData | null>(null);

  const [suggestedThreshold, setSuggestedThreshold] = useState(1.8);

  const [sliderMin, setSliderMin] = useState(0.1);

  const [sliderMax, setSliderMax] = useState(5.0);

  // Step 3 variables

  const [appliedThreshold, setAppliedThreshold] = useState(0);

  const [appliedGate, setAppliedGate] = useState(0);

  const [isPersisted, setIsPersisted] = useState(true);

  const [applyError, setApplyError] = useState<string | null>(null);

  const NUM_PHRASES = 3;

  const REPS_PER_PHRASE = 3;

  const stopRecordingCleanup = () => {
    if (scriptProcessorRef.current) {
      scriptProcessorRef.current.disconnect();

      scriptProcessorRef.current = null;
    }

    if (audioSourceRef.current) {
      audioSourceRef.current.disconnect();

      audioSourceRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());

      mediaStreamRef.current = null;
    }

    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      audioContextRef.current.close();

      audioContextRef.current = null;
    }

    setIsRecording(false);
  };

  useEffect(() => {
    if (isOpen) {
      // Reset state on open

      setStep(1);

      setCurrentPhrase(0);

      setCurrentRep(0);

      setIsRecording(false);

      setStatusMsg("Record 3 words, 3 times each. Ready to start.");

      setComputeData(null);

      setComputeError(null);

      // Clear samples on backend

      fetch("/api/calibrate/samples", { method: "DELETE" }).catch(
        console.error,
      );
    }

    return () => {
      stopRecordingCleanup();
    };
  }, [isOpen]);

  if (!isOpen) return null;

  const handleRecordToggle = async () => {
    if (isRecording) {
      await handleStopRecording();
    } else {
      await handleStartRecording();
    }
  };

  const handleStartRecording = async () => {
    recordedChunksRef.current = [];

    setIsRecording(true);

    setStatusMsg("Listening... speak your word now.");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,

          noiseSuppression: false,

          autoGainControl: false,
        },
      });

      mediaStreamRef.current = stream;

      const AudioCtx =
        window.AudioContext || (window as any).webkitAudioContext;

      const ctx = new AudioCtx({ sampleRate: SAMPLE_RATE });

      audioContextRef.current = ctx;

      const source = ctx.createMediaStreamSource(stream);

      audioSourceRef.current = source;

      const processor = ctx.createScriptProcessor(4096, 1, 1);

      scriptProcessorRef.current = processor;

      source.connect(processor);

      processor.connect(ctx.destination);

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);

        const chunk = new Float32Array(inputData.length);

        chunk.set(inputData);

        recordedChunksRef.current.push(chunk);
      };
    } catch (err) {
      console.error(err);

      setStatusMsg("Failed to access microphone.");

      setIsRecording(false);
    }
  };

  const handleStopRecording = async () => {
    stopRecordingCleanup();

    setStatusMsg("Processing recording...");

    // Combine chunks

    let totalLen = 0;

    for (const chunk of recordedChunksRef.current) {
      totalLen += chunk.length;
    }

    if (totalLen < 2400) {
      setStatusMsg("Recording too short, please try again.");

      return;
    }

    const samples = new Float32Array(totalLen);

    let offset = 0;

    for (const chunk of recordedChunksRef.current) {
      samples.set(chunk, offset);

      offset += chunk.length;
    }

    // Segment using VAD

    const segments = vadSegmentSamples(samples, gateThreshold);

    if (segments.length === 0) {
      setStatusMsg(
        "No speech detected. Speak louder or reduce Gate Threshold.",
      );

      return;
    }

    setStatusMsg(`Sending segment to server...`);

    const label = `phrase_${currentPhrase + 1}`;

    try {
      let success = true;

      for (const segment of segments) {
        const resp = await fetch("/api/calibrate/sample", {
          method: "POST",

          headers: { "Content-Type": "application/json" },

          body: JSON.stringify({
            label: label,

            audio_data: Array.from(segment),
          }),
        });

        const res = await resp.json();

        if (!res.ok) {
          success = false;

          setStatusMsg(`Error: ${res.error?.message || "Failed to submit"}`);

          break;
        }
      }

      if (success) {
        const nextRep = currentRep + 1;

        if (nextRep >= REPS_PER_PHRASE) {
          setCurrentRep(0);

          setCurrentPhrase((prev) => prev + 1);
        } else {
          setCurrentRep(nextRep);
        }

        setStatusMsg("Success! Record next rep.");
      }
    } catch (err) {
      console.error(err);

      setStatusMsg("Network error. Try again.");
    }
  };

  const handleCompute = async () => {
    setStep(2);

    setIsComputing(true);

    setComputeError(null);

    try {
      const resp = await fetch("/api/calibrate/compute", { method: "POST" });

      const result = await resp.json();

      if (result.ok && result.data) {
        const d = result.data as ComputeData;

        setComputeData(d);

        setSuggestedThreshold(d.suggested_threshold);

        setSliderMin(Number(Math.max(0.1, d.intra_class.max * 0.8).toFixed(2)));

        setSliderMax(Number((d.inter_class.max * 1.2).toFixed(2)));
      } else {
        setComputeError(result.error?.message || "Computation failed.");
      }
    } catch (err) {
      console.error(err);

      setComputeError("Network error during computation.");
    } finally {
      setIsComputing(false);
    }
  };

  const handleApply = async () => {
    setApplyError(null);

    const payload: {
      dtw_threshold_36?: number;

      dtw_threshold_12?: number;

      min_confidence_gate?: number;

      persist: boolean;
    } = { persist: true };

    try {
      const cfgResp = await fetch("/api/audio_config");

      const cfgData = await cfgResp.json();

      if (cfgData.ok && cfgData.data) {
        if (cfgData.data.use_deltas) {
          payload.dtw_threshold_36 = suggestedThreshold;
        } else {
          payload.dtw_threshold_12 = suggestedThreshold;
        }
      }
    } catch (e) {
      payload.dtw_threshold_36 = suggestedThreshold;
    }

    try {
      const resp = await fetch("/api/calibrate/apply", {
        method: "POST",

        headers: { "Content-Type": "application/json" },

        body: JSON.stringify(payload),
      });

      const result = await resp.json();

      if (result.ok && result.data) {
        setAppliedThreshold(result.data.dtw_threshold);

        setAppliedGate(result.data.min_confidence_gate);

        setIsPersisted(result.data.persisted);

        onCalibrationComplete(result.data.dtw_threshold);

        setStep(3);
      } else {
        setApplyError(
          "Failed to apply: " + (result.error?.message || "Unknown error"),
        );
      }
    } catch (err) {
      console.error(err);

      setApplyError("Network error applying calibration.");
    }
  };

  const isStep1Done = currentPhrase >= NUM_PHRASES;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className="bg-popover border border-border rounded-xl max-w-xl w-full shadow-[var(--shadow-lg)] p-6 relative flex flex-col gap-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold text-foreground flex items-center gap-2">
            <Mic className="h-5 w-5 text-accent-foreground" /> Microphone
            Calibration Wizard
          </h2>

          <button
            onClick={onClose}
            className="p-1 rounded-md text-muted-foreground hover:bg-surface-raised hover:text-foreground transition-colors cursor-pointer"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Wizard Progress Dots */}

        <div className="flex items-center justify-center gap-4">
          {[1, 2, 3].map((s) => (
            <React.Fragment key={s}>
              <div
                className={`h-7 w-7 rounded-full flex items-center justify-center text-xs font-bold transition-all ${
                  step === s
                    ? "bg-accent text-foreground ring-4 ring-accent/20"
                    : step > s
                      ? "bg-success text-foreground"
                      : "bg-surface-raised text-muted-foreground border border-border-strong"
                }`}
              >
                {step > s ? "✓" : s}
              </div>

              {s < 3 && (
                <div
                  className={`h-0.5 w-12 rounded transition-colors ${
                    step > s ? "bg-success" : "bg-surface-raised"
                  }`}
                />
              )}
            </React.Fragment>
          ))}
        </div>

        {/* STEP 1: Record Samples */}

        {step === 1 && (
          <div className="flex flex-col gap-4">
            <div>
              <h3 className="text-sm font-semibold text-foreground">
                Step 1: Record Calibration Words
              </h3>

              <p className="text-xs text-muted-foreground mt-1">
                Record <strong>3 different words</strong>, each{" "}
                <strong>3 times</strong>. Press record, speak one word, and
                press stop. This measures acoustic variety.
              </p>
            </div>

            {/* Phrase status list */}

            <div className="grid grid-cols-3 gap-3">
              {[0, 1, 2].map((pIdx) => (
                <div
                  key={pIdx}
                  className={`p-3 rounded-lg border transition-all flex flex-col gap-2 ${
                    currentPhrase === pIdx
                      ? "bg-surface-raised border-accent/40 shadow-[var(--shadow-sm)]"
                      : pIdx < currentPhrase
                        ? "bg-card/30 border-success/20 opacity-70"
                        : "bg-surface-sunken/20 border-border"
                  }`}
                >
                  <span className="text-xs font-medium text-foreground">
                    Word {pIdx + 1}
                  </span>

                  <div className="flex gap-1.5">
                    {[0, 1, 2].map((rIdx) => {
                      const isFilled =
                        pIdx < currentPhrase ||
                        (pIdx === currentPhrase && rIdx < currentRep);

                      return (
                        <div
                          key={rIdx}
                          className={`h-2.5 w-2.5 rounded-full transition-all ${
                            isFilled ? "bg-accent" : "bg-surface-raised"
                          }`}
                        />
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>

            {/* Recording panel */}

            <div className="bg-surface-sunken/40 border border-border rounded-xl p-5 flex flex-col items-center justify-center gap-4 text-center">
              <div className="flex flex-col gap-1">
                <span className="text-sm font-bold text-foreground">
                  {isStep1Done
                    ? "Recording complete!"
                    : `Word ${currentPhrase + 1}`}
                </span>

                <span className="text-xs font-mono text-muted-foreground">
                  {isStep1Done
                    ? "9 / 9 samples recorded"
                    : `Rep ${currentRep + 1} of 3`}
                </span>
              </div>

              {!isStep1Done && (
                <button
                  onClick={handleRecordToggle}
                  className={`py-3 px-6 rounded-lg text-sm font-semibold flex items-center gap-2 cursor-pointer transition-all active:scale-[0.98] ${
                    isRecording
                      ? "bg-destructive hover:bg-destructive text-destructive-foreground pulse-ring-active"
                      : "bg-primary hover:bg-primary text-primary-foreground"
                  }`}
                >
                  <Mic className="h-4.5 w-4.5" />

                  <span>{isRecording ? "Stop" : "Record"}</span>
                </button>
              )}

              <p className="text-xs font-medium text-muted-foreground">
                {statusMsg}
              </p>
            </div>

            {/* Next button */}

            <button
              onClick={handleCompute}
              disabled={!isStep1Done}
              className={`w-full py-2.5 px-4 rounded-lg font-semibold text-sm transition-all flex items-center justify-center gap-1.5 cursor-pointer active:scale-[0.98] ${
                isStep1Done
                  ? "bg-accent hover:bg-accent-hover text-foreground shadow-[var(--shadow-md)]"
                  : "bg-surface-raised text-text-dim cursor-not-allowed border border-border-strong"
              }`}
            >
              <span>Compute Threshold</span>

              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        )}

        {/* STEP 2: Results & Slider */}

        {step === 2 && (
          <div className="flex flex-col gap-4">
            <div>
              <h3 className="text-sm font-semibold text-foreground">
                Step 2: Distance Analysis
              </h3>

              <p className="text-xs text-muted-foreground mt-1">
                Calculates similarity metrics using Dynamic Time Warping (DTW)
                distance matrices.
              </p>
            </div>

            {isComputing && (
              <div className="flex flex-col items-center justify-center p-12 gap-3">
                <RefreshCw className="h-8 w-8 text-accent-foreground animate-spin" />

                <p className="text-xs text-muted-foreground">
                  Computing distance matrix...
                </p>
              </div>
            )}

            {computeError && (
              <div className="p-4 bg-destructive/10 border border-destructive/30 rounded-lg text-center">
                <p className="text-sm text-destructive">{computeError}</p>

                <button
                  onClick={() => setStep(1)}
                  className="mt-2 text-xs text-accent-foreground hover:underline"
                >
                  ← Back to Recording
                </button>
              </div>
            )}

            {applyError && (
              <div className="p-4 bg-destructive/10 border border-destructive/30 rounded-lg text-center">
                <p className="text-sm text-destructive">{applyError}</p>
              </div>
            )}

            {computeData && !isComputing && (
              <div className="flex flex-col gap-4">
                {/* Result cards grid */}

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div className="bg-surface-sunken border border-border p-3.5 rounded-lg flex flex-col">
                    <span className="text-xs text-muted-foreground font-medium">
                      Same Phrase
                    </span>

                    <span className="text-lg font-bold text-foreground font-mono mt-1">
                      {computeData.intra_class.mean.toFixed(3)}
                    </span>

                    <span className="text-[10px] text-text-dim font-mono mt-0.5">
                      Range: {computeData.intra_class.min.toFixed(2)} –{" "}
                      {computeData.intra_class.max.toFixed(2)}
                    </span>
                  </div>

                  <div className="bg-surface-sunken border border-border p-3.5 rounded-lg flex flex-col">
                    <span className="text-xs text-muted-foreground font-medium">
                      Different Phrases
                    </span>

                    <span className="text-lg font-bold text-foreground font-mono mt-1">
                      {computeData.inter_class.mean.toFixed(3)}
                    </span>

                    <span className="text-[10px] text-text-dim font-mono mt-0.5">
                      Range: {computeData.inter_class.min.toFixed(2)} –{" "}
                      {computeData.inter_class.max.toFixed(2)}
                    </span>
                  </div>

                  <div className="bg-surface-sunken border border-border p-3.5 rounded-lg flex flex-col">
                    <span className="text-xs text-muted-foreground font-medium">
                      Separation Ratio
                    </span>

                    <span className="text-lg font-bold text-foreground font-mono mt-1">
                      {computeData.separation_ratio.toFixed(2)}x
                    </span>

                    <span
                      className={`text-[10px] font-bold mt-1 ${
                        computeData.well_separated
                          ? "text-success"
                          : "text-accent-foreground"
                      }`}
                    >
                      {computeData.well_separated
                        ? "✓ Well Separated"
                        : "⚠ High Overlap"}
                    </span>
                  </div>
                </div>

                {/* Slider adjustments */}

                <div className="flex flex-col gap-2 p-4 bg-surface-sunken/40 border border-border rounded-xl">
                  <div className="flex justify-between items-center text-xs font-semibold">
                    <span className="text-muted-foreground">
                      Suggested Threshold:
                    </span>

                    <span className="font-mono text-accent-foreground text-sm font-bold">
                      {suggestedThreshold.toFixed(3)}
                    </span>
                  </div>

                  <input
                    type="range"
                    min={sliderMin}
                    max={sliderMax}
                    step={0.01}
                    value={suggestedThreshold}
                    onChange={(e) =>
                      setSuggestedThreshold(Number(e.target.value))
                    }
                    className="w-full h-1.5 bg-surface-raised rounded-lg appearance-none cursor-pointer accent-accent"
                  />

                  <div className="flex justify-between text-[10px] text-text-dim">
                    <span>Stricter (fewer false positives)</span>

                    <span>Looser (fewer rejections)</span>
                  </div>
                </div>

                <div className="flex gap-3">
                  <button
                    onClick={() => setStep(1)}
                    className="flex-1 py-2.5 px-4 rounded-lg border border-border-strong text-muted-foreground hover:bg-surface-raised hover:text-foreground text-sm font-semibold transition-all cursor-pointer active:scale-[0.98]"
                  >
                    ← Back
                  </button>

                  <button
                    onClick={handleApply}
                    className="flex-1 bg-primary hover:bg-primary text-primary-foreground py-2.5 px-4 rounded-lg text-sm font-semibold transition-all cursor-pointer active:scale-[0.98]"
                  >
                    Apply Threshold ✓
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* STEP 3: Success Confirmation */}

        {step === 3 && (
          <div className="flex flex-col items-center justify-center p-6 gap-5 text-center">
            <CheckCircle className="h-14 w-14 text-success" />

            <div className="flex flex-col gap-1.5">
              <h3 className="text-lg font-bold text-foreground">
                Calibration Applied!
              </h3>

              <p className="text-xs text-muted-foreground">
                The computed VAD parameters have been applied successfully.
              </p>
            </div>

            {/* Parameters card */}

            <div className="bg-surface-sunken border border-border rounded-xl p-4 max-w-xs w-full flex flex-col gap-2 text-sm font-mono text-left">
              <div className="flex justify-between py-1 border-b border-border">
                <span className="text-text-dim">DTW Threshold:</span>

                <span className="text-accent-foreground font-bold">
                  {appliedThreshold.toFixed(4)}
                </span>
              </div>

              <div className="flex justify-between py-1">
                <span className="text-text-dim">Confidence Gate:</span>

                <span className="text-success font-bold">
                  {appliedGate.toFixed(2)}
                </span>
              </div>
            </div>

            <p className="text-xs font-semibold text-success">
              {isPersisted
                ? "Saved to calibration.json — will persist across restarts."
                : "Applied in-memory only."}
            </p>

            <button
              onClick={onClose}
              className="bg-accent hover:bg-accent-hover text-foreground font-semibold text-sm py-2.5 px-6 rounded-lg transition-all w-full max-w-xs cursor-pointer shadow-[var(--shadow-md)] active:scale-[0.98]"
            >
              Done
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
