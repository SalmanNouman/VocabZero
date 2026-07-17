import { useState, useEffect, useRef, useCallback } from "react";
import { Header } from "./components/Header";
import { AcousticControlPanel } from "./components/AcousticControlPanel";
import { VisualizerCanvas } from "./components/VisualizerCanvas";
import {
  TranslationBoard,
  type StaticWordBubble,
} from "./components/TranslationBoard";
import { LexiconPanel } from "./components/LexiconPanel";
import { TeachDialog } from "./components/TeachDialog";
import { CalibrationWizard } from "./components/CalibrationWizard";
import { Toast } from "./components/Toast";
import { ConfirmDialog } from "./components/ConfirmDialog";
import type { LexiconEntry } from "./types";
import {
  SAMPLE_RATE,
  extractFrequenciesForSegment,
  vadSegmentSamplesSilero,
} from "./utils/audio";

export default function App() {
  // App mode & audio states
  const [isRecording, setIsRecording] = useState(false);
  const [peaksPerFrame, setPeaksPerFrame] = useState(2);
  const [gateThreshold, setGateThreshold] = useState(0.5);
  const [wordsDetected, setWordsDetected] = useState(0);
  const [dtwThreshold36, setDtwThreshold36] = useState<number | null>(null);
  const [dtwThreshold12, setDtwThreshold12] = useState<number | null>(null);

  // Lists & data tables
  const [lexiconData, setLexiconData] = useState<LexiconEntry[]>([]);
  const [isLexiconLoading, setIsLexiconLoading] = useState(false);
  const [staticBubbles, setStaticBubbles] = useState<StaticWordBubble[]>([]);

  // Dialog/Modal overlays
  const [isCalibrateOpen, setIsCalibrateOpen] = useState(false);
  const [isTeachOpen, setIsTeachOpen] = useState(false);
  const [isLexiconOpen, setIsLexiconOpen] = useState(false);

  // Toast + Confirm dialog state
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [confirmState, setConfirmState] = useState<{
    sourceTerm: string;
  } | null>(null);

  const showToast = useCallback((msg: string) => {
    setToastMessage(msg);
  }, []);

  // Active teaching selection states
  const [activeTeachSignature, setActiveTeachSignature] = useState<
    string | null
  >(null);
  const [activeTeachSamples, setActiveTeachSamples] =
    useState<Float32Array | null>(null);
  const [activeTeachIndex, setActiveTeachIndex] = useState<number | null>(null);
  const [dialogInitialValue, setDialogInitialValue] = useState("");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const autocompleteReqIdRef = useRef(0);

  // Web Audio Context & Node references
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);

  // State for visualizer to trigger re-renders
  const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);

  // Buffers for accumulating static audio
  const staticAudioBuffersRef = useRef<Float32Array[]>([]);

  // Fetch initial config & lexicon
  const fetchLexicon = async () => {
    setIsLexiconLoading(true);
    try {
      const resp = await fetch("/api/lexicon");
      const result = await resp.json();
      if (result.ok && result.data) {
        setLexiconData(result.data || []);
      }
    } catch (err) {
      console.error("Failed to fetch lexicon:", err);
    } finally {
      setIsLexiconLoading(false);
    }
  };

  const fetchAudioConfig = async () => {
    try {
      const resp = await fetch("/api/audio_config");
      const result = await resp.json();
      if (result.ok && result.data) {
        setDtwThreshold36(
          result.data.dtw_threshold_36 || result.data.dtw_threshold,
        );
        setDtwThreshold12(result.data.dtw_threshold_12);
        setGateThreshold(result.data.min_confidence_gate || 0.5);
      }
    } catch (err) {
      console.error("Failed to fetch audio config:", err);
    }
  };

  useEffect(() => {
    fetchLexicon();
    fetchAudioConfig();
  }, []);

  // Clean up all audio nodes on component unmount
  useEffect(() => {
    return () => {
      stopAudioCapture();
    };
  }, []);

  const stopAudioCapture = () => {
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
    setAnalyserNode(null);
    setIsRecording(false);
  };

  // Reconstructed translation string
  const reconstructedText = staticBubbles
    .map((b) => b.translatedText || "[unknown]")
    .join(" ");

  // Toggle record
  const handleRecordToggle = async () => {
    if (isRecording) {
      await handleStopStaticRecord();
    } else {
      await handleStartStaticRecord();
    }
  };

  // --- STATIC MODE RECORDING ---
  const handleStartStaticRecord = async () => {
    staticAudioBuffersRef.current = [];
    setIsRecording(true);
    setStaticBubbles([]);

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

      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      setAnalyserNode(analyser);
      source.connect(analyser);

      const processor = ctx.createScriptProcessor(4096, 1, 1);
      scriptProcessorRef.current = processor;
      source.connect(processor);
      processor.connect(ctx.destination);

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        const chunk = new Float32Array(inputData.length);
        chunk.set(inputData);
        staticAudioBuffersRef.current.push(chunk);
      };
    } catch (err) {
      console.error("Microphone access failed:", err);
      stopAudioCapture();
    }
  };

  const handleStopStaticRecord = async () => {
    stopAudioCapture();

    // Compile samples
    let totalLen = 0;
    for (const b of staticAudioBuffersRef.current) {
      totalLen += b.length;
    }

    if (totalLen < 2400) {
      showToast("Recording too short, please try again.");
      return;
    }

    const samples = new Float32Array(totalLen);
    let offset = 0;
    for (const b of staticAudioBuffersRef.current) {
      samples.set(b, offset);
      offset += b.length;
    }

    // Segment audio using local VAD
    const segments = await vadSegmentSamplesSilero(samples, gateThreshold);
    if (segments.length === 0) {
      showToast(
        "No word segments detected. Speak louder or reduce Speech Probability.",
      );
      return;
    }

    // Map segments to static bubbles state
    const newBubbles: StaticWordBubble[] = segments.map((seg, idx) => {
      const { signature } = extractFrequenciesForSegment(seg, peaksPerFrame);
      return {
        id: `static_${idx}_${Date.now()}`,
        index: idx,
        signature: signature,
        translatedText: null,
        confidence: null,
        wordSamples: seg,
      };
    });

    setStaticBubbles(newBubbles);
    setWordsDetected(newBubbles.length);

    // Fire off translation requests for each segment asynchronously
    newBubbles.forEach((bubble) => {
      queryTranslationStatic(bubble);
    });
  };

  const queryTranslationStatic = async (bubble: StaticWordBubble) => {
    try {
      const response = await fetch("/api/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_term: bubble.signature,
          audio_data: Array.from(bubble.wordSamples),
        }),
      });
      const result = await response.json();
      if (result.ok && result.data && result.data.status === "translated") {
        setStaticBubbles((prev) =>
          prev.map((b) =>
            b.id === bubble.id
              ? {
                  ...b,
                  translatedText: result.data.translated_text,
                  confidence: result.data.confidence ?? 1.0,
                  signature: result.data.source_term || b.signature,
                }
              : b,
          ),
        );
      }
    } catch (err) {
      console.error("Translation lookup failed for static bubble:", err);
    }
  };

  // --- TEACHING WORKFLOW ---
  const handleTeachOpenStatic = async (
    signature: string,
    index: number,
    samples: Float32Array,
    currentValue: string,
  ) => {
    setActiveTeachSignature(signature);
    setActiveTeachSamples(samples);
    setActiveTeachIndex(index);
    setDialogInitialValue(currentValue);

    // Compute autocomplete context
    const wordsBefore = staticBubbles
      .filter((b) => b.index < index && b.translatedText)
      .map((b) => b.translatedText);
    const wordsAfter = staticBubbles
      .filter((b) => b.index > index && b.translatedText)
      .map((b) => b.translatedText);

    const maskedSentence =
      `${wordsBefore.slice(-4).join(" ")} [unknown] ${wordsAfter.slice(0, 4).join(" ")}`.trim();
    fetchAutocomplete(maskedSentence);
    setIsTeachOpen(true);
  };

  const fetchAutocomplete = async (maskedSentence: string) => {
    const reqId = ++autocompleteReqIdRef.current;
    try {
      const response = await fetch("/api/autocomplete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentence: maskedSentence }),
      });
      const result = await response.json();
      if (reqId !== autocompleteReqIdRef.current) return;
      if (result.ok && result.data?.suggestions) {
        setSuggestions(result.data.suggestions);
      } else {
        setSuggestions([]);
      }
    } catch (err) {
      if (reqId !== autocompleteReqIdRef.current) return;
      console.error("Failed to fetch autocomplete suggestions:", err);
      setSuggestions([]);
    }
  };

  const handleSaveTeachSound = async (translation: string) => {
    if (
      !activeTeachSignature ||
      !activeTeachSamples ||
      activeTeachIndex === null
    )
      return;

    try {
      const response = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_term: activeTeachSignature,
          target_term: translation,
          audio_data: Array.from(activeTeachSamples),
        }),
      });
      const result = await response.json();

      if (!response.ok || !result.ok) {
        const errMsg = result.error?.message || "Failed to save teach sound";
        showToast(errMsg);
        throw new Error(errMsg);
      }

      const canonicalSignature =
        result.data?.source_term || activeTeachSignature;

      setStaticBubbles((prev) =>
        prev.map((b) =>
          b.index === activeTeachIndex
            ? {
                ...b,
                translatedText: translation,
                confidence: 1.0,
                signature: canonicalSignature,
              }
            : b,
        ),
      );

      fetchLexicon();
    } catch (err) {
      showToast(
        err instanceof Error
          ? err.message
          : "Failed to submit teach sound feedback",
      );
      throw err;
    }
  };

  // --- DELETE LEXICON ENTRY ---
  const handleDeleteLexicon = async (sourceTerm: string) => {
    setConfirmState({ sourceTerm });
  };

  const handleConfirmDelete = async () => {
    if (!confirmState) return;
    const sourceTerm = confirmState.sourceTerm;
    setConfirmState(null);

    try {
      const response = await fetch(
        `/api/lexicon/${encodeURIComponent(sourceTerm)}`,
        {
          method: "DELETE",
        },
      );
      const result = await response.json();
      if (result.ok) {
        setLexiconData((prev) =>
          prev.filter((item) => item.source_term !== sourceTerm),
        );

        setStaticBubbles((prev) =>
          prev.map((b) =>
            b.signature === sourceTerm
              ? { ...b, translatedText: null, confidence: null }
              : b,
          ),
        );
      }
    } catch (err) {
      console.error("Failed to delete lexicon entry:", err);
    }
  };

  const handleCalibrationComplete = (newThreshold: number) => {
    setDtwThreshold36(newThreshold);
  };

  return (
    <div className="flex flex-col min-h-[100dvh] bg-background text-foreground selection:bg-accent/30 selection:text-accent-foreground">
      <Header onViewLexicon={() => setIsLexiconOpen(true)} />

      <main className="flex-1 p-6 max-w-[1440px] w-full mx-auto flex flex-col gap-6 lg:min-h-[calc(100dvh-73px)]">
        {/* Upper Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-stretch lg:flex-1">
          {/* Controls column */}
          <div className="lg:col-span-1">
            <AcousticControlPanel
              isRecording={isRecording}
              onRecordToggle={handleRecordToggle}
              peaksPerFrame={peaksPerFrame}
              onPeaksChange={setPeaksPerFrame}
              gateThreshold={gateThreshold}
              onGateThresholdChange={setGateThreshold}
              wordsDetected={wordsDetected}
              dtwThreshold36={dtwThreshold36}
              dtwThreshold12={dtwThreshold12}
              onCalibrateOpen={() => setIsCalibrateOpen(true)}
              visualizerComponent={
                <VisualizerCanvas
                  analyserNode={analyserNode}
                  isRecording={isRecording}
                />
              }
            />
          </div>

          {/* Translation board column */}
          <div className="lg:col-span-2 flex flex-col lg:h-full">
            <TranslationBoard
              reconstructedText={reconstructedText}
              staticBubbles={staticBubbles}
              onTeachOpenStatic={handleTeachOpenStatic}
            />
          </div>
        </div>
      </main>

      {/* Lexicon Dictionary Slide-over Panel */}
      <LexiconPanel
        isOpen={isLexiconOpen}
        onClose={() => setIsLexiconOpen(false)}
        lexiconData={lexiconData}
        onDelete={handleDeleteLexicon}
        isLoading={isLexiconLoading}
      />

      {/* Mic Calibration Wizard Overlay */}
      <CalibrationWizard
        isOpen={isCalibrateOpen}
        onClose={() => setIsCalibrateOpen(false)}
        gateThreshold={gateThreshold}
        onCalibrationComplete={handleCalibrationComplete}
      />

      {/* Sound Teaching Dialog Overlay */}
      <TeachDialog
        isOpen={isTeachOpen}
        onClose={() => setIsTeachOpen(false)}
        signature={activeTeachSignature}
        initialValue={dialogInitialValue}
        suggestions={suggestions}
        onSave={handleSaveTeachSound}
      />

      {/* Toast notifications */}
      <Toast
        message={toastMessage}
        variant="error"
        onDismiss={() => setToastMessage(null)}
      />

      {/* Confirm delete dialog */}
      <ConfirmDialog
        isOpen={confirmState !== null}
        title="Delete Lexicon Mapping"
        message={`Are you sure you want to delete the mapping for sound "${confirmState?.sourceTerm ?? ""}"?`}
        confirmLabel="Delete"
        onConfirm={handleConfirmDelete}
        onCancel={() => setConfirmState(null)}
      />
    </div>
  );
}
