import { NonRealTimeVAD } from "@ricky0123/vad-web";

export const SAMPLE_RATE = 16000;
export const PROBE_FREQS = [
  200, 250, 300, 350, 400, 450, 500, 550, 600, 660, 730, 800, 880, 970,
  1060, 1170, 1280, 1400, 1540, 1700, 1870, 2050, 2250, 2480, 2730, 3000,
  3300, 3630, 4000,
];

export function goertzel(samples: Float32Array | number[], sampleRate: number, targetFreq: number): number {
  const n = samples.length;
  const k = (targetFreq / sampleRate) * n;
  const w = (2 * Math.PI * k) / n;
  const coef = 2 * Math.cos(w);
  let s0 = 0;
  let s1 = 0;
  let s2 = 0;
  for (let i = 0; i < n; i++) {
    s0 = samples[i] + coef * s1 - s2;
    s2 = s1;
    s1 = s0;
  }
  return Math.sqrt(s2 * s2 + s1 * s1 - coef * s1 * s2);
}

export function extractFrequenciesForSegment(
  samples: Float32Array | number[],
  peaksPerFrame: number
): { signature: string } {
  const numFrames = 3;
  const frameSamples = Math.floor(samples.length / numFrames);
  const allPeaks: number[] = [];
  
  for (let f = 0; f < numFrames; f++) {
    const start = f * frameSamples;
    const end = f === numFrames - 1 ? samples.length : (f + 1) * frameSamples;
    const frame = samples.slice(start, end);
    
    let sumSq = 0;
    for (let i = 0; i < frame.length; i++) {
      sumSq += frame[i] * frame[i];
    }
    const rms = Math.sqrt(sumSq / frame.length);
    if (rms < 0.0001) continue;
    
    const mags = PROBE_FREQS.map((freq) => ({
      freq,
      mag: goertzel(frame, SAMPLE_RATE, freq) / rms,
    })).sort((a, b) => b.mag - a.mag);
    
    mags.slice(0, peaksPerFrame).forEach((p) => allPeaks.push(p.freq));
  }
  
  return { signature: allPeaks.sort((a, b) => a - b).join("_") };
}

export function vadSegmentSamples(
  samples: Float32Array | number[],
  silenceThreshold: number,
  windowSize = 320,
  silenceLimitFrames = 12,
  minWordSamples = 2400
): Float32Array[] {
  const words: Float32Array[] = [];
  let inWord = false;
  let wordStart = 0;
  let silenceFrameCount = 0;

  const floatSamples = samples instanceof Float32Array ? samples : new Float32Array(samples);

  for (let i = 0; i <= floatSamples.length - windowSize; i += windowSize) {
    let sumSq = 0;
    for (let j = 0; j < windowSize; j++) {
      sumSq += floatSamples[i + j] * floatSamples[i + j];
    }
    const rms = Math.sqrt(sumSq / windowSize);

    if (rms >= silenceThreshold) {
      if (!inWord) {
        inWord = true;
        wordStart = i;
      }
      silenceFrameCount = 0;
    } else {
      if (inWord) {
        silenceFrameCount++;
        if (silenceFrameCount >= silenceLimitFrames) {
          const wordEnd = i - silenceFrameCount * windowSize + windowSize;
          if (wordEnd - wordStart >= minWordSamples) {
            words.push(floatSamples.slice(wordStart, wordEnd));
          }
          inWord = false;
          silenceFrameCount = 0;
        }
      }
    }
  }
  if (inWord && floatSamples.length - wordStart >= minWordSamples) {
    words.push(floatSamples.slice(wordStart));
  }
  return words;
}

const SILERO_MODEL_URL = "/vad-silero_vad_legacy.onnx";
const SILERO_WASM_PATH = "/";
const SILERO_MIN_SPEECH_MS = 150;
const SILERO_REDEMPTION_MS = 240;
let sileroVadPromise: Promise<NonRealTimeVAD> | null = null;
let sileroVadThreshold: number | null = null;

function getSileroVAD(positiveSpeechThreshold: number): Promise<NonRealTimeVAD> {
  if (!sileroVadPromise || sileroVadThreshold !== positiveSpeechThreshold) {
    const negativeSpeechThreshold = Math.max(0.05, positiveSpeechThreshold - 0.15);
    sileroVadThreshold = positiveSpeechThreshold;
    sileroVadPromise = NonRealTimeVAD.new({
      modelURL: SILERO_MODEL_URL,
      positiveSpeechThreshold,
      negativeSpeechThreshold,
      redemptionMs: SILERO_REDEMPTION_MS,
      preSpeechPadMs: 0,
      minSpeechMs: SILERO_MIN_SPEECH_MS,
      submitUserSpeechOnPause: true,
      ortConfig: (ort) => {
        ort.env.wasm.wasmPaths = SILERO_WASM_PATH;
      },
    });
  }
  return sileroVadPromise;
}

export async function vadSegmentSamplesSilero(
  samples: Float32Array | number[],
  positiveSpeechThreshold: number,
  minWordSamples = 2400,
  fallbackSilenceThreshold = 0.01,
  collapseSegments = false,
): Promise<Float32Array[]> {
  const floatSamples = samples instanceof Float32Array ? samples : new Float32Array(samples);

  try {
    const vad = await getSileroVAD(positiveSpeechThreshold);
    const segments: Float32Array[] = [];
    let firstSpeechStart: number | null = null;
    let lastSpeechEnd: number | null = null;
    for await (const speech of vad.run(floatSamples, SAMPLE_RATE)) {
      if (speech.audio.length >= minWordSamples) {
        segments.push(speech.audio);
        firstSpeechStart ??= speech.start;
        lastSpeechEnd = speech.end;
      }
    }
    if (collapseSegments && segments.length > 1 && firstSpeechStart !== null && lastSpeechEnd !== null) {
      const start = Math.max(0, Math.floor((firstSpeechStart * SAMPLE_RATE) / 1000));
      const end = Math.min(floatSamples.length, Math.ceil((lastSpeechEnd * SAMPLE_RATE) / 1000));
      if (end - start >= minWordSamples) {
        return [floatSamples.slice(start, end)];
      }
    }
    return segments;
  } catch (error) {
    console.warn("Silero VAD unavailable; falling back to energy-gate segmentation.", error);
    sileroVadPromise = null;
    sileroVadThreshold = null;
    const fallbackSegments = vadSegmentSamples(
      floatSamples,
      fallbackSilenceThreshold,
      320,
      12,
      minWordSamples,
    );
    return collapseSegments ? mergeSpeechSegments(fallbackSegments) : fallbackSegments;
  }
}

export function mergeSpeechSegments(segments: Float32Array[]): Float32Array[] {
  if (segments.length <= 1) return segments;

  const totalLength = segments.reduce((total, segment) => total + segment.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const segment of segments) {
    merged.set(segment, offset);
    offset += segment.length;
  }
  return [merged];
}
