export interface LexiconEntry {
  source_term: string;
  target_term: string;
  confidence?: number;
  count?: number;
  audio_hash?: string;
  context_examples: string[];
}

export interface Detection {
  type: "detection";
  source_term: string;
  target_term: string;
  start_time: number;
  end_time: number;
  confidence: number;
}

export interface EnergyHistoryItem {
  time: number;
  rms: number;
  active: boolean;
}

export interface AudioConfig {
  sample_rate: number;
  dtw_threshold_36: number;
  dtw_threshold_12: number;
  dtw_threshold: number;
  min_confidence_gate: number;
  use_deltas: boolean;
  use_cmvn: boolean;
  use_vtln: boolean;
  use_liftering: boolean;
}
