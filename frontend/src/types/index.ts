export interface LexiconEntry {
  source_term: string;
  target_term: string;
  confidence?: number;
  count?: number;
  audio_hash?: string;
  context?: string;
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
  channels: number;
  dtw_threshold_36: number;
  dtw_threshold_12: number;
  min_confidence_gate: number;
}
