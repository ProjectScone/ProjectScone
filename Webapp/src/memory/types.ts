// Wire shapes of the Python /v1 API the memory page reads. Optional fields
// are optional because the Rust server omits them (lanes, event_id,
// metadata, origin); the page shows what it receives and nothing else.

export interface Fact {
  fact_id: number;
  subject: string;
  predicate: string;
  object: string;
  confidence: number;
  valid_from: string;
  valid_until: string | null;
  status: "active" | "closed" | "proposed" | "declined";
  closed_reason?: string | null;
  source_episode_id?: number | null;
  origin?: "stated" | "extracted" | "inferred";
  excluded_reason?: string | null;
  superseded_by?: number | null;
  /** Exact substring of the source episode the claim rests on, when it came from one. */
  quote?: string | null;
  /** true: quote present and checked; false: source but no saved quote; null: stated, no source. Not a truth verdict. */
  grounded?: boolean | null;
}

export interface RecallItem {
  chunk_id: number;
  episode_id: number;
  text: string;
  score: number;
  similarity?: number | null;
  lanes?: Record<string, number>;
  created_at: string;
  source?: string | null;
  tags?: string[];
  metadata?: Record<string, string>;
}

export interface RecallResponse {
  event_id?: number | null;
  items: RecallItem[];
  facts: Fact[];
  degraded: string[];
  returned_bytes: number;
  space_bytes: number;
  context_reduction?: number;
}

export interface Episode {
  episode_id: number;
  kind: string;
  content: string;
  source: string | null;
  tags: string[];
  metadata: Record<string, string>;
  created_at: string;
}

export interface Status {
  space: string;
  episodes: number;
  chunks: number;
  bytes?: number;
  pending_review?: number;
  revision: number;
  embedder?: string;
  document_store?: string;
  vector_index?: string;
  semantic_lane: string;
  pending_distill: number;
  last_distill?: Record<string, unknown> | null;
}

export interface Scopes {
  scopes: Record<string, Record<string, number>>;
}

export interface Metric {
  name: string;
  value: number | Record<string, number | null> | null;
  n: number;
  denominator: string;
  unit: string;
  definition: string;
  caveat?: string | null;
}

export interface MetricsReport {
  evidence: string;
  coverage?: {
    events_considered: number;
    earliest_retained: string | null;
    latest: string | null;
    truncated: boolean;
    schema_versions: number[];
  } | null;
  failures?: Record<string, number>;
  embedders?: string[];
  metrics?: Metric[];
  daily?: Record<string, Record<string, number>>;
}

export interface EventRecord {
  event_id: number;
  ts: string;
  kind: string;
  schema_version: number;
  payload: Record<string, unknown>;
}

export interface EventsResponse {
  events: EventRecord[];
  evidence: string;
  queries_recorded?: "text" | "hash";
  truncated?: boolean;
  next_after_id?: number;
}

/** What the shell hands the page. Matches the root-owned client. */
export type { ApiClient } from '../api';

export type View = "search" | "documents" | "profile" | "beliefs" | "review" | "live" | "analytics" | "scopes" | "status";

export interface SearchState {
  q: string;
  where: Record<string, string>;
  tags: string[];
  asOf: string;
}
