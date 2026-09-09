export type NodeKind = 'session' | 'turn' | 'tool_call' | 'episode' | 'chunk' | 'claim' | 'recall' | 'feedback';
export interface EvidenceNode {
  id: string;
  kind: NodeKind;
  label: string;
  ts?: string;
  data?: Record<string, unknown>;
}
export interface EvidenceEdge { source: string; target: string; kind: string; label?: string }
export interface EvidenceGraph {
  nodes: EvidenceNode[];
  edges: EvidenceEdge[];
  truncated?: boolean;
  provenance_omitted?: number;
  facts_truncated?: boolean;
  fact_read_status?: 'not_read' | 'bounded' | 'unavailable' | 'failed';
}
export interface Status { space?: string; name?: string; episodes?: number; chunks?: number; [key: string]: unknown }
export interface RecallItem { episode_id: number | string; text: string; score?: number }
export interface RecallResult { items: RecallItem[]; facts?: unknown[]; event_id?: number; evidence_graph?: unknown }
export const EMPTY_GRAPH: EvidenceGraph = { nodes: [], edges: [] };
