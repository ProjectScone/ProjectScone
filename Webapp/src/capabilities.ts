export const FEATURE_KEYS = ['recall', 'facts.read', 'facts.review', 'facts.close',
  'facts.exclude', 'facts.include', 'events.read', 'metrics.read', 'scopes.read', 'status.read'] as const;
export type Feature = typeof FEATURE_KEYS[number];
type OptionalFeature = 'episodes.attachments';
export interface Capabilities {
  schema_version: 1;
  implementation: string;
  features: Record<Feature | OptionalFeature, boolean>;
}

export function parseCapabilities(value: unknown): Capabilities {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw Error('Invalid capability response');
  const data = value as Record<string, unknown>;
  if (data.schema_version !== 1 || typeof data.implementation !== 'string' || !data.implementation.trim()
    || !data.features || typeof data.features !== 'object' || Array.isArray(data.features)) throw Error('Unsupported capability response');
  const features = data.features as Record<string, unknown>;
  if (FEATURE_KEYS.some(key => typeof features[key] !== 'boolean')) throw Error('Incomplete or invalid capability flags');
  if (features['episodes.attachments'] !== undefined && typeof features['episodes.attachments'] !== 'boolean') throw Error('Invalid upload capability flag');
  return {schema_version:1, implementation:data.implementation, features:{...Object.fromEntries(FEATURE_KEYS.map(key => [key, features[key]])), 'episodes.attachments':features['episodes.attachments'] === true} as Capabilities['features']};
}
