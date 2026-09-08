export const FEATURE_KEYS = ['recall', 'facts.read', 'facts.review', 'facts.close',
  'facts.exclude', 'facts.include', 'events.read', 'metrics.read', 'scopes.read', 'status.read'] as const;
export type Feature = typeof FEATURE_KEYS[number];
type OptionalFeature = 'images.understand' | 'models.manage' | 'episodes.attachments' | 'episodes.list' | 'episodes.read' | 'facts.links' | 'integrity.read' | 'profile.read' | 'processing.distill' | 'processing.derive' | 'jobs.read' | 'recall.conditions';
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
  for(const key of ['processing.distill','processing.derive']){
    if(features[key]!==undefined&&typeof features[key]!=='boolean')throw Error('Invalid processing capability flag');
  }
  if (FEATURE_KEYS.some(key => typeof features[key] !== 'boolean')) throw Error('Incomplete or invalid capability flags');
  if (features['episodes.attachments'] !== undefined && typeof features['episodes.attachments'] !== 'boolean') throw Error('Invalid upload capability flag');
  if (features['episodes.list'] !== undefined && typeof features['episodes.list'] !== 'boolean') throw Error('Invalid inventory capability flag');
  if (features['episodes.read'] !== undefined && typeof features['episodes.read'] !== 'boolean') throw Error('Invalid source-read capability flag');
  if (features['facts.links'] !== undefined && typeof features['facts.links'] !== 'boolean') throw Error('Invalid relationship capability flag');
  if (features['integrity.read'] !== undefined && typeof features['integrity.read'] !== 'boolean') throw Error('Invalid integrity capability flag');
  if (features['profile.read'] !== undefined && typeof features['profile.read'] !== 'boolean') throw Error('Invalid profile capability flag');
  if(features['jobs.read']!==undefined&&typeof features['jobs.read']!=='boolean')throw Error('Invalid job history capability');
  if(features['recall.conditions']!==undefined&&typeof features['recall.conditions']!=='boolean')throw Error('Invalid recall conditions capability');
  if(features['images.understand']!==undefined&&typeof features['images.understand']!=='boolean')throw Error('Invalid image understanding capability');
  if(features['models.manage']!==undefined&&typeof features['models.manage']!=='boolean')throw Error('Invalid model management capability');
  return {schema_version:1, implementation:data.implementation, features:{...Object.fromEntries(FEATURE_KEYS.map(key => [key, features[key]])), 'episodes.attachments':features['episodes.attachments'] === true, 'episodes.list':features['episodes.list'] === true, 'episodes.read':features['episodes.read'] === true, 'facts.links':features['facts.links'] === true, 'integrity.read':features['integrity.read'] === true, 'profile.read':features['profile.read'] === true, 'processing.distill':features['processing.distill']===true, 'processing.derive':features['processing.derive']===true, 'jobs.read':features['jobs.read']===true, 'recall.conditions':features['recall.conditions']===true, 'models.manage':features['models.manage']===true, 'images.understand':features['images.understand']===true} as Capabilities['features']};
}
