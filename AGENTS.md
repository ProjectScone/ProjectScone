# Scone architecture boundary

Scone is a general-purpose framework. Memory, retrieval, ingestion, evaluation,
agent integrations and MCP must remain useful for arbitrary applications and
domains. The writing/editorial workspace is one application built on this
framework; it does not redefine the framework as a writing-only product.

- Keep writing-specific source filters, manuscript schemas and interfaces in
  optional writing modules or the web application.
- Preserve generic source kinds, framework APIs and harness/MCP integrations.
  Removing coding-session material from the writing UI does not authorize
  deleting stored records or removing general agent capabilities.
- Keep infrastructure locally managed. Optional writing-provider connections
  require explicit configuration; their availability does not enable other
  cloud infrastructure.
- Test shared contracts when adding an application adapter. Keep persistence,
  retrieval and model-provider concerns independently usable.

This boundary records the user's explicit direction on 2026-10-04.
