/* Scone memory engine — C ABI. See crates/scone-ffi/src/lib.rs for the
 * authoritative contracts. Strings in are NUL-terminated UTF-8; strings
 * out are owned by the caller and freed with scone_free_string. */
#ifndef SCONE_H
#define SCONE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SconeEngine SconeEngine;

/* embedder_kind: 0 = deterministic hash (no model), 1 = local ONNX. */
SconeEngine *scone_open(const char *data_dir, uint32_t embedder_kind);
void scone_close(SconeEngine *engine);

/* Returns episode id (>0), 0 if deduplicated, -1 on error. */
int64_t scone_add_note(SconeEngine *engine, const char *space, const char *text);

/* Returns {"facts": [...], "items": [...]} as JSON, or NULL on error. */
char *scone_recall_json(SconeEngine *engine, const char *space,
                        const char *query, size_t limit);

void scone_free_string(char *ptr);

/* Borrowed; valid until the next call on the same engine. */
const char *scone_last_error(const SconeEngine *engine);

#ifdef __cplusplus
}
#endif

#endif /* SCONE_H */
