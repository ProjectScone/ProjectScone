//! C ABI for the Scone engine (spec §4): embed persistent temporal memory
//! in any language. Header: include/scone.h (kept in sync by hand — the
//! surface is seven functions; a test asserts the header lists them all).
//!
//! Contract: no panics cross the boundary (every entry point is wrapped),
//! errors are explicit (`NULL` / negative returns + `scone_last_error`),
//! and every returned string is owned by the caller until
//! `scone_free_string` (memory/bugs.md P-10: absent-data states are
//! documented, never conflated with crashes).

use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};

use scone_core::embed::HashEmbedder;
use scone_core::{Engine, IngestInput, IngestOutcome, RecallOpts, auth};

pub struct SconeEngine {
    engine: Engine,
    last_error: CString,
}

const FFI_HASH_DIM: usize = 256;

fn set_error(handle: &mut SconeEngine, message: impl Into<Vec<u8>>) {
    handle.last_error =
        CString::new(message).unwrap_or_else(|_| CString::from(c"error message had NUL"));
}

/// # Safety
/// `ptr` must be a valid NUL-terminated C string.
unsafe fn cstr<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(ptr) }.to_str().ok()
}

/// Open an engine at `data_dir`. `embedder_kind`: 0 = deterministic hash
/// (no model), 1 = local ONNX (downloads once; requires the local-embed
/// build). Returns NULL on failure.
///
/// # Safety
/// `data_dir` must be a valid NUL-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scone_open(
    data_dir: *const c_char,
    embedder_kind: u32,
) -> *mut SconeEngine {
    let result = catch_unwind(|| {
        let dir = unsafe { cstr(data_dir) }?;
        let embedder: Box<dyn scone_core::embed::EmbeddingProvider> = match embedder_kind {
            0 => Box::new(HashEmbedder::new(FFI_HASH_DIM)),
            #[cfg(feature = "local-embed")]
            1 => Box::new(
                scone_core::embed::OnnxEmbedder::new(&std::path::Path::new(dir).join("models"))
                    .ok()?,
            ),
            _ => return None,
        };
        let engine = Engine::open(std::path::Path::new(dir), embedder).ok()?;
        Some(Box::new(SconeEngine {
            engine,
            last_error: CString::from(c""),
        }))
    });
    match result {
        Ok(Some(handle)) => Box::into_raw(handle),
        _ => std::ptr::null_mut(),
    }
}

/// Close and free an engine. NULL is a no-op.
///
/// # Safety
/// `handle` must be NULL or a pointer returned by `scone_open`, not yet
/// closed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scone_close(handle: *mut SconeEngine) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

/// Store a note in `space`. Returns the episode id (> 0), 0 when the
/// content was already stored (deduplicated), or -1 on error (see
/// `scone_last_error`).
///
/// # Safety
/// `handle` must be a live engine; `space` and `text` valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scone_add_note(
    handle: *mut SconeEngine,
    space: *const c_char,
    text: *const c_char,
) -> i64 {
    let Some(state) = (unsafe { handle.as_mut() }) else {
        return -1;
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let (Some(space), Some(text)) = (unsafe { cstr(space) }, unsafe { cstr(text) }) else {
            set_error(state, "space/text must be valid UTF-8 C strings");
            return -1;
        };
        let scoped = match auth::resolve(&mut state.engine, space, true) {
            Ok(s) => s,
            Err(e) => {
                set_error(state, e.to_string());
                return -1;
            }
        };
        match state.engine.ingest(
            &scoped,
            IngestInput::Note {
                text: text.to_owned(),
            },
        ) {
            Ok(IngestOutcome::Ingested { episode_id, .. }) => episode_id,
            Ok(IngestOutcome::Deduplicated { .. }) => 0,
            Err(e) => {
                set_error(state, e.to_string());
                -1
            }
        }
    }));
    result.unwrap_or(-1)
}

/// Hybrid recall as a JSON string (`{"facts": [...], "items": [...]}`).
/// Returns NULL on error; free the result with `scone_free_string`.
///
/// # Safety
/// `handle` must be a live engine; `space` and `query` valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scone_recall_json(
    handle: *mut SconeEngine,
    space: *const c_char,
    query: *const c_char,
    limit: usize,
) -> *mut c_char {
    let Some(state) = (unsafe { handle.as_mut() }) else {
        return std::ptr::null_mut();
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let (Some(space), Some(query)) = (unsafe { cstr(space) }, unsafe { cstr(query) }) else {
            set_error(state, "space/query must be valid UTF-8 C strings");
            return std::ptr::null_mut();
        };
        let scoped = match auth::resolve(&mut state.engine, space, true) {
            Ok(s) => s,
            Err(e) => {
                set_error(state, e.to_string());
                return std::ptr::null_mut();
            }
        };
        let opts = RecallOpts {
            limit: limit.clamp(1, 50),
            budget_bytes: None,
            as_of: None,
            expand_neighbors: false,
            tags: Vec::new(),
        };
        match state.engine.recall(&scoped, query, &opts) {
            Ok(pack) => {
                let value = serde_json::json!({
                    "facts": pack.facts.iter().map(|f| serde_json::json!({
                        "fact_id": f.fact_id, "subject": f.subject,
                        "predicate": f.predicate, "object": f.object,
                        "confidence": f.confidence, "status": f.status,
                    })).collect::<Vec<_>>(),
                    "items": pack.items.iter().map(|i| serde_json::json!({
                        "episode_id": i.episode_id, "text": i.text, "score": i.score,
                    })).collect::<Vec<_>>(),
                });
                match CString::new(value.to_string()) {
                    Ok(s) => s.into_raw(),
                    Err(_) => std::ptr::null_mut(),
                }
            }
            Err(e) => {
                set_error(state, e.to_string());
                std::ptr::null_mut()
            }
        }
    }));
    result.unwrap_or(std::ptr::null_mut())
}

/// Free a string returned by this library. NULL is a no-op.
///
/// # Safety
/// `ptr` must be NULL or a string returned by this library, not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scone_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(unsafe { CString::from_raw(ptr) });
    }
}

/// Borrow the last error message for this engine. Valid until the next
/// call on the same engine. Empty string when no error has occurred.
///
/// # Safety
/// `handle` must be a live engine.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scone_last_error(handle: *const SconeEngine) -> *const c_char {
    match unsafe { handle.as_ref() } {
        Some(state) => state.last_error.as_ptr(),
        None => c"".as_ptr(),
    }
}
