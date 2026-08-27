#![allow(clippy::unwrap_used)]
use std::ffi::{CStr, CString};

use scone_ffi::{
    scone_add_note, scone_close, scone_free_string, scone_last_error, scone_open, scone_recall_json,
};

#[test]
fn open_add_recall_close_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = CString::new(dir.path().to_str().unwrap()).unwrap();
    let space = CString::new("default").unwrap();
    unsafe {
        let engine = scone_open(path.as_ptr(), 0);
        assert!(!engine.is_null());
        let text = CString::new("the ffi layer works end to end").unwrap();
        let id = scone_add_note(engine, space.as_ptr(), text.as_ptr());
        assert!(id > 0, "{:?}", CStr::from_ptr(scone_last_error(engine)));
        let dup = scone_add_note(engine, space.as_ptr(), text.as_ptr());
        assert_eq!(dup, 0, "duplicate reports 0");
        let query = CString::new("ffi layer").unwrap();
        let json = scone_recall_json(engine, space.as_ptr(), query.as_ptr(), 5);
        assert!(!json.is_null());
        let parsed: serde_json::Value =
            serde_json::from_str(CStr::from_ptr(json).to_str().unwrap()).unwrap();
        assert!(
            parsed["items"][0]["text"]
                .as_str()
                .unwrap()
                .contains("ffi layer"),
            "{parsed}"
        );
        scone_free_string(json);
        scone_close(engine);
    }
}

#[test]
fn errors_are_reported_not_crashed() {
    let dir = tempfile::tempdir().unwrap();
    let path = CString::new(dir.path().to_str().unwrap()).unwrap();
    unsafe {
        let engine = scone_open(path.as_ptr(), 99);
        assert!(engine.is_null(), "unknown embedder kind yields NULL");
        let engine = scone_open(path.as_ptr(), 0);
        let bad_space = CString::new("BAD SPACE!").unwrap();
        let text = CString::new("x").unwrap();
        let r = scone_add_note(engine, bad_space.as_ptr(), text.as_ptr());
        assert_eq!(r, -1);
        let err = CStr::from_ptr(scone_last_error(engine)).to_str().unwrap();
        assert!(err.contains("space name"), "{err}");
        let r = scone_add_note(engine, std::ptr::null(), text.as_ptr());
        assert_eq!(r, -1, "null pointers are errors, not crashes");
        scone_close(engine);
        scone_close(std::ptr::null_mut());
        scone_free_string(std::ptr::null_mut());
    }
}

#[test]
fn header_declares_the_whole_surface() {
    let header = include_str!("../include/scone.h");
    for symbol in [
        "scone_open",
        "scone_close",
        "scone_add_note",
        "scone_recall_json",
        "scone_free_string",
        "scone_last_error",
    ] {
        assert!(header.contains(symbol), "header missing {symbol}");
    }
}
