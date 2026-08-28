#![allow(clippy::unwrap_used)]
//! HTTP providers tested against an in-process TCP stub — no network.
use std::io::{Read, Write};
use std::net::TcpListener;

use scone_core::llm::{AnthropicProvider, LlmProvider, OpenAiCompatible};

/// Serve one HTTP request with a canned JSON body; return what we read.
fn stub_once(body: &'static str) -> (String, std::thread::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = std::thread::spawn(move || {
        let (mut sock, _) = listener.accept().unwrap();
        let mut buf = [0u8; 65536];
        let mut req = String::new();
        loop {
            let n = sock.read(&mut buf).unwrap();
            req.push_str(&String::from_utf8_lossy(&buf[..n]));
            // Read until the JSON body closes (requests here are small).
            if req.contains("\r\n\r\n") && req.trim_end().ends_with('}') {
                break;
            }
        }
        let resp = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        sock.write_all(resp.as_bytes()).unwrap();
        req
    });
    (format!("http://{addr}"), handle)
}

#[test]
fn openai_compatible_parses_extraction_json() {
    let body = r#"{"choices":[{"message":{"content":"[{\"subject\":\"mark\",\"predicate\":\"prefers\",\"object\":\"bun\",\"confidence\":0.9}]"}}]}"#;
    let (url, handle) = stub_once(body);
    let p = OpenAiCompatible::new(&url, "test-model", None);
    let facts = p.extract_facts("mark prefers bun").unwrap();
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].object, "bun");
    let req = handle.join().unwrap();
    assert!(req.contains("test-model"), "model travels in the request");
    assert!(req.contains("POST /chat/completions"), "{req}");
}

#[test]
fn openai_compatible_reports_unparseable_model_output_typed() {
    let body = r#"{"choices":[{"message":{"content":"sorry, I cannot do that"}}]}"#;
    let (url, _handle) = stub_once(body);
    let p = OpenAiCompatible::new(&url, "m", None);
    let err = p.extract_facts("text").unwrap_err();
    assert!(err.to_string().contains("did not return JSON"), "{err}");
}

#[test]
fn anthropic_wire_format_and_auth_header() {
    let body = r#"{"content":[{"type":"text","text":"[{\"subject\":\"mark\",\"predicate\":\"uses\",\"object\":\"rust\",\"confidence\":0.8}]"}]}"#;
    let (url, handle) = stub_once(body);
    let p = AnthropicProvider::new(&url, "claude-sonnet-5", "sk-test");
    let facts = p.extract_facts("mark uses rust").unwrap();
    assert_eq!(facts[0].predicate, "uses");
    let req = handle.join().unwrap();
    assert!(req.contains("POST /v1/messages"), "{req}");
    assert!(req.contains("x-api-key: sk-test"), "{req}");
    assert!(req.contains("anthropic-version"), "{req}");
}

#[test]
fn answer_returns_model_text() {
    let body = r#"{"choices":[{"message":{"content":"bun, since 2026"}}]}"#;
    let (url, _handle) = stub_once(body);
    let p = OpenAiCompatible::new(&url, "m", Some("key".into()));
    let a = p
        .answer("what does mark prefer?", "facts: mark prefers bun")
        .unwrap();
    assert_eq!(a, "bun, since 2026");
}

#[test]
fn a_dead_server_times_out_instead_of_hanging_forever() {
    // Accepts the connection, never responds.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let _keep = std::thread::spawn(move || {
        let (_sock, _) = listener.accept().unwrap();
        std::thread::sleep(std::time::Duration::from_secs(60));
    });
    let p = OpenAiCompatible::new(&format!("http://{addr}"), "m", None)
        .with_timeout(std::time::Duration::from_millis(500));
    let started = std::time::Instant::now();
    let err = p.answer("q", "ctx").unwrap_err();
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "must not hang"
    );
    assert!(err.to_string().contains("http"), "{err}");
}
