#![allow(clippy::unwrap_used)]
use assert_cmd::Command;
use serde_json::{Value, json};

#[test]
fn every_prompt_is_structured_without_starting_a_model_or_store() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/prompt-contract.json")).unwrap();
    let dir = tempfile::tempdir().unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let output = Command::cargo_bin("scone")
            .unwrap()
            .args(["prompt-hook"])
            .current_dir(dir.path())
            .write_stdin(
                json!({"hook_event_name":"UserPromptSubmit","prompt":case["input"]}).to_string(),
            )
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let result: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(
            result["hookSpecificOutput"]["hookEventName"],
            "UserPromptSubmit"
        );
        let context = result["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap();
        let payload: Value = serde_json::from_str(
            context
                .strip_prefix(fixture["context_prefix"].as_str().unwrap())
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            payload,
            json!({"schema_version":1,"task":case["task"],"instructions":fixture["instructions"]})
        );
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }
}

#[test]
fn nonprompt_and_malformed_input_do_not_inject_or_block() {
    for input in [
        "not JSON",
        r#"{"hook_event_name":"Stop","prompt":"not a user turn"}"#,
        r#"{"hook_event_name":"UserPromptSubmit","prompt":"  "}"#,
    ] {
        let output = Command::cargo_bin("scone")
            .unwrap()
            .arg("prompt-hook")
            .write_stdin(input)
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        assert_eq!(serde_json::from_slice::<Value>(&output).unwrap(), json!({}));
    }
}
