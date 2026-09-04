#![allow(clippy::unwrap_used)]
//! How much of the benchmark's hardest category is arithmetic we can do
//! rather than language we must generate.
use scone_core::temporal::{Plan, plan};

#[test]
fn coverage_over_the_real_questions() {
    let path = "../../bench-data/longmemeval_s.json";
    let Ok(raw) = std::fs::read_to_string(path) else {
        eprintln!("dataset absent, skipping coverage probe");
        return;
    };
    let data: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let items = data.as_array().unwrap();
    let (mut total, mut interval, mut since, mut order, mut declined) = (0, 0, 0, 0, 0);
    let mut misses = Vec::new();
    for item in items {
        if item["question_type"].as_str() != Some("temporal-reasoning") {
            continue;
        }
        total += 1;
        let q = item["question"].as_str().unwrap_or_default();
        match plan(q) {
            Some(Plan::Interval { .. }) => interval += 1,
            Some(Plan::Since { .. }) => since += 1,
            Some(Plan::Order { .. }) => order += 1,
            None => {
                declined += 1;
                if misses.len() < 12 {
                    misses.push(q.to_owned());
                }
            }
        }
    }
    let covered = total - declined;
    println!("temporal questions: {total}");
    println!("  interval {interval}  since {since}  order {order}");
    println!("  covered {covered} ({}%)", covered * 100 / total.max(1));
    println!("  declined {declined}");
    for m in &misses {
        println!("  ? {}", &m[..m.len().min(88)]);
    }
}
