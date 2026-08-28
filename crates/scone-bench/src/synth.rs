//! Synthetic-from-real dataset generation (methodology gate, 2026-08-28):
//! real published prose supplies carrier sessions and distractors; the
//! generator injects facts it controls, so ground truth is exact by
//! construction. Emits LongMemEval-format JSON so the existing harness
//! runs unchanged.

pub struct SynthConfig {
    pub items: usize,
    pub sessions_per_item: usize,
    pub seed: u64,
}

/// Deterministic xorshift, mirroring stratified_sample's approach.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn pick(&mut self, bound: usize) -> usize {
        (self.next() as usize) % bound.max(1)
    }
}

/// Capitalized word runs from real text become the entity pool.
fn harvest_entities(text: &str) -> Vec<String> {
    let mut entities: Vec<String> = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    for token in text.split_whitespace() {
        let word = token.trim_matches(|c: char| !c.is_alphanumeric());
        let capitalized =
            word.chars().next().is_some_and(|c| c.is_ascii_uppercase()) && word.len() > 2;
        if capitalized {
            current.push(word);
        } else {
            if !current.is_empty() {
                let entity = current.join(" ");
                if !entities.contains(&entity) {
                    entities.push(entity);
                }
            }
            current.clear();
        }
    }
    if !current.is_empty() {
        let entity = current.join(" ");
        if !entities.contains(&entity) {
            entities.push(entity);
        }
    }
    entities
}

fn sentences(text: &str) -> Vec<String> {
    text.split(['.', '!', '?'])
        .map(str::trim)
        .filter(|s| s.len() > 20)
        .map(str::to_owned)
        .collect()
}

const OBJECTS: [&str; 8] = [
    "the north annex",
    "budget code 7741",
    "the cedar room",
    "route 9",
    "pier 4",
    "the amber protocol",
    "desk 12",
    "channel 6",
];

pub fn generate(real_text: &str, cfg: &SynthConfig) -> Result<String, String> {
    let entities = harvest_entities(real_text);
    let carriers = sentences(real_text);
    if entities.len() < 3 || carriers.len() < 3 {
        return Err("source text too small: need >=3 entities and sentences".into());
    }
    let mut rng = Rng(cfg.seed.max(1));
    let types = [
        "lookup",
        "knowledge-update",
        "temporal-reasoning",
        "multi-session",
        "abstention",
    ];
    let mut items = Vec::new();
    for i in 0..cfg.items {
        let qtype = types[i % types.len()];
        let entity = entities[rng.pick(entities.len())].clone();
        let object = OBJECTS[rng.pick(OBJECTS.len())];
        let alt_object = OBJECTS[(rng.pick(OBJECTS.len() - 1) + 1) % OBJECTS.len()];
        let n_sessions = cfg.sessions_per_item.max(2);
        // Sessions of real carrier prose, dated sequentially.
        let mut sessions: Vec<Vec<String>> = (0..n_sessions)
            .map(|_| {
                vec![
                    format!("user: {}", carriers[rng.pick(carriers.len())]),
                    format!("assistant: {}", carriers[rng.pick(carriers.len())]),
                ]
            })
            .collect();
        let dates: Vec<String> = (0..n_sessions)
            .map(|s| format!("2024/0{}/1{} (Mon) 09:00", (s % 8) + 1, s % 9))
            .collect();
        let ids: Vec<String> = (0..n_sessions).map(|s| format!("syn_{i}_{s}")).collect();
        let (question, answer, evidence): (String, String, Vec<String>) = match qtype {
            "lookup" => {
                let ev = rng.pick(n_sessions);
                sessions[ev].push(format!("user: note that {entity} moved to {object}"));
                (
                    format!("where did {entity} move to?"),
                    object.to_owned(),
                    vec![ids[ev].clone()],
                )
            }
            "knowledge-update" => {
                sessions[0].push(format!("user: {entity} is assigned to {alt_object}"));
                let ev = n_sessions - 1;
                sessions[ev].push(format!(
                    "user: update: {entity} is now assigned to {object}"
                ));
                (
                    format!("what is {entity} assigned to now?"),
                    object.to_owned(),
                    vec![ids[ev].clone()],
                )
            }
            "temporal-reasoning" => {
                let ev = rng.pick(n_sessions);
                sessions[ev].push(format!("user: on that day {entity} relocated to {object}"));
                (
                    format!(
                        "in {} what did {entity} relocate to?",
                        month_name(&dates[ev])
                    ),
                    object.to_owned(),
                    vec![ids[ev].clone()],
                )
            }
            "multi-session" => {
                sessions[0].push(format!("user: half the plan: {entity} covers {object}"));
                let ev = n_sessions - 1;
                sessions[ev].push(format!(
                    "user: other half: the backup for {object} is {alt_object}"
                ));
                (
                    format!("what covers {entity}'s area and what is its backup?"),
                    object.to_owned(),
                    vec![ids[0].clone(), ids[ev].clone()],
                )
            }
            _ => (
                format!("what did {entity} say about {object}?"),
                format!("nothing recorded links {entity} and {object}"),
                vec![ids[0].clone()],
            ),
        };
        items.push(serde_json::json!({
            "question_id": format!("syn_{}_{qtype}", i),
            "question_type": qtype,
            "question": question,
            "answer": answer,
            "question_date": "2024/09/01 (Sun) 12:00",
            "haystack_sessions": sessions
                .iter()
                .map(|s| s.iter().map(|line| {
                    let (role, content) = line.split_once(": ").unwrap_or(("user", line));
                    serde_json::json!({"role": role, "content": content})
                }).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            "haystack_dates": dates,
            "haystack_session_ids": ids,
            "answer_session_ids": evidence,
        }));
    }
    serde_json::to_string_pretty(&items).map_err(|e| e.to_string())
}

fn month_name(date: &str) -> &'static str {
    const NAMES: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    date.get(5..7)
        .and_then(|m| m.parse::<usize>().ok())
        .and_then(|m| NAMES.get(m.saturating_sub(1)))
        .copied()
        .unwrap_or("January")
}
