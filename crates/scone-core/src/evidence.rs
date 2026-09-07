//! Scoped diagnostic evidence. Receipt order is independent of source time.
//! Agent records are connector reports; HTTP recall records are engine output.
//! Native callers opt in to recording; legacy operations are not reconstructed.
use crate::{Engine, Result, SconeError, auth::ScopedSpace};
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub(crate) fn init(conn: &Connection) -> Result<()> {
    conn.execute_batch("CREATE TABLE IF NOT EXISTS evidence_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, space_id INTEGER NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
        ts TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL, source_id TEXT,
        UNIQUE(space_id,source_id));
        CREATE INDEX IF NOT EXISTS evidence_space_id ON evidence_events(space_id,id);
        CREATE TABLE IF NOT EXISTS episode_metadata (
        episode_id INTEGER PRIMARY KEY REFERENCES episodes(id) ON DELETE CASCADE, metadata TEXT NOT NULL);
        INSERT OR IGNORE INTO meta(key,value) VALUES('evidence_schema_version','1');")?;
    Ok(())
}
fn invalid(message: &str) -> SconeError {
    SconeError::InvalidInput(message.into())
}
fn bounded(value: &Value, name: &str, max: usize, required: bool) -> Result<()> {
    match value.get(name) {
        None | Some(Value::Null) if !required => Ok(()),
        Some(Value::String(s)) if !s.is_empty() && s.len() <= max => Ok(()),
        _ => Err(invalid(&format!(
            "{name} must be a string of 1..={max} UTF-8 bytes"
        ))),
    }
}
fn redact(text: &str) -> String {
    // Defense in depth for recognizable credential tokens. Not a complete
    // secret classifier; the connector must redact before sending content.
    let mut out = text.to_owned();
    for token in text.split_whitespace() {
        if ["sk-", "ghp_", "github_pat_", "xoxb-", "xoxp-"]
            .iter()
            .any(|p| token.contains(p))
        {
            out = out.replace(token, "[redacted]");
        }
    }
    out
}
impl Engine {
    pub fn record_agent_event(&mut self, space: &ScopedSpace, mut payload: Value) -> Result<Value> {
        let map = payload
            .as_object()
            .ok_or_else(|| invalid("payload must be an object"))?;
        let allowed = [
            "agent",
            "session_id",
            "event",
            "project",
            "text",
            "tool_name",
            "tool_use_id",
            "ok",
            "duration_ms",
            "episode_id",
            "model",
            "source_event_id",
            "turn_id",
            "text_truncated",
        ];
        if map.keys().any(|k| !allowed.contains(&k.as_str())) {
            return Err(invalid("unknown agent event field"));
        }
        if !["claude-code", "codex", "other"].contains(&payload["agent"].as_str().unwrap_or("")) {
            return Err(invalid("unknown agent"));
        }
        if ![
            "session_start",
            "prompt",
            "response",
            "tool_use",
            "tool_result",
            "stop",
            "session_end",
        ]
        .contains(&payload["event"].as_str().unwrap_or(""))
        {
            return Err(invalid("unknown agent event"));
        }
        bounded(&payload, "session_id", 128, true)?;
        for (name, max) in [
            ("project", 120),
            ("tool_name", 120),
            ("tool_use_id", 128),
            ("model", 120),
            ("source_event_id", 128),
            ("turn_id", 128),
        ] {
            bounded(&payload, name, max, false)?;
        }
        if let Some(text) = payload.get("text").filter(|v| !v.is_null()) {
            let text = text
                .as_str()
                .ok_or_else(|| invalid("text must be a string"))?;
            if text.len() > 65536 {
                return Err(invalid("text exceeds 65536 UTF-8 bytes"));
            }
            payload["text"] = json!(redact(text));
        }
        if payload
            .get("ok")
            .is_some_and(|v| !v.is_null() && !v.is_boolean())
        {
            return Err(invalid("ok must be boolean"));
        }
        if payload
            .get("text_truncated")
            .is_some_and(|v| !v.is_boolean())
        {
            return Err(invalid("text_truncated must be boolean"));
        }
        if let Some(v) = payload.get("duration_ms").filter(|v| !v.is_null())
            && v.as_f64().is_none_or(|n| !n.is_finite() || n < 0.)
        {
            return Err(invalid("duration_ms must be nonnegative and finite"));
        }
        if let Some(id) = payload.get("episode_id").filter(|v| !v.is_null()) {
            let id = id
                .as_i64()
                .filter(|n| *n > 0)
                .ok_or_else(|| invalid("episode_id must be a positive integer"))?;
            if self.evidence_episode(space, id).is_err() {
                return Err(invalid("episode_id is not in the authenticated space"));
            }
        }
        let source_id = payload["source_event_id"]
            .as_str()
            .map(|id| json!([payload["agent"], payload["session_id"], id]).to_string());
        self.append_evidence(space, "agent", payload, source_id)
    }
    fn append_evidence(
        &mut self,
        space: &ScopedSpace,
        kind: &str,
        payload: Value,
        source_id: Option<String>,
    ) -> Result<Value> {
        let serialized = payload.to_string();
        if serialized.len() > 100_000 {
            return Err(invalid("event exceeds 100000 UTF-8 bytes"));
        }
        // IMMEDIATE serializes the lookup and insert across separate processes.
        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        if let Some(ref key) = source_id {
            let existing: Option<(i64, String)> = tx
                .query_row(
                    "SELECT id,payload FROM evidence_events WHERE space_id=?1 AND source_id=?2",
                    params![space.id(), key],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .optional()?;
            if let Some((id, prior)) = existing {
                if prior != serialized {
                    return Err(invalid("source_event_id already has different content"));
                }
                return Ok(json!({"recorded":id}));
            }
        }
        tx.execute("INSERT INTO evidence_events(space_id,ts,kind,payload,source_id) VALUES(?1,?2,?3,?4,?5)",params![space.id(),crate::temporal::now_rfc3339(),kind,serialized,source_id])?;
        let id = tx.last_insert_rowid();
        tx.commit()?;
        Ok(json!({"recorded":id}))
    }
    pub fn record_recall_evidence(
        &mut self,
        space: &ScopedSpace,
        query: &str,
        pack: &crate::ContextPack,
    ) -> Result<i64> {
        let receipt=self.append_evidence(space,"recall",json!({
            "query_hashed":true,"query_hash":blake3::hash(query.as_bytes()).to_hex().to_string(),
            "items":pack.items.iter().map(|i|json!({"chunk_id":i.chunk_id,"episode_id":i.episode_id,"score":i.score,"similarity":i.similarity})).collect::<Vec<_>>(),
            "fact_ids":pack.facts.iter().map(|f|f.fact_id).collect::<Vec<_>>(),"returned_bytes":pack.returned_bytes,
            "coverage":"HTTP recall; returned items only; lane ranks unavailable"
        }),None)?;
        receipt["recorded"]
            .as_i64()
            .ok_or_else(|| invalid("invalid receipt"))
    }
    pub fn evidence_events(
        &self,
        space: &ScopedSpace,
        after: Option<i64>,
        limit: usize,
        kind: Option<&str>,
    ) -> Result<Value> {
        if after.is_some_and(|n| n < 0) {
            return Err(invalid("after_id must be nonnegative"));
        }
        let limit = limit.clamp(1, 1000);
        let order = if after.is_some() { "ASC" } else { "DESC" };
        let mut stmt=self.conn.prepare(&format!("SELECT id,ts,kind,payload FROM evidence_events WHERE space_id=?1 AND id>?2 AND (?3 IS NULL OR kind=?3) ORDER BY id {order} LIMIT ?4"))?;
        let rows = stmt.query_map(
            params![space.id(), after.unwrap_or(0), kind, (limit + 1) as i64],
            |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                ))
            },
        )?;
        let mut events = Vec::new();
        for row in rows {
            let (id, ts, kind, payload) = row?;
            let payload: Value =
                serde_json::from_str(&payload).map_err(|_| invalid("stored event is malformed"))?;
            events.push(json!({"event_id":id,"ts":ts,"kind":kind,"space":space.name(),"schema_version":1,"payload":payload}));
        }
        let more = events.len() > limit;
        events.truncate(limit);
        let next = if after.is_some() {
            events
                .last()
                .and_then(|e| e["event_id"].as_i64())
                .unwrap_or(after.unwrap_or(0))
        } else {
            0
        };
        Ok(
            json!({"events":events,"evidence":"sqlite","next_after_id":next,"truncated":more,"coverage":"retained diagnostic events; native callers opt in"}),
        )
    }
    pub fn set_episode_metadata(
        &mut self,
        space: &ScopedSpace,
        id: i64,
        metadata: &BTreeMap<String, String>,
    ) -> Result<()> {
        self.evidence_episode(space, id)?;
        if metadata.len() > 32
            || metadata
                .iter()
                .any(|(k, v)| k.is_empty() || k.len() > 64 || v.len() > 256)
        {
            return Err(invalid(
                "metadata exceeds 32 keys, 64-byte keys or 256-byte values",
            ));
        }
        // Content dedup preserves the first episode's metadata, as provenance.
        self.conn.execute(
            "INSERT OR IGNORE INTO episode_metadata(episode_id,metadata) VALUES(?1,?2)",
            params![id, json!(metadata).to_string()],
        )?;
        Ok(())
    }
    pub fn evidence_episode(&self, space: &ScopedSpace, id: i64) -> Result<Value> {
        let row=self.conn.query_row("SELECT e.kind,e.content,e.source,e.created_at,m.metadata FROM episodes e LEFT JOIN episode_metadata m ON m.episode_id=e.id WHERE e.space_id=?1 AND e.id=?2",params![space.id(),id],|r|Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?,r.get::<_,Option<String>>(2)?,r.get::<_,String>(3)?,r.get::<_,Option<String>>(4)?))).optional()?;
        let (kind, content, source, created_at, metadata) = row.ok_or_else(|| {
            SconeError::NotFound("episode not found in authenticated space".into())
        })?;
        Ok(
            json!({"episode_id":id,"kind":kind,"content":content,"source":source,"created_at":created_at,"metadata":metadata.and_then(|s|serde_json::from_str::<Value>(&s).ok()).unwrap_or(json!({}))}),
        )
    }
    /// Inventory summaries in descending episode-ID order, independent of
    /// relevance and source timestamps. A deleted boundary remains usable.
    /// This is a live keyset walk, not a transactionally frozen snapshot.
    pub fn source_page(
        &self,
        space: &ScopedSpace,
        before: Option<i64>,
        limit: usize,
        kind: Option<&str>,
    ) -> Result<Value> {
        if !(1..=100).contains(&limit) || before.is_some_and(|id| id < 1) {
            return Err(invalid(
                "limit must be 1..=100 and before a positive episode ID",
            ));
        }
        if kind.is_some_and(|k| {
            !["note", "file", "conversation", "observation", "connector"].contains(&k)
        }) {
            return Err(invalid("unknown episode kind"));
        }
        // Do not load every content body merely to build a navigation page.
        // Read enough UTF-8 bytes for 501 scalar values, including embedded NULs
        // (SQLite's text substr stops at NUL). A cut final codepoint is beyond
        // the first 500 characters and is never included in the preview.
        // Where consolidation left each source, in the words the API means:
        // cited (a claim rests on it), parked (the queue gave up on it),
        // done (visited, nothing to cite), pending (the queue will visit it).
        let mut sql = String::from(
            "SELECT id,kind,source,created_at,length(CAST(content AS BLOB)),substr(CAST(content AS BLOB),1,2004),\
             (SELECT count(*) FROM fact_provenance fp JOIN facts f ON f.id = fp.fact_id \
              WHERE fp.episode_id = episodes.id AND f.space_id = episodes.space_id),\
             (SELECT state FROM distill_queue q WHERE q.episode_id = episodes.id),\
             (SELECT last_error FROM distill_queue q WHERE q.episode_id = episodes.id) \
             FROM episodes WHERE space_id=?",
        );
        let mut parameters = vec![rusqlite::types::Value::Integer(space.id())];
        if let Some(id) = before {
            sql.push_str(" AND id < ?");
            parameters.push(id.into());
        }
        if let Some(k) = kind {
            sql.push_str(" AND kind = ?");
            parameters.push(k.to_owned().into());
        }
        sql.push_str(" ORDER BY id DESC LIMIT ?");
        parameters.push(((limit + 1) as i64).into());
        let mut statement = self.conn.prepare(&sql)?;
        let mut items = statement.query_map(rusqlite::params_from_iter(parameters), |row| {
            let bytes: Vec<u8> = row.get(5)?;
            let preview = String::from_utf8_lossy(&bytes);
            let cited: i64 = row.get(6)?;
            let queued: Option<String> = row.get(7)?;
            let last_error: Option<String> = row.get(8)?;
            let mut item = json!({"episode_id":row.get::<_,i64>(0)?,"kind":row.get::<_,String>(1)?,
                "source":row.get::<_,Option<String>>(2)?,"created_at":row.get::<_,String>(3)?,
                "byte_count":row.get::<_,i64>(4)?,"preview":preview.chars().take(500).collect::<String>(),
                "preview_truncated":preview.chars().count()>500});
            let status = if cited > 0 {
                "cited"
            } else {
                match queued.as_deref() {
                    Some("failed") => "parked",
                    Some("done") => "done",
                    _ => "pending",
                }
            };
            item["status"] = json!(status);
            if status == "parked" {
                let reason: String = last_error.unwrap_or_default().chars().take(200).collect();
                item["parked_reason"] = json!(reason);
            }
            Ok(item)
        })?.collect::<std::result::Result<Vec<_>,_>>()?;
        let has_more = items.len() > limit;
        items.truncate(limit);
        let next_before = if has_more {
            items.last().map(|item| item["episode_id"].clone())
        } else {
            None
        };
        Ok(json!({"items":items,"has_more":has_more,"next_before":next_before}))
    }

    pub fn evidence_graph(&self, space: &ScopedSpace, limit: usize) -> Result<Value> {
        let limit = limit.clamp(1, 400);
        let mut nodes = BTreeMap::<String, Value>::new();
        let mut edges = Vec::<Value>::new();
        let mut truncated = false;
        let page = self.evidence_events(space, None, limit, None)?;
        let events = page["events"]
            .as_array()
            .ok_or_else(|| invalid("invalid evidence page"))?;
        truncated |= page["truncated"] == true;
        for e in events {
            let p = &e["payload"];
            let id = e["event_id"].as_i64().unwrap_or(0);
            if e["kind"] == "agent" {
                let session = format!(
                    "session:{}:{}",
                    p["agent"].as_str().unwrap_or("other"),
                    p["session_id"].as_str().unwrap_or("")
                );
                nodes.entry(session.clone()).or_insert_with(||json!({"id":session,"kind":"session","label":format!("{} session",p["agent"].as_str().unwrap_or("agent")),"ts":null,"data":{"agent":p["agent"],"session_id":p["session_id"],"project":p["project"],"provenance":"connector-reported"}}));
                let is_tool =
                    ["tool_use", "tool_result"].contains(&p["event"].as_str().unwrap_or(""));
                let nid = format!("{}:{id}", if is_tool { "tool" } else { "turn" });
                nodes.insert(nid.clone(),json!({"id":nid,"kind":if is_tool{"tool_call"}else{"turn"},"label":if is_tool{p["tool_name"].clone()}else{p["event"].clone()},"ts":e["ts"],"data":p}));
                edges.push(
                    json!({"source":session,"target":nid,"kind":if is_tool{"invoked"}else{"has"}}),
                );
                if let Some(eid) = p["episode_id"].as_i64() {
                    edges.push(json!({"source":nid,"target":format!("episode:{eid}"),"kind":"captured_as"}));
                }
            } else if e["kind"] == "recall" {
                let nid = format!("recall:{id}");
                nodes.insert(nid.clone(),json!({"id":nid,"kind":"recall","label":"recall (query hashed)","ts":e["ts"],"data":p}));
                if let Some(items) = p["items"].as_array() {
                    for i in items {
                        edges.push(json!({"source":nid,"target":format!("chunk:{}",i["chunk_id"]),"kind":"returned","label":"retrieval evidence; lane ranks unavailable","data":i}));
                    }
                }
                if let Some(facts) = p["fact_ids"].as_array() {
                    for f in facts {
                        edges.push(
                            json!({"source":nid,"target":format!("claim:{f}"),"kind":"held"}),
                        );
                    }
                }
            }
        }
        let mut stmt = self
            .conn
            .prepare("SELECT id FROM episodes WHERE space_id=?1 ORDER BY id DESC LIMIT ?2")?;
        let ids = stmt
            .query_map(params![space.id(), (limit + 1) as i64], |r| {
                r.get::<_, i64>(0)
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        truncated |= ids.len() > limit;
        for id in ids.into_iter().take(limit) {
            let ep = self.evidence_episode(space, id)?;
            let nid = format!("episode:{id}");
            let label = ep["content"]
                .as_str()
                .unwrap_or("")
                .chars()
                .take(60)
                .collect::<String>();
            nodes.insert(
                nid.clone(),
                json!({"id":nid,"kind":"episode","label":label,"ts":ep["created_at"],"data":ep}),
            );
            let mut stmt=self.conn.prepare("SELECT id,pos,start_byte,end_byte FROM chunks WHERE episode_id=?1 ORDER BY pos LIMIT ?2")?;
            let rows = stmt.query_map(params![id, (limit + 1) as i64], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, i64>(3)?,
                ))
            })?;
            for (count, row) in rows.enumerate() {
                if count >= limit {
                    truncated = true;
                    break;
                }
                let (cid, ordinal, start, end) = row?;
                let start =
                    usize::try_from(start).map_err(|_| invalid("negative chunk byte offset"))?;
                let end =
                    usize::try_from(end).map_err(|_| invalid("negative chunk byte offset"))?;
                let text = ep["content"]
                    .as_str()
                    .unwrap_or("")
                    .get(start..end)
                    .unwrap_or("");
                let chunk = format!("chunk:{cid}");
                nodes.insert(chunk.clone(),json!({"id":chunk,"kind":"chunk","label":format!("Passage {}",ordinal+1),"ts":ep["created_at"],"data":{"ordinal":ordinal,"start":start,"end":end,"text":text}}));
                edges.push(json!({"source":nid,"target":chunk,"kind":"chunked_into"}));
            }
        }
        for f in self.facts_list(space, true)?.into_iter().take(limit) {
            let id = format!("claim:{}", f.fact_id);
            nodes.insert(id.clone(),json!({"id":id,"kind":"claim","label":format!("{} {} {}",f.subject,f.predicate,f.object),"ts":f.valid_from,"data":{"status":f.status,"valid_from":f.valid_from,"valid_until":f.valid_until,"confidence":f.confidence,"origin":null,"coverage":"origin unavailable on Rust fact record"}}));
            let mut stmt=self.conn.prepare("SELECT fp.episode_id FROM fact_provenance fp JOIN episodes e ON e.id=fp.episode_id WHERE fp.fact_id=?1 AND e.space_id=?2")?;
            for eid in stmt.query_map(params![f.fact_id, space.id()], |r| r.get::<_, i64>(0))? {
                edges.push(
                    json!({"source":format!("episode:{}",eid?),"target":id,"kind":"source_of"}),
                );
            }
        }
        // Bound the rendered projection; missing endpoints are explicit coverage,
        // never synthesized. Underlying events/episodes remain independently readable.
        if nodes.len() > limit {
            truncated = true;
            nodes = nodes.into_iter().take(limit).collect();
        }
        edges.retain(|e| {
            nodes.contains_key(e["source"].as_str().unwrap_or(""))
                && nodes.contains_key(e["target"].as_str().unwrap_or(""))
        });
        Ok(
            json!({"nodes":nodes.into_values().collect::<Vec<_>>(),"edges":edges,"truncated":truncated,"evidence":"sqlite","coverage":"bounded retained snapshot; HTTP recall recorded; no inferred edges"}),
        )
    }
}
