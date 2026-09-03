//! Cloud connectors: pull documents out of the services people already
//! keep their thinking in, and hand them to the engine as episodes.
//!
//! Network lives here in the CLI layer, exactly like [`crate::web`];
//! scone-core stays offline by construction. Credentials live in a
//! 0600 file next to the database, or in the environment for people who
//! would rather not have them on disk at all.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_secs(30);
const MAX_BODY_BYTES: u64 = 5_000_000;
const NOTION_VERSION: &str = "2022-06-28";

/// One fetched thing, normalized across services.
#[derive(Debug, PartialEq)]
pub struct Document {
    /// Stable id at the source, so re-syncs land on the same episode.
    pub id: String,
    pub title: String,
    /// Where a human would go to read it.
    pub url: String,
    pub body: String,
    /// Source's own timestamp, RFC3339. Memory is dated by when the
    /// thing happened, never by when we happened to sync it.
    pub updated_at: Option<String>,
}

/// What every connector must do: name itself and produce documents.
pub trait Connector {
    fn name(&self) -> &'static str;
    /// Fetch documents changed since `since` (RFC3339), or everything.
    fn fetch(&self, since: Option<&str>) -> Result<Vec<Document>, String>;
}

/// Stored connector credentials and sync state.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Credentials {
    #[serde(default)]
    pub providers: std::collections::BTreeMap<String, Provider>,
}

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Provider {
    pub token: String,
    /// RFC3339 timestamp of the last successful sync, for incrementals.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_sync: Option<String>,
}

pub fn credentials_path(data_dir: &Path) -> PathBuf {
    data_dir.join("connectors.toml")
}

pub fn load_credentials(data_dir: &Path) -> Result<Credentials, String> {
    let path = credentials_path(data_dir);
    let raw = std::fs::read_to_string(&path).unwrap_or_default();
    if raw.trim().is_empty() {
        return Ok(Credentials::default());
    }
    toml::from_str(&raw).map_err(|e| format!("{}: {e}", path.display()))
}

/// Write credentials readable only by this user. A token in a
/// world-readable file is a token you have given away.
pub fn save_credentials(data_dir: &Path, creds: &Credentials) -> Result<(), String> {
    let path = credentials_path(data_dir);
    std::fs::create_dir_all(data_dir).map_err(|e| e.to_string())?;
    let body = toml::to_string_pretty(creds).map_err(|e| e.to_string())?;
    std::fs::write(&path, body).map_err(|e| format!("{}: {e}", path.display()))?;
    restrict(&path)
}

#[cfg(unix)]
fn restrict(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(not(unix))]
fn restrict(_path: &Path) -> Result<(), String> {
    Ok(())
}

/// Resolve a provider's token: environment first so nothing has to be
/// written down, then the credentials file.
pub fn token_for(provider: &str, creds: &Credentials) -> Option<String> {
    let var = format!("SCONE_{}_TOKEN", provider.to_uppercase().replace('-', "_"));
    if let Some(v) = std::env::var_os(&var) {
        let v = v.to_string_lossy().trim().to_owned();
        if !v.is_empty() {
            return Some(v);
        }
    }
    creds
        .providers
        .get(provider)
        .map(|p| p.token.clone())
        .filter(|t| !t.is_empty())
}

pub fn connector_for(provider: &str, token: String) -> Result<Box<dyn Connector>, String> {
    match provider {
        "notion" => Ok(Box::new(Notion { token })),
        "github" => Ok(Box::new(GitHub { token })),
        "slack" => Ok(Box::new(Slack { token })),
        "google-drive" => Ok(Box::new(GoogleDrive { token })),
        other => Err(format!(
            "unknown connector {other:?}: known connectors are {}",
            KNOWN.join(", ")
        )),
    }
}

pub const KNOWN: &[&str] = &["notion", "github", "slack", "google-drive"];

/// Convert Unix epoch seconds to RFC3339 UTC. Slack dates everything in
/// epoch seconds, and memory is only as useful as its timeline, so this
/// has to be right rather than approximate. Civil-from-days after
/// Howard Hinnant's algorithm.
pub fn rfc3339_from_epoch(secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (h, mi, sec) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{sec:02}Z")
}

/// Best-effort epoch seconds from an RFC3339 date, for services that
/// want a numeric cursor. Only the date part is required to be exact;
/// a slightly early cursor re-reads, which dedup absorbs.
pub fn epoch_from_rfc3339(ts: &str) -> Option<i64> {
    let (date, time) = ts.split_once('T')?;
    let mut d = date.split('-');
    let (y, m, day): (i64, i64, i64) = (
        d.next()?.parse().ok()?,
        d.next()?.parse().ok()?,
        d.next()?.parse().ok()?,
    );
    let mut t = time.trim_end_matches('Z').split(':');
    let h: i64 = t.next().unwrap_or("0").parse().unwrap_or(0);
    let mi: i64 = t.next().unwrap_or("0").parse().unwrap_or(0);
    let sec: i64 = t
        .next()
        .unwrap_or("0")
        .split('.')
        .next()
        .unwrap_or("0")
        .parse()
        .unwrap_or(0);
    let y_adj = if m <= 2 { y - 1 } else { y };
    let era = if y_adj >= 0 { y_adj } else { y_adj - 399 } / 400;
    let yoe = y_adj - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(days * 86_400 + h * 3600 + mi * 60 + sec)
}

/// Slack, via a bot token. One document per message, so a re-sync
/// dedups per message instead of rewriting a whole channel.
pub struct Slack {
    pub token: String,
}

impl Connector for Slack {
    fn name(&self) -> &'static str {
        "slack"
    }

    fn fetch(&self, since: Option<&str>) -> Result<Vec<Document>, String> {
        let channels = self.get(
            "https://slack.com/api/conversations.list\
             ?types=public_channel,private_channel&limit=200&exclude_archived=true",
        )?;
        let mut out = Vec::new();
        for (id, name) in parse_channels(&channels) {
            let mut url =
                format!("https://slack.com/api/conversations.history?channel={id}&limit=200");
            if let Some(oldest) = since.and_then(epoch_from_rfc3339) {
                url.push_str(&format!("&oldest={oldest}"));
            }
            // One unreadable channel must not sink the whole sync.
            let Ok(history) = self.get(&url) else {
                continue;
            };
            out.extend(parse_messages(&history, &id, &name));
        }
        Ok(out)
    }
}

impl Slack {
    fn get(&self, url: &str) -> Result<serde_json::Value, String> {
        let mut res = ureq::get(url)
            .header("authorization", format!("Bearer {}", self.token))
            .config()
            .timeout_global(Some(TIMEOUT))
            .build()
            .call()
            .map_err(|e| format!("slack: {e}"))?;
        let value = read_json(res.body_mut().as_reader())?;
        // Slack answers 200 with ok:false; treat that as the error it is.
        if value["ok"] == serde_json::Value::Bool(false) {
            return Err(format!(
                "slack: {}",
                value["error"].as_str().unwrap_or("request refused")
            ));
        }
        Ok(value)
    }
}

/// Channel id and name pairs from a conversations.list payload.
pub fn parse_channels(value: &serde_json::Value) -> Vec<(String, String)> {
    value["channels"]
        .as_array()
        .map(|cs| {
            cs.iter()
                .filter_map(|c| {
                    Some((
                        c["id"].as_str()?.to_owned(),
                        c["name"].as_str().unwrap_or("channel").to_owned(),
                    ))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Messages from a conversations.history payload, one document each.
/// Joins and leaves carry a subtype and are skipped as noise.
pub fn parse_messages(value: &serde_json::Value, channel_id: &str, channel: &str) -> Vec<Document> {
    let Some(messages) = value["messages"].as_array() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for m in messages {
        if m["subtype"].is_string() {
            continue;
        }
        let (Some(ts), Some(text)) = (m["ts"].as_str(), m["text"].as_str()) else {
            continue;
        };
        if text.trim().is_empty() {
            continue;
        }
        let who = m["user"]
            .as_str()
            .or(m["bot_id"].as_str())
            .unwrap_or("someone");
        let secs = ts.split('.').next().and_then(|s| s.parse::<i64>().ok());
        out.push(Document {
            id: format!("slack:{channel_id}:{ts}"),
            title: format!("#{channel}"),
            url: format!("slack://channel/{channel_id}"),
            body: format!("#{channel} {who}: {text}"),
            updated_at: secs.map(rfc3339_from_epoch),
        });
    }
    out
}

/// Google Drive documents, via an OAuth access token. Docs are exported
/// as plain text; binary files are left alone.
pub struct GoogleDrive {
    pub token: String,
}

impl Connector for GoogleDrive {
    fn name(&self) -> &'static str {
        "google-drive"
    }

    fn fetch(&self, since: Option<&str>) -> Result<Vec<Document>, String> {
        let mut q =
            String::from("mimeType='application/vnd.google-apps.document' and trashed=false");
        if let Some(since) = since {
            q.push_str(&format!(" and modifiedTime > '{since}'"));
        }
        let url = format!(
            "https://www.googleapis.com/drive/v3/files\
             ?q={}&fields=files(id,name,modifiedTime,webViewLink)&pageSize=100\
             &orderBy=modifiedTime%20desc",
            urlencode(&q)
        );
        let listing = self.get(&url)?;
        let mut docs = parse_drive_files(&listing);
        for doc in &mut docs {
            let export = format!(
                "https://www.googleapis.com/drive/v3/files/{}/export?mimeType=text/plain",
                doc.id
            );
            doc.body = self.get_text(&export).unwrap_or_default();
        }
        docs.retain(|d| !d.body.trim().is_empty());
        Ok(docs)
    }
}

impl GoogleDrive {
    fn get(&self, url: &str) -> Result<serde_json::Value, String> {
        let mut res = ureq::get(url)
            .header("authorization", format!("Bearer {}", self.token))
            .config()
            .timeout_global(Some(TIMEOUT))
            .build()
            .call()
            .map_err(|e| format!("google-drive: {e}"))?;
        read_json(res.body_mut().as_reader())
    }

    fn get_text(&self, url: &str) -> Result<String, String> {
        let mut res = ureq::get(url)
            .header("authorization", format!("Bearer {}", self.token))
            .config()
            .timeout_global(Some(TIMEOUT))
            .build()
            .call()
            .map_err(|e| format!("google-drive: {e}"))?;
        let mut buf = String::new();
        res.body_mut()
            .as_reader()
            .take(MAX_BODY_BYTES)
            .read_to_string(&mut buf)
            .map_err(|e| e.to_string())?;
        Ok(buf)
    }
}

/// Percent-encode a Drive query. Only the characters that actually
/// appear in these queries need handling.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// File metadata from a Drive files.list payload.
pub fn parse_drive_files(value: &serde_json::Value) -> Vec<Document> {
    value["files"]
        .as_array()
        .map(|files| {
            files
                .iter()
                .filter_map(|f| {
                    let id = f["id"].as_str()?;
                    Some(Document {
                        id: id.to_owned(),
                        title: f["name"].as_str().unwrap_or("untitled").to_owned(),
                        url: f["webViewLink"].as_str().unwrap_or_default().to_owned(),
                        body: String::new(),
                        updated_at: f["modifiedTime"].as_str().map(str::to_owned),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// GitHub issues and pull requests across everything the token can see.
/// A personal access token is enough; no OAuth app to register.
pub struct GitHub {
    pub token: String,
}

impl Connector for GitHub {
    fn name(&self) -> &'static str {
        "github"
    }

    fn fetch(&self, since: Option<&str>) -> Result<Vec<Document>, String> {
        let mut url = String::from(
            "https://api.github.com/issues?filter=all&state=all\
             &sort=updated&direction=desc&per_page=100",
        );
        if let Some(since) = since {
            url.push_str(&format!("&since={since}"));
        }
        let mut res = ureq::get(&url)
            .header("authorization", format!("Bearer {}", self.token))
            .header("accept", "application/vnd.github+json")
            .header("x-github-api-version", "2022-11-28")
            // GitHub rejects requests without one.
            .header("user-agent", "scone")
            .config()
            .timeout_global(Some(TIMEOUT))
            .build()
            .call()
            .map_err(|e| format!("github: {e}"))?;
        let value = read_json(res.body_mut().as_reader())?;
        Ok(parse_issues(&value))
    }
}

/// Turn a GitHub issues payload into documents. Pure, so the shape of
/// their API is tested without touching the network.
pub fn parse_issues(value: &serde_json::Value) -> Vec<Document> {
    let Some(items) = value.as_array() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for item in items {
        let Some(number) = item["number"].as_u64() else {
            continue;
        };
        let repo = item["repository"]["full_name"].as_str().unwrap_or("");
        let title = item["title"].as_str().unwrap_or("untitled");
        let kind = if item["pull_request"].is_object() {
            "pull request"
        } else {
            "issue"
        };
        let state = item["state"].as_str().unwrap_or("");
        let body = item["body"].as_str().unwrap_or("");
        let header = if repo.is_empty() {
            format!("{kind} #{number} ({state}): {title}")
        } else {
            format!("{repo} {kind} #{number} ({state}): {title}")
        };
        out.push(Document {
            id: format!("github:{repo}#{number}"),
            title: header.clone(),
            url: item["html_url"].as_str().unwrap_or_default().to_owned(),
            body: if body.trim().is_empty() {
                header
            } else {
                body.to_owned()
            },
            updated_at: item["updated_at"].as_str().map(str::to_owned),
        });
    }
    out
}

/// Notion, via an internal integration token. No OAuth app to register:
/// create an integration, share the pages with it, paste the token.
pub struct Notion {
    pub token: String,
}

impl Connector for Notion {
    fn name(&self) -> &'static str {
        "notion"
    }

    fn fetch(&self, since: Option<&str>) -> Result<Vec<Document>, String> {
        let body = serde_json::json!({
            "filter": {"property": "object", "value": "page"},
            "sort": {"direction": "descending", "timestamp": "last_edited_time"},
            "page_size": 100,
        });
        let value = self.post("https://api.notion.com/v1/search", body)?;
        let mut docs = parse_search(&value, since);
        for doc in &mut docs {
            doc.body = self.page_text(&doc.id)?;
        }
        docs.retain(|d| !d.body.trim().is_empty());
        Ok(docs)
    }
}

impl Notion {
    fn post(&self, url: &str, body: serde_json::Value) -> Result<serde_json::Value, String> {
        let mut res = ureq::post(url)
            .header("authorization", format!("Bearer {}", self.token))
            .header("notion-version", NOTION_VERSION)
            .config()
            .timeout_global(Some(TIMEOUT))
            .build()
            .send_json(&body)
            .map_err(|e| format!("notion: {e}"))?;
        read_json(res.body_mut().as_reader())
    }

    fn get(&self, url: &str) -> Result<serde_json::Value, String> {
        let mut res = ureq::get(url)
            .header("authorization", format!("Bearer {}", self.token))
            .header("notion-version", NOTION_VERSION)
            .config()
            .timeout_global(Some(TIMEOUT))
            .build()
            .call()
            .map_err(|e| format!("notion: {e}"))?;
        read_json(res.body_mut().as_reader())
    }

    fn page_text(&self, page_id: &str) -> Result<String, String> {
        let url = format!("https://api.notion.com/v1/blocks/{page_id}/children?page_size=100");
        let value = self.get(&url)?;
        Ok(blocks_to_text(&value))
    }
}

fn read_json(reader: impl Read) -> Result<serde_json::Value, String> {
    let mut buf = String::new();
    reader
        .take(MAX_BODY_BYTES)
        .read_to_string(&mut buf)
        .map_err(|e| e.to_string())?;
    serde_json::from_str(&buf).map_err(|e| format!("unreadable response: {e}"))
}

/// Pull documents out of a Notion search payload. Pure, so the shape of
/// their API is tested without touching the network.
pub fn parse_search(value: &serde_json::Value, since: Option<&str>) -> Vec<Document> {
    let mut out = Vec::new();
    let Some(results) = value["results"].as_array() else {
        return out;
    };
    for page in results {
        let Some(id) = page["id"].as_str() else {
            continue;
        };
        let edited = page["last_edited_time"].as_str().map(str::to_owned);
        // Incremental: the source's own clock decides, not ours.
        let already_seen =
            matches!((since, edited.as_deref()), (Some(cursor), Some(at)) if at <= cursor);
        if already_seen {
            continue;
        }
        out.push(Document {
            id: id.to_owned(),
            title: page_title(page),
            url: page["url"].as_str().unwrap_or_default().to_owned(),
            body: String::new(),
            updated_at: edited,
        });
    }
    out
}

/// Notion hides the title under whichever property happens to be of
/// type `title`, which is not always called "title".
fn page_title(page: &serde_json::Value) -> String {
    if let Some(props) = page["properties"].as_object() {
        for value in props.values() {
            if value["type"] == "title" {
                let text = rich_text(&value["title"]);
                if !text.trim().is_empty() {
                    return text;
                }
            }
        }
    }
    "untitled".to_owned()
}

/// Flatten a Notion block payload into plain text. Unsupported block
/// types contribute nothing rather than failing the page.
pub fn blocks_to_text(value: &serde_json::Value) -> String {
    let Some(results) = value["results"].as_array() else {
        return String::new();
    };
    let mut lines = Vec::new();
    for block in results {
        let Some(kind) = block["type"].as_str() else {
            continue;
        };
        let text = rich_text(&block[kind]["rich_text"]);
        if text.trim().is_empty() {
            continue;
        }
        lines.push(match kind {
            "heading_1" => format!("# {text}"),
            "heading_2" => format!("## {text}"),
            "heading_3" => format!("### {text}"),
            "bulleted_list_item" | "numbered_list_item" => format!("- {text}"),
            "to_do" => {
                let done = block[kind]["checked"].as_bool().unwrap_or(false);
                format!("- [{}] {text}", if done { "x" } else { " " })
            }
            "code" => format!("    {text}"),
            "quote" => format!("> {text}"),
            _ => text,
        });
    }
    lines.join("\n")
}

fn rich_text(value: &serde_json::Value) -> String {
    value
        .as_array()
        .map(|parts| {
            parts
                .iter()
                .filter_map(|p| p["plain_text"].as_str())
                .collect::<String>()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn search_payload() -> serde_json::Value {
        serde_json::json!({"results": [
            {
                "id": "abc",
                "url": "https://notion.so/abc",
                "last_edited_time": "2026-09-01T10:00:00.000Z",
                "properties": {"Name": {"type": "title", "title": [{"plain_text": "Design notes"}]}}
            },
            {
                "id": "old",
                "url": "https://notion.so/old",
                "last_edited_time": "2026-08-01T10:00:00.000Z",
                "properties": {"Name": {"type": "title", "title": [{"plain_text": "Stale"}]}}
            }
        ]})
    }

    #[test]
    fn search_yields_titled_documents() {
        let docs = parse_search(&search_payload(), None);
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].title, "Design notes");
        assert_eq!(docs[0].url, "https://notion.so/abc");
        assert_eq!(
            docs[0].updated_at.as_deref(),
            Some("2026-09-01T10:00:00.000Z")
        );
    }

    /// A resync must not re-fetch what has not changed; the source's own
    /// edit time decides, so a slow sync cannot skip a page.
    #[test]
    fn search_honors_the_incremental_cursor() {
        let docs = parse_search(&search_payload(), Some("2026-08-15T00:00:00.000Z"));
        assert_eq!(docs.len(), 1, "only the page edited after the cursor");
        assert_eq!(docs[0].id, "abc");
    }

    #[test]
    fn untitled_pages_still_come_through() {
        let payload = serde_json::json!({"results": [
            {"id": "x", "url": "u", "last_edited_time": "2026-09-01T00:00:00.000Z",
             "properties": {}}
        ]});
        assert_eq!(parse_search(&payload, None)[0].title, "untitled");
    }

    #[test]
    fn blocks_flatten_to_readable_text() {
        let payload = serde_json::json!({"results": [
            {"type": "heading_1", "heading_1": {"rich_text": [{"plain_text": "Title"}]}},
            {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "Body text"}]}},
            {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"plain_text": "point"}]}},
            {"type": "to_do", "to_do": {"rich_text": [{"plain_text": "ship it"}], "checked": true}},
            {"type": "paragraph", "paragraph": {"rich_text": []}},
            {"type": "unsupported_block", "unsupported_block": {}}
        ]});
        let text = blocks_to_text(&payload);
        assert_eq!(text, "# Title\nBody text\n- point\n- [x] ship it");
    }

    #[test]
    fn missing_fields_never_panic() {
        assert!(parse_search(&serde_json::json!({}), None).is_empty());
        assert_eq!(blocks_to_text(&serde_json::json!({})), "");
    }

    #[test]
    fn environment_token_wins_over_the_file() {
        let mut creds = Credentials::default();
        creds.providers.insert(
            "notion".into(),
            Provider {
                token: "from-file".into(),
                last_sync: None,
            },
        );
        assert_eq!(token_for("notion", &creds).as_deref(), Some("from-file"));
        // SAFETY: single-threaded test, no other reader of this var.
        unsafe { std::env::set_var("SCONE_NOTION_TOKEN", "from-env") };
        assert_eq!(token_for("notion", &creds).as_deref(), Some("from-env"));
        unsafe { std::env::remove_var("SCONE_NOTION_TOKEN") };
    }

    #[test]
    fn credentials_round_trip_and_stay_private() {
        let dir = tempfile::tempdir().unwrap();
        let mut creds = Credentials::default();
        creds.providers.insert(
            "notion".into(),
            Provider {
                token: "secret".into(),
                last_sync: Some("2026-09-01T00:00:00Z".into()),
            },
        );
        save_credentials(dir.path(), &creds).unwrap();
        let back = load_credentials(dir.path()).unwrap();
        assert_eq!(back.providers["notion"].token, "secret");
        assert_eq!(
            back.providers["notion"].last_sync.as_deref(),
            Some("2026-09-01T00:00:00Z")
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(credentials_path(dir.path()))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600, "a token file must not be readable");
        }
    }

    #[test]
    fn github_issues_become_documents() {
        let payload = serde_json::json!([
            {
                "number": 12,
                "title": "Recall drops timestamps",
                "body": "The reader cannot order events without them.",
                "state": "open",
                "html_url": "https://github.com/o/r/issues/12",
                "updated_at": "2026-09-02T10:00:00Z",
                "repository": {"full_name": "o/r"}
            },
            {
                "number": 13,
                "title": "Add the tap",
                "body": "",
                "state": "closed",
                "html_url": "https://github.com/o/r/pull/13",
                "updated_at": "2026-09-01T10:00:00Z",
                "repository": {"full_name": "o/r"},
                "pull_request": {"url": "https://api.github.com/repos/o/r/pulls/13"}
            }
        ]);
        let docs = parse_issues(&payload);
        assert_eq!(docs.len(), 2);
        assert_eq!(
            docs[0].title,
            "o/r issue #12 (open): Recall drops timestamps"
        );
        assert_eq!(docs[0].id, "github:o/r#12");
        assert!(docs[0].body.contains("order events"));
        // A pull request must not be filed as an issue.
        assert!(
            docs[1].title.contains("pull request #13 (closed)"),
            "{}",
            docs[1].title
        );
        // An empty body still carries the header, so the memory is not blank.
        assert!(docs[1].body.contains("Add the tap"));
    }

    #[test]
    fn github_payload_garbage_never_panics() {
        assert!(parse_issues(&serde_json::json!({})).is_empty());
        assert!(parse_issues(&serde_json::json!([{"no_number": 1}])).is_empty());
        let no_repo = serde_json::json!([{"number": 1, "title": "t", "state": "open"}]);
        assert_eq!(parse_issues(&no_repo)[0].title, "issue #1 (open): t");
    }

    /// Known epochs, including a leap day and the boundary that broke
    /// every naive implementation. A wrong clock here silently misdates
    /// memory, which is worse than failing to store it.
    #[test]
    fn epoch_converts_to_the_right_calendar_day() {
        assert_eq!(rfc3339_from_epoch(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339_from_epoch(1), "1970-01-01T00:00:01Z");
        assert_eq!(rfc3339_from_epoch(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(rfc3339_from_epoch(1_009_843_199), "2001-12-31T23:59:59Z");
        assert_eq!(rfc3339_from_epoch(1_756_800_000), "2025-09-02T08:00:00Z");
        // Before the epoch must not wrap into the future.
        assert_eq!(rfc3339_from_epoch(-1), "1969-12-31T23:59:59Z");
    }

    #[test]
    fn epoch_round_trips_through_rfc3339() {
        for secs in [0_i64, 951_782_400, 1_009_843_199, 1_756_800_000] {
            let text = rfc3339_from_epoch(secs);
            assert_eq!(
                epoch_from_rfc3339(&text),
                Some(secs),
                "round trip failed for {text}"
            );
        }
        assert_eq!(epoch_from_rfc3339("not a date"), None);
    }

    #[test]
    fn slack_messages_become_one_document_each() {
        let channels = serde_json::json!({"channels": [
            {"id": "C1", "name": "engineering"},
            {"id": "C2"}
        ]});
        assert_eq!(
            parse_channels(&channels),
            vec![
                ("C1".to_owned(), "engineering".to_owned()),
                ("C2".to_owned(), "channel".to_owned())
            ]
        );

        let history = serde_json::json!({"messages": [
            {"ts": "1756800000.000100", "user": "U7", "text": "shipping the tap today"},
            {"ts": "1756800100.000200", "subtype": "channel_join", "user": "U8", "text": "joined"},
            {"ts": "1756800200.000300", "user": "U9", "text": "   "},
            {"ts": "1756800300.000400", "bot_id": "B1", "text": "build passed"}
        ]});
        let docs = parse_messages(&history, "C1", "engineering");
        assert_eq!(docs.len(), 2, "joins and blank messages are noise");
        assert_eq!(docs[0].id, "slack:C1:1756800000.000100");
        assert_eq!(docs[0].body, "#engineering U7: shipping the tap today");
        assert_eq!(docs[0].updated_at.as_deref(), Some("2025-09-02T08:00:00Z"));
        assert!(
            docs[1].body.contains("B1: build passed"),
            "bots are people too"
        );
    }

    #[test]
    fn drive_files_carry_their_modified_time() {
        let payload = serde_json::json!({"files": [
            {"id": "f1", "name": "Design doc", "modifiedTime": "2026-09-02T10:00:00.000Z",
             "webViewLink": "https://docs.google.com/document/d/f1"},
            {"name": "no id"}
        ]});
        let docs = parse_drive_files(&payload);
        assert_eq!(docs.len(), 1, "a file without an id cannot be exported");
        assert_eq!(docs[0].title, "Design doc");
        assert_eq!(
            docs[0].updated_at.as_deref(),
            Some("2026-09-02T10:00:00.000Z")
        );
    }

    #[test]
    fn drive_query_encoding_survives_quotes_and_spaces() {
        let encoded = urlencode("mimeType='x' and trashed=false");
        assert!(!encoded.contains(' '), "{encoded}");
        assert!(!encoded.contains('\''), "{encoded}");
        assert!(encoded.contains("mimeType"), "{encoded}");
    }

    #[test]
    fn slack_and_drive_payload_garbage_never_panics() {
        assert!(parse_channels(&serde_json::json!({})).is_empty());
        assert!(parse_messages(&serde_json::json!({}), "C", "c").is_empty());
        assert!(parse_drive_files(&serde_json::json!({})).is_empty());
    }

    #[test]
    fn unknown_connector_is_named_in_the_error() {
        let err = match connector_for("dropbox", "t".into()) {
            Err(e) => e,
            Ok(c) => panic!("dropbox should not resolve, got {}", c.name()),
        };
        assert!(err.contains("dropbox") && err.contains("notion"), "{err}");
    }
}
