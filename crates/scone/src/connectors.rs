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
        other => Err(format!(
            "unknown connector {other:?}: known connectors are {}",
            KNOWN.join(", ")
        )),
    }
}

pub const KNOWN: &[&str] = &["notion"];

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
        if let (Some(since), Some(edited)) = (since, edited.as_deref()) {
            if edited <= since {
                continue;
            }
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
    fn unknown_connector_is_named_in_the_error() {
        let err = match connector_for("dropbox", "t".into()) {
            Err(e) => e,
            Ok(c) => panic!("dropbox should not resolve, got {}", c.name()),
        };
        assert!(err.contains("dropbox") && err.contains("notion"), "{err}");
    }
}
