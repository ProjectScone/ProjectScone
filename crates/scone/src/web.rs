//! URL ingestion (gap-analysis P3): fetch a page, convert to clean
//! markdown locally, feed the engine. Network lives here in the CLI
//! layer; scone-core stays offline by construction.

use std::time::Duration;

const MAX_PAGE_BYTES: u64 = 5_000_000;
const FETCH_TIMEOUT: Duration = Duration::from_secs(30);

/// Fetch a URL and return (markdown_text, domain).
pub fn fetch_page(url: &str) -> Result<(String, String), String> {
    let domain = url
        .split("//")
        .nth(1)
        .and_then(|rest| rest.split(['/', ':']).next())
        .filter(|d| !d.is_empty())
        .ok_or_else(|| format!("cannot parse a host from {url:?}"))?
        .to_owned();
    let mut res = ureq::get(url)
        .config()
        .timeout_global(Some(FETCH_TIMEOUT))
        .build()
        .call()
        .map_err(|e| format!("fetch {url}: {e}"))?;
    let html = res
        .body_mut()
        .with_config()
        .limit(MAX_PAGE_BYTES)
        .read_to_string()
        .map_err(|e| format!("read {url}: {e}"))?;
    let markdown = htmd::HtmlToMarkdown::builder()
        .skip_tags(vec![
            "script", "style", "nav", "header", "footer", "aside", "noscript",
        ])
        .build()
        .convert(&html)
        .map_err(|e| format!("convert {url}: {e}"))?;
    if markdown.trim().is_empty() {
        return Err(format!("{url} produced no readable text"));
    }
    Ok((markdown, domain))
}
