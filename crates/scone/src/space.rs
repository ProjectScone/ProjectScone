//! Deriving a space name from where the work lives.

use std::path::Path;

/// The space a repository shares. Two clones of the same project must
/// land on the same name, so it comes from the remote rather than from
/// the local path: colleagues check out different directories, and a
/// fork lives somewhere else entirely.
///
/// Falls back to the directory name outside a repository, and to
/// "default" when there is nothing to go on, because a memory command
/// should never fail over what to call a space.
pub fn auto_space(cwd: &Path) -> String {
    let remote = std::process::Command::new("git")
        .args([
            "-C",
            &cwd.display().to_string(),
            "remote",
            "get-url",
            "origin",
        ])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .filter(|u| !u.is_empty());
    if let Some(remote) = remote {
        return slug(&normalize_remote(&remote));
    }
    let dir = std::process::Command::new("git")
        .args([
            "-C",
            &cwd.display().to_string(),
            "rev-parse",
            "--show-toplevel",
        ])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .filter(|p| !p.is_empty())
        .unwrap_or_else(|| cwd.display().to_string());
    Path::new(&dir)
        .file_name()
        .map(|n| slug(&n.to_string_lossy()))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "default".to_owned())
}

/// Reduce a remote URL to the part that identifies the project, so ssh
/// and https clones of the same repo agree.
pub fn normalize_remote(url: &str) -> String {
    let url = url.trim().trim_end_matches('/');
    let url = url.strip_suffix(".git").unwrap_or(url);
    // git@host:owner/repo and https://host/owner/repo differ only in
    // how they spell the same thing.
    let after_host = match url.split_once("://") {
        Some((_, rest)) => rest.split_once('/').map(|(_, r)| r).unwrap_or(rest),
        None => match url.split_once(':') {
            Some((_, rest)) => rest,
            None => url,
        },
    };
    // Strip any credentials that rode along in an https remote.
    after_host
        .rsplit('@')
        .next()
        .unwrap_or(after_host)
        .to_lowercase()
}

/// A space name that is safe in a path, a URL, and a shell.
pub fn slug(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last_dash = true;
    for c in text.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    out.trim_matches('-').chars().take(64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two clones of one project must land on one space, however they
    /// were cloned, or a team silently keeps two half-brains.
    #[test]
    fn every_way_of_writing_a_remote_agrees() {
        let expected = "drdrewcain/projectscone";
        for url in [
            "git@github.com:DrDrewCain/ProjectScone.git",
            "https://github.com/DrDrewCain/ProjectScone.git",
            "https://github.com/DrDrewCain/ProjectScone",
            "ssh://git@github.com/DrDrewCain/ProjectScone.git",
            "https://token@github.com/DrDrewCain/ProjectScone.git",
            "git@github.com:DrDrewCain/ProjectScone/",
        ] {
            assert_eq!(normalize_remote(url), expected, "{url}");
        }
    }

    /// Different projects must not collide, including a fork whose
    /// name matches under another owner.
    #[test]
    fn different_projects_get_different_spaces() {
        assert_ne!(
            normalize_remote("git@github.com:alice/scone.git"),
            normalize_remote("git@github.com:bob/scone.git")
        );
    }

    #[test]
    fn slugs_are_safe_and_bounded() {
        assert_eq!(slug("DrDrewCain/ProjectScone"), "drdrewcain-projectscone");
        assert_eq!(slug("a  b//c"), "a-b-c");
        assert_eq!(slug("---"), "");
        assert!(slug(&"x".repeat(200)).len() <= 64);
        // Nothing that could steer a path or a shell survives.
        assert_eq!(slug("../../etc/passwd"), "etc-passwd");
    }
}
