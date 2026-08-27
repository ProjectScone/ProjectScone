/// Errors surfaced by the Scone engine.
///
/// Failures are values, never deletions or silent fallbacks
/// (memory/bugs.md P-2).
#[derive(thiserror::Error, Debug)]
pub enum SconeError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("db: {0}")]
    Db(#[from] rusqlite::Error),
    #[error("index: {0}")]
    Index(String),
    #[error("embed: {0}")]
    Embed(String),
    #[error("llm: {0}")]
    Llm(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("not found: {0}")]
    NotFound(String),
}

pub type Result<T> = std::result::Result<T, SconeError>;
