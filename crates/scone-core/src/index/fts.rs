//! Full-text index over chunks (tantivy, BM25), scoped by space.
//!
//! Derived data only: rebuildable from SQLite at any time. The predecessor
//! declared a weighted FTS index in its schema and never queried it
//! (memory/rationales.md R-9); this one is exercised by tests from birth.

use std::path::Path;

use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, Occur, Query, QueryParser, TermQuery};
use tantivy::schema::{Field, IndexRecordOption, STORED, Schema, TantivyDocument, Value};
use tantivy::{Index, IndexReader, IndexWriter, Term, doc};

use crate::error::{Result, SconeError};

const WRITER_HEAP: usize = 50_000_000;

pub struct FtsIndex {
    index: Index,
    writer: IndexWriter,
    reader: IndexReader,
    f_chunk: Field,
    f_space: Field,
    f_text: Field,
}

fn ix(e: impl std::fmt::Display) -> SconeError {
    SconeError::Index(format!("fts: {e}"))
}

impl FtsIndex {
    pub fn open(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let mut schema = Schema::builder();
        let f_chunk = schema.add_u64_field("chunk_id", tantivy::schema::INDEXED | STORED);
        let f_space = schema.add_u64_field("space_id", tantivy::schema::INDEXED);
        let f_text = schema.add_text_field("text", tantivy::schema::TEXT);
        let schema = schema.build();
        let mmap = tantivy::directory::MmapDirectory::open(dir).map_err(ix)?;
        let index = Index::open_or_create(mmap, schema).map_err(ix)?;
        let writer = index.writer(WRITER_HEAP).map_err(ix)?;
        let reader = index.reader().map_err(ix)?;
        Ok(Self {
            index,
            writer,
            reader,
            f_chunk,
            f_space,
            f_text,
        })
    }

    /// Add `(chunk_id, space_id, text)` rows and commit.
    pub fn add(&mut self, rows: &[(u64, u64, &str)]) -> Result<()> {
        for (chunk_id, space_id, text) in rows {
            self.writer
                .add_document(doc!(
                    self.f_chunk => *chunk_id,
                    self.f_space => *space_id,
                    self.f_text => *text,
                ))
                .map_err(ix)?;
        }
        self.writer.commit().map_err(ix)?;
        self.reader.reload().map_err(ix)?;
        Ok(())
    }

    /// BM25 top-`k` chunk ids within one space.
    pub fn search(&self, space_id: u64, query: &str, k: usize) -> Result<Vec<(u64, f32)>> {
        let parser = QueryParser::for_index(&self.index, vec![self.f_text]);
        let text_query = parser
            .parse_query(query)
            .map_err(|e| SconeError::InvalidInput(format!("query: {e}")))?;
        let space_query = TermQuery::new(
            Term::from_field_u64(self.f_space, space_id),
            IndexRecordOption::Basic,
        );
        let combined = BooleanQuery::new(vec![
            (Occur::Must, Box::new(space_query) as Box<dyn Query>),
            (Occur::Must, text_query),
        ]);
        let searcher = self.reader.searcher();
        let top = searcher
            .search(&combined, &TopDocs::with_limit(k.max(1)).order_by_score())
            .map_err(ix)?;
        let mut hits = Vec::with_capacity(top.len());
        for (score, addr) in top {
            let stored: TantivyDocument = searcher.doc(addr).map_err(ix)?;
            let id = stored
                .get_first(self.f_chunk)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| ix("document missing chunk_id"))?;
            hits.push((id, score));
        }
        Ok(hits)
    }

    pub fn wipe(&mut self) -> Result<()> {
        self.writer.delete_all_documents().map_err(ix)?;
        self.writer.commit().map_err(ix)?;
        self.reader.reload().map_err(ix)?;
        Ok(())
    }
}
