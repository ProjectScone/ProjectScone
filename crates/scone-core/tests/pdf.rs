#![allow(clippy::unwrap_used)]
#![cfg(feature = "pdf")]
use scone_core::embed::HashEmbedder;
use scone_core::{Engine, IngestInput, IngestOutcome, RecallOpts, auth};

/// Build a real single-page PDF with the given text.
fn make_pdf(path: &std::path::Path, text: &str) {
    use lopdf::content::{Content, Operation};
    use lopdf::{Document, Object, Stream, dictionary};
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let font_id = doc.add_object(dictionary! {
        "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica",
    });
    let resources_id = doc.add_object(dictionary! {
        "Font" => dictionary! { "F1" => font_id },
    });
    let content = Content {
        operations: vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec!["F1".into(), 24.into()]),
            Operation::new("Td", vec![72.into(), 700.into()]),
            Operation::new("Tj", vec![Object::string_literal(text)]),
            Operation::new("ET", vec![]),
        ],
    };
    let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));
    let page_id = doc.add_object(dictionary! {
        "Type" => "Page", "Parent" => pages_id, "Contents" => content_id,
    });
    let pages = dictionary! {
        "Type" => "Pages", "Kids" => vec![page_id.into()],
        "Count" => 1, "Resources" => resources_id,
        "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages));
    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog", "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);
    doc.save(path).unwrap();
}

#[test]
fn pdf_files_ingest_as_searchable_text() {
    let dir = tempfile::tempdir().unwrap();
    let pdf = dir.path().join("paper.pdf");
    make_pdf(&pdf, "the retrieval fusion paper conclusion");
    let mut e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let out = e.ingest(&space, IngestInput::File { path: pdf }).unwrap();
    assert!(matches!(out, IngestOutcome::Ingested { .. }));
    let pack = e
        .recall(&space, "fusion paper conclusion", &RecallOpts::default())
        .unwrap();
    assert!(
        pack.items[0].text.contains("retrieval fusion"),
        "{:?}",
        pack.items[0].text
    );
    assert!(
        pack.items[0]
            .source
            .as_deref()
            .unwrap()
            .ends_with("paper.pdf")
    );
}

#[test]
fn corrupt_pdf_is_a_typed_error_not_a_crash() {
    let dir = tempfile::tempdir().unwrap();
    let bad = dir.path().join("broken.pdf");
    std::fs::write(&bad, b"%PDF-1.5 this is not really a pdf").unwrap();
    let mut e = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut e, "default", true).unwrap();
    let err = e
        .ingest(&space, IngestInput::File { path: bad })
        .unwrap_err();
    assert!(
        matches!(err, scone_core::SconeError::InvalidInput(_)),
        "{err}"
    );
}
