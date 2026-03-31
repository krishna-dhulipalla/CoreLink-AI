import json

from agent.benchmarks.officeqa_index import (
    build_officeqa_index,
    resolve_source_files_to_manifest,
    search_officeqa_corpus_index,
)
from agent.retrieval_tools import search_reference_corpus


def test_build_officeqa_index_persists_manifest_and_metadata(tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_1940.json").write_text(
        json.dumps(
            {
                "title": "Treasury Bulletin 1940",
                "page": 17,
                "section_title": "National Defense",
                "headers": ["Month", "Expenditures (million dollars)"],
                "rows": [["January", "100.0"], ["February", "101.5"]],
                "unit": "million dollars",
            }
        ),
        encoding="utf-8",
    )
    (corpus_root / "notes.txt").write_text("Reference notes for an unrelated file.", encoding="utf-8")

    summary = build_officeqa_index(corpus_root=corpus_root)

    assert summary["document_count"] == 2
    manifest_path = corpus_root / ".officeqa_index" / "manifest.jsonl"
    metadata_path = corpus_root / ".officeqa_index" / "index_metadata.json"
    assert manifest_path.exists()
    assert metadata_path.exists()
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert "national defense" in manifest_text.lower()
    assert "million dollars" in manifest_text.lower()
    assert '"years": ["1940"]' in manifest_text


def test_search_reference_corpus_uses_officeqa_index_when_available(monkeypatch, tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_1940.json").write_text(
        json.dumps(
            {
                "title": "Treasury Bulletin 1940",
                "section_title": "National Defense",
                "headers": ["Month", "Expenditures (million dollars)"],
                "rows": [["January", "100.0"], ["February", "101.5"]],
                "unit": "million dollars",
            }
        ),
        encoding="utf-8",
    )
    (corpus_root / "treasury_1953.json").write_text(
        json.dumps(
            {
                "title": "Treasury Bulletin 1953",
                "section_title": "Agriculture",
                "headers": ["Month", "Expenditures (million dollars)"],
                "rows": [["January", "50.0"]],
                "unit": "million dollars",
            }
        ),
        encoding="utf-8",
    )
    build_officeqa_index(corpus_root=corpus_root)
    monkeypatch.setenv("OFFICEQA_CORPUS_DIR", str(corpus_root))

    result = search_reference_corpus.invoke({"query": "Treasury Bulletin 1940 national defense expenditures", "top_k": 2})

    assert result["index_mode"] == "officeqa_manifest"
    assert result["results"][0]["document_id"] == "treasury_1940_json"
    assert "1940" in result["results"][0]["metadata"]["years"]


def test_resolve_source_files_to_manifest_matches_relative_and_stem_names(tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "Treasury_1940_National_Defense.json").write_text(
        json.dumps({"title": "Treasury Bulletin 1940"}),
        encoding="utf-8",
    )
    build_officeqa_index(corpus_root=corpus_root)

    matches = resolve_source_files_to_manifest(
        ["Treasury_1940_National_Defense.json", "Treasury_1940_National_Defense"],
        corpus_root=corpus_root,
    )

    assert matches[0]["matched"] is True
    assert matches[1]["matched"] is True
    assert matches[0]["document_id"] == matches[1]["document_id"]


def test_search_officeqa_corpus_index_ranks_metadata_backed_match_first(tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_1940.json").write_text(
        json.dumps(
            {
                "title": "Treasury Bulletin 1940",
                "section_title": "National Defense",
                "headers": ["Month", "Expenditures (million dollars)"],
                "rows": [["January", "100.0"]],
            }
        ),
        encoding="utf-8",
    )
    (corpus_root / "misc_1940.txt").write_text("General 1940 notes without the right section.", encoding="utf-8")
    build_officeqa_index(corpus_root=corpus_root)

    result = search_officeqa_corpus_index(
        "Treasury Bulletin 1940 national defense expenditures",
        corpus_root=corpus_root,
        top_k=2,
    )

    assert result["results"][0]["document_id"] == "treasury_1940_json"
