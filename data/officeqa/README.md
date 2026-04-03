# Local OfficeQA Corpus

This directory is the canonical local landing zone for the untracked OfficeQA dataset.

Recommended layout:

```text
data/
  officeqa/
    README.md
    treasury_bulletins_parsed/
    treasury_bulletin_pdfs/
    officeqa.csv
```

Use this path for local agent testing:

- `OFFICEQA_CORPUS_DIR=data/officeqa/treasury_bulletins_parsed`

Build the local index with:

```bash
uv run python scripts/build_officeqa_index.py --corpus-root "$OFFICEQA_CORPUS_DIR"
```

Verify the local bundle with:

```bash
uv run python scripts/verify_officeqa_corpus.py --corpus-root "$OFFICEQA_CORPUS_DIR"
```

Do not commit corpus files here. `.gitignore` is configured to keep this README tracked while leaving the dataset untracked.
