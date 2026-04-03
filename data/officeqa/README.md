# Local OfficeQA Corpus

This directory is the canonical local landing zone for the untracked OfficeQA dataset.

## Quick Setup

For local testing, clone the official OfficeQA dataset repo into this folder:

```powershell
git clone https://github.com/databricks/officeqa.git data/officeqa/source
```

After cloning, you have two supported local corpus choices:

1. Recommended if you already have parsed artifacts:
   - use `data/officeqa/treasury_bulletins_parsed/`
2. Works with the official repo directly:
   - use `data/officeqa/source/treasury_bulletin_pdfs/`

If you only want the official corpus quickly, use the PDF path. The runtime can index raw PDFs, JSON, CSV, TSV, and text files.

Recommended layout:

```text
data/
  officeqa/
    README.md
    source/
      treasury_bulletin_pdfs/
      officeqa.csv
    treasury_bulletins_parsed/
    treasury_bulletin_pdfs/
    officeqa.csv
```

## Local Testing Path

If you cloned the official repo exactly as above, set:

```powershell
$env:OFFICEQA_CORPUS_DIR="data/officeqa/source/treasury_bulletin_pdfs"
```

If you already have parsed files locally, set:

```powershell
$env:OFFICEQA_CORPUS_DIR="data/officeqa/treasury_bulletins_parsed"
```

## Build And Verify

Build the local index with:

```powershell
uv run python scripts/build_officeqa_index.py --corpus-root "$env:OFFICEQA_CORPUS_DIR"
```

Verify the local bundle with:

```powershell
uv run python scripts/verify_officeqa_corpus.py --corpus-root "$env:OFFICEQA_CORPUS_DIR"
```

Then run the local smoke test:

```powershell
$env:BENCHMARK_NAME="officeqa"
$env:BENCHMARK_STATELESS="1"
uv run python scripts/run_officeqa_regression.py --smoke
```

Do not commit corpus files here. `.gitignore` is configured to keep this README tracked while leaving the dataset untracked.
