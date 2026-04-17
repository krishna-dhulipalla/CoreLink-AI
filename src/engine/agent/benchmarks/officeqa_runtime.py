"""Runtime bootstrap helpers for OfficeQA corpus access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import benchmark_name_from_env, truthy_env
from .officeqa_manifest import (
    _CORPUS_ENV_NAMES,
    load_officeqa_index_metadata,
    load_officeqa_manifest,
    officeqa_index_schema_version,
    resolve_officeqa_corpus_root,
    resolve_officeqa_index_dir,
)


class OfficeQACorpusBootstrapError(RuntimeError):
    """Raised when OfficeQA competition mode starts without a usable corpus bundle."""


def officeqa_competition_corpus_required() -> bool:
    return benchmark_name_from_env() == "officeqa" and truthy_env("COMPETITION_MODE")


def verify_officeqa_corpus_bundle(
    *,
    corpus_root: str | Path | None = None,
    index_dir: str | Path | None = None,
    require_index: bool = True,
) -> dict[str, Any]:
    root = resolve_officeqa_corpus_root(str(corpus_root) if corpus_root is not None else None)
    if root is None:
        env_tip = " or ".join(_CORPUS_ENV_NAMES)
        raise OfficeQACorpusBootstrapError(
            f"OfficeQA competition mode requires a packaged corpus, but no corpus root was found. "
            f"Please set one of the following environment variables to the valid corpus path: {env_tip}. "
            f"In containerized environments, ensure the dataset is correctly mounted to a candidate path (e.g., /data/treasury_bulletins_parsed)."
        )

    resolved_index_dir = resolve_officeqa_index_dir(
        root,
        str(index_dir) if index_dir is not None else None,
        create=False,
    )
    metadata = load_officeqa_index_metadata(root, resolved_index_dir)
    records = load_officeqa_manifest(root, resolved_index_dir)

    if require_index and not metadata:
        raise OfficeQACorpusBootstrapError(
            f"OfficeQA competition mode requires a built corpus index, but {resolved_index_dir / 'index_metadata.json'} "
            "was not found. Build the index with scripts/build_officeqa_index.py before startup."
        )
    if require_index and not records:
        raise OfficeQACorpusBootstrapError(
            f"OfficeQA competition mode requires a populated manifest, but no records were loaded from "
            f"{resolved_index_dir / 'manifest.jsonl'}. Rebuild the index before startup."
        )

    expected_version = officeqa_index_schema_version()
    actual_version = int(metadata.get("index_schema_version", 0) or 0) if metadata else 0
    if require_index and actual_version != expected_version:
        raise OfficeQACorpusBootstrapError(
            "OfficeQA corpus index metadata is missing or incompatible. "
            f"Expected schema version {expected_version}, found {actual_version or 'missing'}. "
            "Rebuild the index with scripts/build_officeqa_index.py before startup."
        )

    missing_files: list[str] = []
    for record in records:
        relative = str(record.get("relative_path", "")).strip()
        if not relative:
            missing_files.append("<missing relative_path>")
            continue
        if not (root / relative).exists():
            missing_files.append(relative)
    if missing_files:
        sample = ", ".join(missing_files[:5])
        extra = "" if len(missing_files) <= 5 else f" (+{len(missing_files) - 5} more)"
        raise OfficeQACorpusBootstrapError(
            "OfficeQA corpus index references files that are not readable from the configured corpus root: "
            f"{sample}{extra}. Verify the mounted dataset matches the generated manifest."
        )

    return {
        "corpus_root": str(root),
        "index_dir": str(resolved_index_dir),
        "document_count": len(records),
        "partial_document_count": int(metadata.get("partial_document_count", 0) or 0),
        "index_schema_version": actual_version,
    }


def verify_officeqa_competition_bootstrap() -> dict[str, Any] | None:
    if not officeqa_competition_corpus_required():
        return None
    return verify_officeqa_corpus_bundle(require_index=True)
