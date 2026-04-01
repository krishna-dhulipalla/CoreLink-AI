"""Verify that the OfficeQA corpus bundle is usable by the runtime."""

from __future__ import annotations

import argparse
import json

from agent.benchmarks.officeqa_runtime import OfficeQACorpusBootstrapError, verify_officeqa_corpus_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the OfficeQA corpus bundle and index.")
    parser.add_argument("--corpus-root", default="", help="Path to the packaged or mounted OfficeQA corpus root.")
    parser.add_argument("--index-dir", default="", help="Optional path to the OfficeQA index directory.")
    args = parser.parse_args()

    try:
        summary = verify_officeqa_corpus_bundle(
            corpus_root=args.corpus_root or None,
            index_dir=args.index_dir or None,
            require_index=True,
        )
    except OfficeQACorpusBootstrapError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=True, indent=2))
        return 1

    print(json.dumps({"ok": True, **summary}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
