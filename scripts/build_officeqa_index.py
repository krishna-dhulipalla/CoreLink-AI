"""Build a persistent local index for the OfficeQA corpus."""

from __future__ import annotations

import argparse
import json

from agent.benchmarks.officeqa_index import build_officeqa_index, resolve_source_files_to_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the local OfficeQA corpus index.")
    parser.add_argument("--corpus-root", default="", help="Path to the local OfficeQA corpus root.")
    parser.add_argument("--index-dir", default="", help="Optional output directory for the generated index.")
    parser.add_argument("--max-files", type=int, default=4000, help="Maximum number of corpus files to index.")
    parser.add_argument(
        "--source-file",
        action="append",
        default=[],
        help="Optional benchmark source_file value to resolve against the built manifest. Repeatable.",
    )
    args = parser.parse_args()

    summary = build_officeqa_index(
        corpus_root=args.corpus_root or None,
        index_dir=args.index_dir or None,
        max_files=args.max_files,
    )
    if args.source_file:
        summary["source_file_matches"] = resolve_source_files_to_manifest(
            args.source_file,
            corpus_root=args.corpus_root or None,
            index_dir=args.index_dir or None,
        )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
