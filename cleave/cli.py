"""Command line: ``cleave chunk | serve | eval | eval-retrieval``.

cleave chunk report.pdf data.xlsx -o units.jsonl
cleave serve --host 127.0.0.1 --port 8321
cleave eval-retrieval benchmarks/qa_fixtures.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_chunk(args: argparse.Namespace) -> int:
    from .api import chunk_file_detailed

    out = args.output.open("w") if args.output else sys.stdout
    failed = 0
    try:
        for path in args.files:
            try:
                res = chunk_file_detailed(path, use_llm=args.llm)
            except Exception as exc:
                failed += 1
                sys.stderr.write(f"cleave: {path}: {type(exc).__name__}: {exc}\n")
                continue
            route = res.profile.get("route_actual") or res.profile.get("route", "?")
            sys.stderr.write(f"cleave: {path}: {len(res.units)} units via {route}\n")
            for u in res.units:
                d = u.to_dict()
                d["source_file"] = str(path)
                out.write(json.dumps(d, ensure_ascii=False) + "\n")
    finally:
        if args.output:
            out.close()
    return 1 if failed == len(args.files) else 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .config import settings
    from .logging_setup import configure_logging

    if args.host not in ("127.0.0.1", "localhost", "::1") and not settings().api_key:
        sys.stderr.write(
            "cleave: refusing to listen on a non-loopback address without CLEAVE_API_KEY\n"
        )
        return 2
    configure_logging()
    uvicorn.run("cleave.app:app", host=args.host, port=args.port, reload=False)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import main as eval_main

    eval_main([str(p) for p in args.files])
    return 0


def _cmd_eval_retrieval(rest: list[str]) -> int:
    from .eval_retrieval import main as er_main

    er_main(rest)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["eval-retrieval"]:
        return _cmd_eval_retrieval(argv[1:])

    ap = argparse.ArgumentParser(
        prog="cleave", description="Understand information before you cut it."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("chunk", help="chunk files to JSONL knowledge units")
    c.add_argument("files", nargs="+", type=Path)
    c.add_argument("-o", "--output", type=Path, help="write JSONL here (default: stdout)")
    c.add_argument("--llm", action="store_true", help="allow LLM enrichment of flagged chunks")
    c.set_defaults(func=_cmd_chunk)

    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8321)
    s.set_defaults(func=_cmd_serve)

    e = sub.add_parser("eval", help="context preservation scorecard vs a fixed-size baseline")
    e.add_argument("files", nargs="+", type=Path)
    e.set_defaults(func=_cmd_eval)

    sub.add_parser("eval-retrieval", help="retrieval recall/MRR benchmark (see --help)")

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
