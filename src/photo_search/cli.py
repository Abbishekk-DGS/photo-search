"""photo-search: search your photos by what's in them.

photo-search index ~/Pictures
photo-search search "sunset over the sea with a boat"
photo-search similar ~/Pictures/IMG_2041.jpg
photo-search dupes --threshold 0.96
photo-search tag --labels "receipt,whiteboard,screenshot,pet"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .encoder import ClipEncoder
from .index import PhotoIndex

DEFAULT_INDEX = Path.home() / ".photo-search"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="photo-search",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__} · by Abbishekk DGS · abbishekkdgs.tech",
    )
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("index").add_argument("folder")
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=10)
    m = sub.add_parser("similar")
    m.add_argument("image")
    m.add_argument("-k", type=int, default=10)
    d = sub.add_parser("dupes")
    d.add_argument("--threshold", type=float, default=0.95)
    t = sub.add_parser("tag")
    t.add_argument("--labels", default=None, help="comma-separated labels")
    t.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    index = PhotoIndex(Path(args.index), ClipEncoder(args.model, args.pretrained))

    if args.command == "index":

        def progress(done: int, total: int) -> None:
            sys.stdout.write(f"\r  embedded {done}/{total}")
            sys.stdout.flush()

        stats = index.update(Path(args.folder).expanduser(), progress=progress)
        print(f"\n{stats}")
    elif args.command in ("search", "similar"):
        results = (
            index.search_text(args.query, args.k)
            if args.command == "search"
            else index.search_image(Path(args.image).expanduser(), args.k)
        )
        for r in results:
            print(f"{r.score:.3f}  {r.path}")
    elif args.command == "dupes":
        groups = index.duplicates(args.threshold)
        for n, group in enumerate(groups, 1):
            print(f"group {n} ({len(group)} photos)")
            for p in group:
                print(f"   {p}")
        print(f"{sum(len(g) - 1 for g in groups)} redundant photos in {len(groups)} groups")
    else:
        labels = [x.strip() for x in args.labels.split(",")] if args.labels else None
        tags = index.tag(labels)
        if args.json:
            print(json.dumps(tags, indent=1))
        else:
            for path, best in tags.items():
                print(f"{Path(path).name}: " + ", ".join(f"{lab} ({p:.0%})" for lab, p in best))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
