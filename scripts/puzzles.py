"""Measure search quality against the Lichess puzzle database.

The arena says whether one version beats another. It does not say *why*, and it costs
hundreds of games to say anything at all. This is the other half of the picture: a few
thousand puzzles with known solutions, scored by rating band, run in a couple of minutes
and tell you directly whether a change made the engine see more or less.

Puzzles are CC0, from https://database.lichess.org/#puzzles. In that format the FEN is
the position *before* the opponent's move, and the first entry in Moves is that move, so
the position to solve is FEN with the first move played, and the answer is the second.

    python scripts/puzzles.py --fetch 40000            # once, downloads and caches
    python scripts/puzzles.py --nodes 20000 --count 1500
"""

import argparse
import csv
import io
import time
import urllib.request
from pathlib import Path

import chess
import numpy as np
import zstandard as zstd

from engine_api import search, search_position, to_move, warm

import nx_params as params

URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
BANDS = ((0, 1200), (1200, 1600), (1600, 2000), (2000, 2400), (2400, 3500))


def fetch(path, limit):
    raw = urllib.request.urlopen(URL, timeout=120)
    reader = zstd.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(raw)
    text = io.TextIOWrapper(reader, encoding="utf-8")
    rows = csv.DictReader(text)
    kept = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["fen", "moves", "rating"])
        for row in rows:
            try:
                rating = int(row["Rating"])
            except (KeyError, ValueError):
                continue
            writer.writerow([row["FEN"], row["Moves"], rating])
            kept += 1
            if kept >= limit:
                break
    print(f"cached {kept:,} puzzles in {path}")


def load(path, count):
    puzzles = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            moves = row["moves"].split(" ")
            if len(moves) < 2:
                continue
            board = chess.Board(row["fen"])
            try:
                board.push_uci(moves[0])
            except (ValueError, AssertionError):
                continue
            puzzles.append((board.fen(), moves[1], int(row["rating"])))
            if len(puzzles) >= count:
                break
    return puzzles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("data/puzzles.csv"))
    parser.add_argument("--fetch", type=int, default=0, help="download this many, then exit")
    parser.add_argument("--nodes", type=int, default=20000)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--weights", type=Path, help="weight vector to test (default: built-in)")
    args = parser.parse_args()

    if args.fetch:
        fetch(args.cache, args.fetch)
        return
    if not args.cache.exists():
        raise SystemExit(f"{args.cache} missing; run with --fetch 40000 first")

    puzzles = load(args.cache, args.count)
    weights = (np.load(args.weights).astype(np.int32) if args.weights
               else params.load().astype(np.int32))
    ctx = search.make_ctx(weights)
    warm(ctx)
    print(f"{len(puzzles):,} puzzles at {args.nodes:,} nodes, "
          f"weights {args.weights or 'built-in'}", flush=True)

    solved = {b: 0 for b in BANDS}
    total = {b: 0 for b in BANDS}
    t0 = time.time()
    for i, (fen, answer, rating) in enumerate(puzzles):
        board = chess.Board(fen)
        ctx.tt[:] = 0
        uci, _, _ = search_position(ctx, board, args.nodes)
        band = next(b for b in BANDS if b[0] <= rating < b[1])
        total[band] += 1
        # A different move that mates just as fast is also a solution.
        if uci == answer:
            solved[band] += 1
        elif to_move(board, uci) is not None:
            probe = board.copy()
            probe.push_uci(uci)
            if probe.is_checkmate():
                solved[band] += 1
        if (i + 1) % 250 == 0:
            done = sum(solved.values())
            print(f"  {i + 1:>5}: {done} solved ({100 * done / (i + 1):.1f}%)  "
                  f"{time.time() - t0:.0f}s", flush=True)

    print()
    for band in BANDS:
        if total[band]:
            print(f"  {band[0]:>4}-{band[1]:<4} {solved[band]:>5}/{total[band]:<5} "
                  f"{100 * solved[band] / total[band]:5.1f}%")
    overall = sum(solved.values()) / max(1, sum(total.values()))
    print(f"  {'overall':<9} {sum(solved.values()):>5}/{sum(total.values()):<5} "
          f"{100 * overall:5.1f}%   in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
