"""Build a tuning set from the Lichess evaluation database.

https://database.lichess.org/#evals is 394 million positions annotated by Stockfish,
released under CC0. The competition rules permit this explicitly and in two places:
"Training data is unrestricted, including positions annotated by an existing engine",
and "the ban covers what ships and runs inside the zip". Nothing from it is shipped. It
is used only to choose the numbers in nx_params.py, and what ships is our own evaluation
function with our own weights.

Why it is worth doing: self-play labels a position with the result of one game between
two copies of a middling engine, which is a very noisy signal. This labels it with a
depth-37 search. The same tuner, pointed at better labels, finds better weights.

The file is a 21 GB zstandard stream, so this reads it incrementally and stops once it
has enough. Output is written in the same shape scripts/selfplay.py produces, so
scripts/tune.py needs no changes at all.

    python scripts/fetch_evals.py --positions 5000000 --out-dir data
"""

import argparse
import io
import json
import time
import urllib.request
from pathlib import Path

import chess
import numpy as np
import zstandard as zstd

from engine_api import core

URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"


def quiet_best_move(board, uci):
    """True if the engine's best move is neither a capture nor a promotion.

    A static evaluation is being fitted, so it should be fitted on positions where a
    static evaluation is the right tool. If the best move wins material, the score
    describes the tactic and not the position.

    Lines are written in the Chess960 convention, where castling is king-takes-rook, so
    a friendly piece on the destination square means castling rather than a capture.
    """
    if len(uci) > 4:
        return False
    try:
        source = chess.SQUARE_NAMES.index(uci[:2])
        target = chess.SQUARE_NAMES.index(uci[2:4])
    except ValueError:
        return False
    victim = board.piece_at(target)
    if victim is not None:
        return victim.color == board.turn
    mover = board.piece_at(source)
    return not (mover is not None and mover.piece_type == chess.PAWN
                and target == board.ep_square)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=4_000_000)
    parser.add_argument("--min-depth", type=int, default=12)
    parser.add_argument("--cap", type=int, default=1500, help="drop |cp| above this")
    parser.add_argument("--shard", type=int, default=500_000)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--prefix", default="ev")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bb = np.zeros(15, dtype=np.int64)
    sq = np.full(64, core.NO_PIECE, dtype=np.int8)
    st = np.zeros(8, dtype=np.int64)
    core.set_from_board(bb, sq, st, chess.Board())     # pay the numba compile once

    bbs, sqs, sts, tgt, cps = [], [], [], [], []
    shard = 0
    seen = kept = mates = shallow = noisy = 0
    t0 = time.time()

    raw = urllib.request.urlopen(URL, timeout=120)
    reader = zstd.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(raw)
    try:
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            seen += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            best = max(row["evals"], key=lambda e: e.get("depth", 0))
            if best.get("depth", 0) < args.min_depth:
                shallow += 1
                continue
            pv = best["pvs"][0]
            cp = pv.get("cp")
            if cp is None:
                mates += 1
                continue
            if abs(cp) > args.cap:
                continue
            try:
                board = chess.Board(row["fen"])
            except ValueError:
                continue
            if board.is_check() or not quiet_best_move(board, pv.get("line", "").split(" ")[0]):
                noisy += 1
                continue

            core.set_from_board(bb, sq, st, board)
            bbs.append(bb.copy())
            sqs.append(sq.copy())
            sts.append(st.copy())
            # The label is a winning chance, not a centipawn count, so that a +1500
            # position does not pull on the weights fifteen times as hard as a +100 one.
            tgt.append(1.0 / (1.0 + 10.0 ** (-cp / 400.0)))
            cps.append(cp)
            kept += 1

            if len(tgt) >= args.shard:
                write(args, shard, bbs, sqs, sts, tgt, cps)
                shard += 1
                bbs, sqs, sts, tgt, cps = [], [], [], [], []
            if kept >= args.positions:
                break
            if seen % 500_000 == 0:
                rate = kept / max(1, seen)
                print(f"  read {seen:,}  kept {kept:,} ({100 * rate:.0f}%)  "
                      f"{time.time() - t0:.0f}s", flush=True)
    except (OSError, zstd.ZstdError) as error:
        print(f"stream ended early ({error!r}); keeping what we have")

    if tgt:
        write(args, shard, bbs, sqs, sts, tgt, cps)
    print(f"\nread {seen:,} lines, kept {kept:,}")
    print(f"  skipped: {mates:,} mate scores, {shallow:,} too shallow, "
          f"{noisy:,} in check or tactical")
    print(f"  {time.time() - t0:.0f}s total")


def write(args, shard, bbs, sqs, sts, tgt, cps):
    path = args.out_dir / f"{args.prefix}{shard:03d}.npz"
    np.savez_compressed(
        path,
        bb=np.array(bbs, dtype=np.int64), sq=np.array(sqs, dtype=np.int8),
        st=np.array(sts, dtype=np.int64), result=np.array(tgt, dtype=np.float32),
        cp=np.array(cps, dtype=np.int32))
    print(f"  wrote {path} ({len(tgt):,} positions)", flush=True)


if __name__ == "__main__":
    main()
