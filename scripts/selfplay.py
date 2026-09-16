"""Generate training data by playing the engine against itself.

Every position is labelled with the result of the game it occurred in, which is what a
texel tuner fits the evaluation to. No outside engine and no outside database is
involved: the data is entirely this engine's own play.

Games open from a short random walk so the set is diverse, and only quiet positions are
kept, because an evaluation is not supposed to be right in the middle of an exchange.

    python scripts/selfplay.py --games 400 --nodes 8000 --out data/sp0.npz
"""

import argparse
import random
import time
from pathlib import Path

import chess
import numpy as np

from engine_api import core, search, search_position, to_move, warm

OPENING_PLIES = 8
SKIP_PLIES = 10          # do not label the random opening itself
MAX_PLIES = 300


def random_opening(rng):
    """A short random walk that stays roughly balanced, for opening variety."""
    for _ in range(60):
        board = chess.Board()
        ok = True
        for _ in range(OPENING_PLIES):
            moves = [m for m in board.legal_moves if not board.is_capture(m)] \
                or list(board.legal_moves)
            if not moves:
                ok = False
                break
            board.push(rng.choice(moves))
        if ok and not board.is_game_over():
            return board
    return chess.Board()


def play(ctx, rng, nodes):
    """One self-play game. Returns (positions, result) with result from white's view."""
    board = random_opening(rng)
    kept = []
    result = 0.5
    for ply in range(MAX_PLIES):
        if board.is_game_over(claim_draw=True):
            outcome = board.outcome(claim_draw=True)
            result = 0.5 if outcome.winner is None else (
                1.0 if outcome.winner == chess.WHITE else 0.0)
            break
        uci, score, _ = search_position(ctx, board, nodes)
        move = to_move(board, uci)
        if move is None:
            break
        # Keep the position only if judging it does not need tactics resolved first.
        if (ply >= SKIP_PLIES and not board.is_check()
                and not board.is_capture(move) and abs(score) < 1200
                and move.promotion is None):
            core.set_from_board(ctx.bb, ctx.sq, ctx.st, board)
            kept.append((ctx.bb.copy(), ctx.sq.copy(), ctx.st.copy()))
        board.push(move)
    else:
        result = 0.5
    return kept, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--nodes", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("data/selfplay.npz"))
    parser.add_argument("--minutes", type=float, default=0.0, help="stop after this long")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    ctx = search.make_ctx()
    t0 = time.time()
    warm(ctx)
    print(f"compiled in {time.time() - t0:.0f}s, generating", flush=True)

    bbs, sqs, sts, res = [], [], [], []
    t0 = time.time()
    for game in range(args.games):
        ctx.tt[:] = 0
        kept, result = play(ctx, rng, args.nodes)
        for bb, sq, st in kept:
            bbs.append(bb)
            sqs.append(sq)
            sts.append(st)
            res.append(result)
        if (game + 1) % 10 == 0:
            rate = (time.time() - t0) / (game + 1)
            print(f"  {game + 1} games, {len(res)} positions, {rate:.1f}s/game",
                  flush=True)
        if args.minutes and (time.time() - t0) / 60.0 >= args.minutes:
            print(f"  stopping after {game + 1} games on the time limit", flush=True)
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        bb=np.array(bbs, dtype=np.int64), sq=np.array(sqs, dtype=np.int8),
        st=np.array(sts, dtype=np.int64), result=np.array(res, dtype=np.float32))
    print(f"wrote {len(res)} positions to {args.out}")


if __name__ == "__main__":
    main()
