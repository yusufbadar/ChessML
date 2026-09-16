"""Randomized rule-engine checks against python-chess.

This exercises legal generation and every make/undo state field on positions reached
through real games, plus awkward hand-written positions for special moves.
"""

import random
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nx_core as C  # noqa: E402
import nx_search as S  # noqa: E402


SPECIAL_FENS = (
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "4k3/PPP4p/8/2pP4/8/8/4p2P/4K3 w - c6 0 1",
    "8/P6k/8/8/8/8/6p1/4K3 w - - 0 1",
    "7k/5K2/6B1/8/8/8/8/8 b - - 0 1",
)

MATERIAL_FENS = (
    ("8/8/8/8/8/8/7k/K7 w - - 0 1", True),
    ("8/8/8/8/8/8/6Bk/K7 w - - 0 1", True),
    ("8/8/8/8/8/8/5BNk/K7 w - - 0 1", False),
    ("8/8/8/8/8/8/5Bnk/K7 w - - 0 1", False),
    ("8/8/8/8/8/8/4b1Bk/K7 w - - 0 1", True),
    ("8/8/8/8/8/8/3b2Bk/K7 w - - 0 1", False),
)


def assert_position_equal(ctx, board):
    bb = np.zeros_like(ctx.bb)
    sq = np.zeros_like(ctx.sq)
    st = np.zeros_like(ctx.st)
    C.set_from_board(bb, sq, st, board)
    if not np.array_equal(ctx.bb, bb):
        raise AssertionError(f"bitboards differ after {board.fen()}")
    if not np.array_equal(ctx.sq, sq):
        raise AssertionError(f"square map differs after {board.fen()}")
    for slot in (C.ST_SIDE, C.ST_CASTLE, C.ST_EP, C.ST_R50, C.ST_KEY, C.ST_PKEY):
        if ctx.st[slot] != st[slot]:
            raise AssertionError(
                f"state slot {slot}: {ctx.st[slot]} != {st[slot]} after {board.fen()}"
            )


def check_position(ctx, board, rng):
    C.set_from_board(ctx.bb, ctx.sq, ctx.st, board)
    before_bb = ctx.bb.copy()
    before_sq = ctx.sq.copy()
    before_st = ctx.st.copy()

    n = int(C.gen_moves(ctx.bb, ctx.sq, ctx.st, ctx.ml, S.I0, S.I0))
    actual = {C.move_to_uci(int(ctx.ml[0, i])) for i in range(n)}
    expected = {move.uci() for move in board.legal_moves}
    if actual != expected:
        raise AssertionError(
            f"move mismatch at {board.fen()}\n"
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    if bool(C.in_check(ctx.bb, ctx.st)) != board.is_check():
        raise AssertionError(f"check status differs at {board.fen()}")
    if not expected:
        return None

    uci = rng.choice(sorted(expected))
    move = chess.Move.from_uci(uci)
    internal = next(
        int(ctx.ml[0, i]) for i in range(n) if C.move_to_uci(int(ctx.ml[0, i])) == uci
    )
    C.do_move(ctx.bb, ctx.sq, ctx.st, ctx.und, S.I0, np.int32(internal))
    board.push(move)
    assert_position_equal(ctx, board)
    if bool(C.in_check(ctx.bb, ctx.st)) != board.is_check():
        raise AssertionError(f"post-move check status differs at {board.fen()}")

    C.undo_move(ctx.bb, ctx.sq, ctx.st, ctx.und, S.I0)
    board.pop()
    if not np.array_equal(ctx.bb, before_bb) or not np.array_equal(ctx.sq, before_sq):
        raise AssertionError(f"make/undo board mismatch at {board.fen()}")
    if not np.array_equal(ctx.st, before_st):
        raise AssertionError(f"make/undo state mismatch at {board.fen()}")
    return move


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_plies = int(sys.argv[2]) if len(sys.argv) > 2 else 160
    rng = random.Random(0xA1C4E55)
    ctx = S.make_ctx()
    checked = 0

    for fen in SPECIAL_FENS:
        check_position(ctx, chess.Board(fen), rng)
        checked += 1

    for game in range(games):
        board = chess.Board()
        for _ in range(max_plies):
            move = check_position(ctx, board, rng)
            checked += 1
            if move is None:
                break
            board.push(move)
            if board.is_game_over(claim_draw=True):
                check_position(ctx, board, rng)
                checked += 1
                break

    for fen, expected in MATERIAL_FENS:
        board = chess.Board(fen)
        C.set_from_board(ctx.bb, ctx.sq, ctx.st, board)
        actual = bool(S.insufficient_material(ctx.bb))
        reference = board.is_insufficient_material()
        if actual != expected or actual != reference:
            raise AssertionError(
                f"insufficient material mismatch: engine={actual}, "
                f"python-chess={reference}, expected={expected}, {fen}"
            )

    print(f"differential: {checked:,} positions across {games} games, all exact")


if __name__ == "__main__":
    main()
