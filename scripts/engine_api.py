"""Thin python helpers around the compiled engine, shared by the offline scripts."""

import os
import sys

os.environ.setdefault("NUMBA_OPT", "2")
os.environ.setdefault("NUMBA_SLP_VECTORIZE", "0")
os.environ.setdefault("NUMBA_LOOP_VECTORIZE", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess  # noqa: E402

import nx_core as core  # noqa: E402
import nx_search as search  # noqa: E402


def warm(ctx):
    """Pay the numba compile once per process."""
    core.set_from_board(ctx.bb, ctx.sq, ctx.st, chess.Board())
    search.new_search(ctx)
    n = search.root_moves(ctx)
    search.score_moves(ctx, 0, n, 0)
    search.search_root(ctx, search.I4, -search.INF, search.INF, n)


def search_position(ctx, board, nodes, max_depth=64):
    """Search to a node budget. Returns (uci move or None, score, depth reached)."""
    core.set_from_board(ctx.bb, ctx.sq, ctx.st, board)
    search.new_search(ctx)
    ctx.ctl[search.C_GLEN] = 0
    ctx.gkeys[0] = ctx.st[core.ST_KEY]
    ctx.ctl[search.C_LIMIT] = nodes
    n = search.root_moves(ctx)
    if n == 0:
        return None, 0, 0
    search.score_moves(ctx, 0, n, 0)
    best = int(ctx.ml[0, 0])
    score = 0
    reached = 0
    for depth in range(1, max_depth + 1):
        value = search.search_root(ctx, depth, -search.INF, search.INF, n)
        if ctx.ctl[search.C_STOP]:
            break
        score = int(value)
        best = int(ctx.ctl[search.C_BESTMOVE])
        reached = depth
        if abs(score) >= search.MATE - core.MAXPLY:
            break
    return core.move_to_uci(best), score, reached


def to_move(board, uci):
    if uci is None:
        return None
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return None
    return move if move in board.legal_moves else None
