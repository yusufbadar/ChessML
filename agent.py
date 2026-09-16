"""AI Chessathon submission.

get_move(fen, time_left_ms) -> uci. Everything under it is a bitboard engine compiled by
numba: magic-bitboard move generation, alpha-beta with a transposition table and the usual
reductions, and a tapered hand evaluation whose weights live in one flat vector.

How the parts fit together:

  nx_magic.py    magic multipliers, found offline by scripts/genmagic.py
  nx_core.py     bitboards, tables, legal move generation, make/unmake, SEE
  nx_eval.py     tapered evaluation
  nx_params.py   every evaluation weight, as one vector a tuner can rewrite
  nx_search.py   alpha-beta, one depth per call
  nx_fallback.py a small pure-python search, the last resort if nothing has compiled yet
  agent.py       this file: clock, repetition history, warm-up, and the call into search

Three things about this file are worth knowing before reading it.

**Start-up is staged.** numba needs about thirty seconds on one core and the platform
allows ninety. A slower judge machine can still overrun, so the
warm-up compiles in dependency order and publishes each tier as it lands: the evaluation
and quiescence first, then the full search. If the clock runs out mid-way we still answer
from compiled code -- a Python-driven alpha-beta over the compiled primitives -- rather
than from the pure-python fallback. Losing depth for a move or two is survivable; losing
the evaluation as well is not.

**The clock is enforced by a watchdog thread**, never by reading a clock inside the
search. Compiled search functions release the GIL, so the watchdog can set a stop flag in
a shared array and the search sees it within a few thousand nodes. No timing code runs on
the hot path at all.

**We are told a position, never a move**, so the engine keeps its own list of the
positions this game has passed through. Without it a won position can be drawn by a
repetition we were never told about.
"""

import os
import time

# The 90 second init budget is measured from when the process starts, not from when the
# imports finish, so the clock we compare against has to start on the first line.
START = time.monotonic()

# numba reads these when it is imported, so they have to be set first. Neither
# vectoriser can help scalar bitboard code and both are expensive to run, and dropping
# one optimisation level costs nothing measurable in node rate.
os.environ.setdefault("NUMBA_OPT", "2")
os.environ.setdefault("NUMBA_SLP_VECTORIZE", "0")
os.environ.setdefault("NUMBA_LOOP_VECTORIZE", "0")

import threading  # noqa: E402

import chess  # noqa: E402
import numpy as np  # noqa: E402

import nx_core as core  # noqa: E402
import nx_fallback as fallback  # noqa: E402
import nx_search as search  # noqa: E402

# The platform allows 90 seconds before the clock starts and a missing ready line is a
# lost game. Stop waiting with enough left for the interpreter's own start-up.
WARMUP_DEADLINE_S = 85.0

MOVE_OVERHEAD_MS = 35      # protocol and process scheduling, charged to us
MIN_BUDGET_MS = 8          # never think for less than this, however low the clock is
INCREMENT_MS = 500         # the platform's per-move increment, landing after we move

ctx = search.make_ctx()

# Warm-up tiers, published as they compile. `quiet_ready` means the evaluation, move
# generation and quiescence are compiled; `ready` means the whole search is.
quiet_ready = threading.Event()
ready = threading.Event()
warm_done = threading.Event()
_warm_error = []


def _warm_up():
    """Compile in dependency order, publishing each tier as soon as it is usable."""
    try:
        board = chess.Board()
        core.set_from_board(ctx.bb, ctx.sq, ctx.st, board)
        n = core.gen_moves(ctx.bb, ctx.sq, ctx.st, ctx.ml, 0, 0)
        search.static_eval(ctx)
        core.see_value(ctx.bb, ctx.sq, ctx.st, ctx.ml[0, 0], ctx.gain)
        search.score_moves(ctx, 0, n, 0)
        search.qsearch(ctx, 0, -search.INF, search.INF)
        core.do_move(ctx.bb, ctx.sq, ctx.st, ctx.und, 0, ctx.ml[0, 0])
        core.undo_move(ctx.bb, ctx.sq, ctx.st, ctx.und, 0)
        core.in_check(ctx.bb, ctx.st)
        quiet_ready.set()
        print(f"warm-up: evaluation and quiescence at {time.monotonic() - START:.1f}s")

        # Depth four touches the rest: null move, reductions, history updates, the
        # transposition table, and the aspiration path through search_root.
        search.new_search(ctx)
        n = search.root_moves(ctx)
        search.score_moves(ctx, 0, n, 0)
        search.search_root(ctx, search.I4, -search.INF, search.INF, n)
        # A position with pawns on the seventh and an en passant right, so promotion and
        # en passant compile here rather than in the middle of a game.
        core.set_from_board(ctx.bb, ctx.sq, ctx.st,
                            chess.Board("4k3/PPP4p/8/2pP4/8/8/4p2P/4K3 w - c6 0 1"))
        search.new_search(ctx)
        n = search.root_moves(ctx)
        search.score_moves(ctx, 0, n, 0)
        search.search_root(ctx, search.I4, -search.INF, search.INF, n)
        ctx.tt[:] = 0
        ctx.ptt[:] = 0
        ready.set()
    except Exception as error:            # pragma: no cover - defensive only
        _warm_error.append(error)
        print(f"warm-up failed: {error!r}")
    finally:
        warm_done.set()


_warmer = threading.Thread(target=_warm_up, daemon=True)
_warmer.start()
warm_done.wait(timeout=max(1.0, WARMUP_DEADLINE_S - (time.monotonic() - START)))
print(f"warm-up {'done' if ready.is_set() else 'INCOMPLETE'} "
      f"after {time.monotonic() - START:.1f}s")

# ---- game state that a single fen does not carry -----------------------------------------

_keys: list[int] = []      # every position this game has stood in, ours and theirs
_last_fullmove = 0


def _remember(board):
    """Keep the repetition history current, and set the root index for the search."""
    global _keys, _last_fullmove
    core.set_from_board(ctx.bb, ctx.sq, ctx.st, board)
    key = int(ctx.st[core.ST_KEY])
    if board.fullmove_number < _last_fullmove:
        _keys = []                       # a new game in a reused process
    _last_fullmove = board.fullmove_number
    if board.halfmove_clock == 0:
        _keys = []                       # nothing before an irreversible move can repeat
    if not _keys or _keys[-1] != key:
        _keys.append(key)
    if len(_keys) > search.GAME_KEYS - core.MAXPLY - 8:
        del _keys[:len(_keys) // 2]
    for i, k in enumerate(_keys):
        ctx.gkeys[i] = k
    ctx.ctl[search.C_GLEN] = len(_keys) - 1


def _record_reply(board, move):
    """Remember the position our own move produces, so we can see it repeat later."""
    board.push(move)
    core.set_from_board(ctx.bb, ctx.sq, ctx.st, board)
    _keys.append(int(ctx.st[core.ST_KEY]))
    board.pop()


# ---- clock ---------------------------------------------------------------------------------


def _budget(board, time_left_ms):
    """Soft and hard limits in milliseconds.

    Soft is what we expect to spend: stop starting new iterations past it. Hard is where
    the watchdog cuts the search off mid-iteration. Flagging loses the game outright, so
    the hard limit is always a fraction of the clock rather than a multiple of the soft
    one, and the two are chosen so that a normal game finishes with a few seconds spare
    rather than a third of the clock unspent.
    """
    usable = max(0, time_left_ms - MOVE_OVERHEAD_MS)
    if usable <= 0:
        return MIN_BUDGET_MS, MIN_BUDGET_MS
    moves_left = max(20, min(30, 42 - board.fullmove_number))
    soft = usable / moves_left + 0.75 * INCREMENT_MS
    hard = min(usable * 0.22, soft * 2.2)
    soft = min(soft, hard)
    return max(MIN_BUDGET_MS, int(soft)), max(MIN_BUDGET_MS, int(hard))


def _think(board, soft_ms, hard_ms):
    """Iterative deepening with aspiration windows, driven from here so the clock and the
    best-move stability check stay out of compiled code."""
    started = time.monotonic()
    search.new_search(ctx)
    n = search.root_moves(ctx)
    if n == 0:
        return None
    search.score_moves(ctx, 0, n, 0)

    stop_at = started + hard_ms / 1000.0
    watchdog = threading.Timer(hard_ms / 1000.0, lambda: ctx.ctl.__setitem__(search.C_STOP, 1))
    watchdog.daemon = True
    watchdog.start()

    best_uci = core.move_to_uci(int(ctx.ml[0, 0]))
    try:
        score = 0
        previous = None
        unstable = 0
        for depth in range(1, core.MAXPLY - 8):
            iteration_started = time.monotonic()
            # Aspiration: assume this depth lands near the last one and search a narrow
            # window, widening only when it does not.
            window = 24
            while True:
                if depth <= 3:
                    alpha, beta = -search.INF, search.INF
                else:
                    alpha, beta = score - window, score + window
                value = search.search_root(ctx, depth, alpha, beta, n)
                if ctx.ctl[search.C_STOP]:
                    break
                if alpha < value < beta or window > 1200:
                    score = value
                    break
                window *= 3

            if ctx.ctl[search.C_STOP]:
                break
            move = int(ctx.ctl[search.C_BESTMOVE])
            if move:
                best_uci = core.move_to_uci(move)
            unstable = unstable + 1 if move != previous else 0
            previous = move

            now = time.monotonic()
            elapsed_ms = (now - started) * 1000.0
            iteration_ms = (now - iteration_started) * 1000.0
            if abs(score) >= search.MATE - core.MAXPLY and depth >= 6:
                break                       # a forced mate is not going to improve
            # A move that keeps changing is worth more time. Otherwise stop, and stop
            # predictively: each iteration costs well over the last, so finishing inside
            # the budget means not starting one that clearly cannot fit in what is left.
            limit = soft_ms * (1.35 if unstable else 1.0)
            if (elapsed_ms >= limit or elapsed_ms + 1.6 * iteration_ms >= limit
                    or now >= stop_at):
                break
    finally:
        watchdog.cancel()
        ctx.ctl[search.C_STOP] = 1
    return best_uci


# ---- the interim engine, used only while the full search is still compiling ----------------

PIECE_ORDER = (100, 320, 330, 500, 950, 20000)


def _interim(board, budget_ms):
    """Alpha-beta driven from Python over whatever is already compiled.

    Every node's evaluation, move generation and quiescence are the real ones; only the
    recursion and the move ordering are interpreted. That is perhaps three or four plies
    in the time available, which is far short of the real search and far better than a
    pure-python evaluation.
    """
    deadline = time.monotonic() + budget_ms / 1000.0
    core.set_from_board(ctx.bb, ctx.sq, ctx.st, board)
    ctx.ctl[search.C_STOP] = 0
    best_move = 0
    for depth in range(1, 5):
        value, move = _interim_ab(0, depth, -search.INF, search.INF, deadline)
        if move and time.monotonic() <= deadline:
            best_move = move
        if time.monotonic() > deadline:
            break
    return core.move_to_uci(best_move) if best_move else None


def _interim_ab(ply, depth, alpha, beta, deadline):
    if depth <= 0 or ply >= 24:
        return int(search.qsearch(ctx, ply, alpha, beta)), 0
    n = int(core.gen_moves(ctx.bb, ctx.sq, ctx.st, ctx.ml, ply, 0))
    if n == 0:
        return (-search.MATE + ply if core.in_check(ctx.bb, ctx.st) else 0), 0
    # Captures first, biggest victim first. Cheap, and enough to make alpha-beta work.
    moves = sorted(
        (int(ctx.ml[ply, i]) for i in range(n)),
        key=lambda m: PIECE_ORDER[int(ctx.sq[(m >> 6) & 63]) % 6]
        if int(ctx.sq[(m >> 6) & 63]) != core.NO_PIECE else -1,
        reverse=True)
    best, best_move = -search.INF, 0
    for m in moves:
        core.do_move(ctx.bb, ctx.sq, ctx.st, ctx.und, ply, np.int32(m))
        value = -_interim_ab(ply + 1, depth - 1, -beta, -alpha, deadline)[0]
        core.undo_move(ctx.bb, ctx.sq, ctx.st, ctx.und, ply)
        if value > best:
            best, best_move = value, m
            if value > alpha:
                alpha = value
                if alpha >= beta:
                    break
        if time.monotonic() > deadline:
            break
    return best, best_move


# ---- the platform's entry point --------------------------------------------------------------


def get_move(fen: str, time_left_ms: int) -> str:
    entered = time.monotonic()
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        return "0000"
    chosen = legal[0]
    soft_ms, hard_ms = _budget(board, time_left_ms)
    try:
        if ready.is_set():
            _remember(board)
            uci = _think(board, soft_ms, hard_ms)
        elif quiet_ready.is_set():
            # Still compiling the search, but the evaluation is real.
            _remember(board)
            uci = _interim(board, min(soft_ms, 900))
        else:
            move = fallback.choose(board, min(soft_ms, 800))
            uci = move.uci() if move is not None else None
        if uci is not None:
            candidate = chess.Move.from_uci(uci)
            if candidate in board.legal_moves:
                chosen = candidate
                if ready.is_set():
                    _record_reply(board, chosen)
    except Exception as error:              # never lose a game to an exception
        print(f"search failed, playing a legal move: {error!r}")
    spent = (time.monotonic() - entered) * 1000.0
    if spent > hard_ms + 250:
        # The watchdog should make this impossible. If it ever prints, the clock is at
        # risk and the validation log is where we will find out.
        print(f"OVERSHOOT move {board.fullmove_number}: {spent:.0f}ms spent, "
              f"hard limit {hard_ms}ms, clock was {time_left_ms}ms")
    return chosen.uci()
