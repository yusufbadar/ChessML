"""Alpha-beta search: iterative deepening, principal variation search, a transposition
table, null-move and late-move reductions, and the usual family of shallow prunings.

Every function is compiled with nogil=True so the watchdog thread can set the stop flag
while a search is running. This enforces the clock without reading a clock inside
compiled code.

One depth per call. Python drives the deepening loop, so time management, aspiration
windows and best-move stability live in code that is easy to read and to change.

The node function is split into small pieces rather than written as one long routine.
numba's compile time grows far faster than linearly with the size of a single function,
and the whole engine has to compile inside a 90 second init budget.
"""

from collections import namedtuple

import numpy as np
from numba import njit

from nx_core import (
    BISHOP, F_EP, F_PROMO, KNIGHT, MAXMOVES, MAXPLY, NO_PIECE, PAWN, QUEEN, ROOK,
    SEE_VALUE, ST_KEY, ST_R50, ST_SIDE, U_N, do_move, do_null, gen_moves, in_check,
    m_flag, m_from, m_promo, m_to, see_ge, shr, undo_move, undo_null,
)
from nx_eval import S_N, evaluate, insufficient_material
from nx_params import load

INF = 32001
MATE = 32000
MATE_IN_MAX = MATE - MAXPLY
NO_EVAL = -32001

BOUND_UPPER, BOUND_LOWER, BOUND_EXACT = 1, 2, 3

TT_BITS = 22
TT_SIZE = 1 << TT_BITS
PTT_BITS = 15

# control slots
C_NODES, C_STOP, C_SOFT, C_GLEN, C_SELDEPTH, C_BESTMOVE, C_BESTSCORE, C_AGE = range(8)
C_LIMIT = 8       # node ceiling, so A/B tests can be run on work rather than wall time
C_CONTEMPT = 9    # how much worse than nothing a draw is, from the root side's view.
                  # Measured at 16, 28 and 45 against 0 over 150 games each and none of
                  # them gained, so it ships at zero. The knob stays because self-play is
                  # the wrong test for it: contempt pays when you are stronger than the
                  # field, and both sides of an A/B are equally strong by construction.
C_ROOTSIDE = 10   # whose turn it was at the root, so contempt has a sign
C_N = 16

# per-ply stack slots
SK_EVAL, SK_PIECE, SK_TO, SK_MOVE, SK_N = 0, 1, 2, 3, 4

GAME_KEYS = 1400

# numba compiles a separate specialisation when an argument arrives as a Python literal,
# so every scalar handed to another jitted function goes through one of these instead.
I0 = np.int64(0)
I1 = np.int64(1)
I2 = np.int64(2)
I4 = np.int64(4)
HIST_MAX = np.int64(16384)
TRUE = np.bool_(True)
FALSE = np.bool_(False)

# ---- reduction and pruning tables, shaped once at import -------------------------------

_lmr = np.zeros((64, 64), dtype=np.int32)
for _d in range(1, 64):
    for _m in range(1, 64):
        _lmr[_d, _m] = int(0.70 + np.log(_d) * np.log(_m) / 2.30)
LMR = _lmr

_lmp = np.zeros((2, 12), dtype=np.int32)
for _d in range(12):
    _lmp[0, _d] = int(2.4 + 0.55 * _d * _d)      # not improving
    _lmp[1, _d] = int(3.9 + 0.95 * _d * _d)      # improving
LMP = _lmp

Ctx = namedtuple("Ctx", [
    "bb", "sq", "st", "und", "ml", "ms", "tt", "killers", "hist", "conth", "cmove",
    "pv", "pvlen", "stack", "ctl", "gkeys", "gain", "sc", "ptt", "P", "qbuf",
])


def make_ctx(params=None):
    """Allocate every array the search owns. Called once, at import."""
    return Ctx(
        bb=np.zeros(15, dtype=np.int64),
        sq=np.full(64, NO_PIECE, dtype=np.int8),
        st=np.zeros(8, dtype=np.int64),
        und=np.zeros((MAXPLY + 8, U_N), dtype=np.int64),
        ml=np.zeros((MAXPLY + 8, MAXMOVES), dtype=np.int32),
        ms=np.zeros((MAXPLY + 8, MAXMOVES), dtype=np.int32),
        tt=np.zeros((TT_SIZE, 2), dtype=np.int64),
        killers=np.zeros((MAXPLY + 8, 2), dtype=np.int32),
        hist=np.zeros((2, 64, 64), dtype=np.int32),
        conth=np.zeros((13 * 64, 12 * 64), dtype=np.int16),
        cmove=np.zeros((13, 64), dtype=np.int32),
        pv=np.zeros((MAXPLY + 2, MAXPLY + 2), dtype=np.int32),
        pvlen=np.zeros(MAXPLY + 2, dtype=np.int32),
        stack=np.zeros((MAXPLY + 8, SK_N), dtype=np.int64),
        ctl=np.zeros(C_N, dtype=np.int64),
        gkeys=np.zeros(GAME_KEYS, dtype=np.int64),
        gain=np.zeros(40, dtype=np.int64),
        sc=np.zeros(S_N, dtype=np.int64),
        ptt=np.zeros((1 << PTT_BITS, 5), dtype=np.int64),
        P=(load() if params is None else params).astype(np.int32),
        qbuf=np.zeros((MAXPLY + 8, 64), dtype=np.int32),
    )


# ---- transposition table ---------------------------------------------------------------


@njit(nogil=True, cache=False, error_model="numpy")
def tt_index(key):
    return (key ^ shr(key, TT_BITS) ^ shr(key, 2 * TT_BITS)) & (TT_SIZE - 1)


@njit(nogil=True, cache=False, error_model="numpy")
def tt_store(c, key, move, score, ev, depth, bound, ply):
    """Key and payload are stored xored together, so a torn pair simply fails to match."""
    if score >= MATE_IN_MAX:
        score += ply
    elif score <= -MATE_IN_MAX:
        score -= ply
    i = tt_index(key)
    old = c.tt[i, 1]
    if (c.tt[i, 0] ^ old) == key:
        if (bound != BOUND_EXACT and ((old >> 58) & 63) == c.ctl[C_AGE]
                and ((old >> 48) & 0xFF) - 1 > depth + 3):
            return
        if move == 0:
            move = old & 0xFFFF
    data = ((move & 0xFFFF) | ((score + 32768) << 16) | ((ev + 32768) << 32)
            | ((depth + 1) << 48) | (bound << 56) | ((c.ctl[C_AGE] & 63) << 58))
    c.tt[i, 1] = data
    c.tt[i, 0] = key ^ data


@njit(nogil=True, cache=False, error_model="numpy")
def tt_probe(c, key):
    """Returns (hit, move, score, eval, depth, bound). Score is not yet ply-corrected."""
    i = tt_index(key)
    data = c.tt[i, 1]
    if (c.tt[i, 0] ^ data) != key or data == 0:
        return False, 0, 0, NO_EVAL, -1, 0
    return (True, data & 0xFFFF, ((data >> 16) & 0xFFFF) - 32768,
            ((data >> 32) & 0xFFFF) - 32768, ((data >> 48) & 0xFF) - 1, (data >> 56) & 3)


@njit(nogil=True, cache=False, error_model="numpy")
def from_tt_score(score, ply):
    if score >= MATE_IN_MAX:
        return score - ply
    if score <= -MATE_IN_MAX:
        return score + ply
    return score


# ---- draws --------------------------------------------------------------------------------


@njit(nogil=True, cache=False, error_model="numpy")
def is_repetition(c, ply):
    """Twofold inside the search counts as a draw: whoever can force it, will."""
    key = c.st[ST_KEY]
    j = c.ctl[C_GLEN] + ply
    back = c.st[ST_R50]
    if back > j:
        back = j
    i = j - 2
    while i >= j - back:
        if c.gkeys[i] == key:
            return True
        i -= 2
    return False


@njit(nogil=True, cache=False, error_model="numpy")
def draw_score(c):
    """What a draw is worth here.

    Zero is the honest answer and a bad practical one: in a level position it makes a
    repetition exactly as attractive as playing on, and the engine shuffles into a
    threefold rather than trying. Scoring a draw slightly against the side that started
    the search makes it play on when it is level and still take the draw when it is
    actually worse.
    """
    v = c.ctl[C_CONTEMPT]
    return -v if c.st[ST_SIDE] == c.ctl[C_ROOTSIDE] else v


@njit(nogil=True, cache=False, error_model="numpy")
def has_pieces(c, side):
    o = side * 6
    return (c.bb[o + KNIGHT] | c.bb[o + BISHOP] | c.bb[o + ROOK] | c.bb[o + QUEEN]) != 0


# ---- move ordering -------------------------------------------------------------------------


@njit(nogil=True, cache=False, error_model="numpy")
def is_noisy(c, m):
    return c.sq[m_to(m)] != NO_PIECE or m_flag(m) == F_EP or m_flag(m) == F_PROMO


@njit(nogil=True, cache=False, error_model="numpy")
def conth_prev(c, ply, back):
    """Index of the move made `back` plies ago, or the null slot when there is not one."""
    if ply < back:
        return 12 * 64
    p = c.stack[ply - back, SK_PIECE]
    if p >= NO_PIECE:
        return 12 * 64
    return p * 64 + c.stack[ply - back, SK_TO]


@njit(nogil=True, cache=False, error_model="numpy")
def score_moves(c, ply, n, tt_move):
    """One pass over the move list, assigning the key each move is later picked by."""
    side = c.st[ST_SIDE]
    prev1 = conth_prev(c, ply, I1)
    prev2 = conth_prev(c, ply, I2)
    k0 = c.killers[ply, 0]
    k1 = c.killers[ply, 1]
    counter = c.cmove[prev1 // 64, prev1 % 64]
    for i in range(n):
        m = c.ml[ply, i]
        if m == tt_move:
            c.ms[ply, i] = 2_000_000_000
            continue
        frm = m_from(m)
        to = m_to(m)
        flag = m_flag(m)
        victim = c.sq[to]
        if victim != NO_PIECE or flag == F_EP or flag == F_PROMO:
            if victim != NO_PIECE:
                mvv = SEE_VALUE[victim % 6]
            elif flag == F_EP:
                mvv = SEE_VALUE[PAWN]
            else:
                mvv = 0
            if flag == F_PROMO:
                mvv += SEE_VALUE[1 + m_promo(m)] - SEE_VALUE[PAWN]
            rank = mvv * 32 - SEE_VALUE[c.sq[frm] % 6] // 8
            if see_ge(c.bb, c.sq, c.st, m, I0, c.gain):
                c.ms[ply, i] = 1_000_000 + rank
            else:
                c.ms[ply, i] = -1_500_000 + rank
        elif m == k0:
            c.ms[ply, i] = 900_000
        elif m == k1:
            c.ms[ply, i] = 890_000
        elif m == counter:
            c.ms[ply, i] = 880_000
        else:
            cur = c.sq[frm] * 64 + to
            c.ms[ply, i] = (c.hist[side, frm, to]
                            + c.conth[prev1, cur] + c.conth[prev2, cur])


@njit(nogil=True, cache=False, error_model="numpy")
def pick_move(c, ply, i, n):
    """Selection sort, one step at a time: most nodes never look past the first few."""
    best = i
    for j in range(i + 1, n):
        if c.ms[ply, j] > c.ms[ply, best]:
            best = j
    if best != i:
        c.ml[ply, i], c.ml[ply, best] = c.ml[ply, best], c.ml[ply, i]
        c.ms[ply, i], c.ms[ply, best] = c.ms[ply, best], c.ms[ply, i]
    return c.ml[ply, i]


@njit(nogil=True, cache=False, error_model="numpy")
def _bump(value, bonus, limit):
    """Move a history value towards its bound, so old evidence decays instead of saturating."""
    return value + bonus - value * (bonus if bonus > 0 else -bonus) // limit


@njit(nogil=True, cache=False, error_model="numpy")
def update_quiet_history(c, ply, move, depth, ntried):
    """Reward the move that caused the cutoff, penalise the quiets tried before it."""
    side = c.st[ST_SIDE]
    bonus = 32 * depth * depth + 64 * depth
    if bonus > 4800:
        bonus = 4800
    prev1 = conth_prev(c, ply, I1)
    prev2 = conth_prev(c, ply, I2)

    if c.killers[ply, 0] != move:
        c.killers[ply, 1] = c.killers[ply, 0]
        c.killers[ply, 0] = move
    c.cmove[prev1 // 64, prev1 % 64] = move

    for i in range(ntried + 1):
        m = move if i == ntried else c.qbuf[ply, i]
        b = bonus if i == ntried else -bonus
        frm = m_from(m)
        to = m_to(m)
        c.hist[side, frm, to] = _bump(np.int64(c.hist[side, frm, to]), b, HIST_MAX)
        cur = c.sq[frm] * 64 + to
        c.conth[prev1, cur] = _bump(np.int64(c.conth[prev1, cur]), b // 2, HIST_MAX)
        c.conth[prev2, cur] = _bump(np.int64(c.conth[prev2, cur]), b // 2, HIST_MAX)


@njit(nogil=True, cache=False, error_model="numpy")
def copy_pv(c, ply, m):
    c.pv[ply, 0] = m
    ln = c.pvlen[ply + 1]
    for k in range(ln):
        c.pv[ply, k + 1] = c.pv[ply + 1, k]
    c.pvlen[ply] = ln + 1


# ---- quiescence ----------------------------------------------------------------------------


@njit(nogil=True, cache=False, error_model="numpy")
def qsearch(c, ply, alpha, beta):
    """Search on past the horizon until the position is quiet enough to judge."""
    c.ctl[C_NODES] += 1
    if c.ctl[C_NODES] >= c.ctl[C_LIMIT]:
        c.ctl[C_STOP] = 1
    if c.ctl[C_STOP] != 0:
        return 0
    if ply > c.ctl[C_SELDEPTH]:
        c.ctl[C_SELDEPTH] = ply
    if ply >= MAXPLY - 4:
        return evaluate(c.bb, c.sq, c.st, c.P, c.sc, c.ptt)

    key = c.st[ST_KEY]
    c.gkeys[c.ctl[C_GLEN] + ply] = key
    if c.st[ST_R50] >= 100 or is_repetition(c, ply):
        return draw_score(c)

    hit, tt_move, tt_score, tt_eval, tt_depth, tt_bound = tt_probe(c, key)
    if hit:
        s = from_tt_score(tt_score, ply)
        if tt_bound == BOUND_EXACT or (tt_bound == BOUND_LOWER and s >= beta) \
                or (tt_bound == BOUND_UPPER and s <= alpha):
            return s

    checked = in_check(c.bb, c.st)
    if checked:
        best = -INF
        stand = NO_EVAL
    else:
        stand = tt_eval if hit and tt_eval != NO_EVAL else \
            evaluate(c.bb, c.sq, c.st, c.P, c.sc, c.ptt)
        best = stand
        if best >= beta:
            return best
        if best > alpha:
            alpha = best

    n = gen_moves(c.bb, c.sq, c.st, c.ml, ply, I0 if checked else I1)
    if n == 0:
        return -MATE + ply if checked else best
    score_moves(c, ply, n, tt_move)

    best_move = 0
    old_alpha = alpha
    for i in range(n):
        m = pick_move(c, ply, i, n)
        if not checked:
            if c.ms[ply, i] < -1_000_000:
                break  # every remaining capture loses material outright
            victim = c.sq[m_to(m)]
            swing = SEE_VALUE[victim % 6] if victim != NO_PIECE else SEE_VALUE[PAWN]
            if m_flag(m) == F_PROMO:
                swing += SEE_VALUE[1 + m_promo(m)]
            if stand + swing + 140 <= alpha:
                continue  # delta pruning: even winning this much would not reach alpha
        c.stack[ply, SK_PIECE] = c.sq[m_from(m)]
        c.stack[ply, SK_TO] = m_to(m)
        c.stack[ply, SK_MOVE] = m
        do_move(c.bb, c.sq, c.st, c.und, ply, m)
        v = -qsearch(c, ply + 1, -beta, -alpha)
        undo_move(c.bb, c.sq, c.st, c.und, ply)
        if c.ctl[C_STOP] != 0:
            return 0
        if v > best:
            best = v
            best_move = m
            if v > alpha:
                alpha = v
                if v >= beta:
                    break

    bound = BOUND_LOWER if best >= beta else (
        BOUND_EXACT if best > old_alpha else BOUND_UPPER)
    tt_store(c, key, best_move, best, stand, I0, bound, ply)
    return best


# ---- the pieces of a node, kept small on purpose --------------------------------------------


@njit(nogil=True, cache=False, error_model="numpy")
def probe(c, ply, depth, alpha, beta, pv_node):
    """Table lookup and its cutoff test. Returns (cut, value, move, hit, bound, eval)."""
    hit, tt_move, tt_score, tt_eval, tt_depth, tt_bound = tt_probe(c, c.st[ST_KEY])
    value = from_tt_score(tt_score, ply) if hit else 0
    cut = (hit and not pv_node and tt_depth >= depth
           and (tt_bound == BOUND_EXACT
                or (tt_bound == BOUND_LOWER and value >= beta)
                or (tt_bound == BOUND_UPPER and value <= alpha)))
    return cut, value, tt_move, hit, tt_bound, tt_eval


@njit(nogil=True, cache=False, error_model="numpy")
def node_static(c, checked, hit, tt_eval, tt_value, tt_bound):
    """The evaluation of this node, and the sharpened version used for pruning.

    A score the table got from a real search is a better estimate of the position than
    the evaluation function is, so pruning uses it when it points the right way. The raw
    evaluation is still what gets stored, because that is what it means.
    """
    if checked:
        return NO_EVAL, NO_EVAL
    raw = tt_eval if hit and tt_eval != NO_EVAL else \
        evaluate(c.bb, c.sq, c.st, c.P, c.sc, c.ptt)
    static = raw
    if hit and (tt_bound == BOUND_EXACT or (tt_bound == BOUND_LOWER and tt_value > raw)
                or (tt_bound == BOUND_UPPER and tt_value < raw)):
        static = tt_value
    return static, raw


@njit(nogil=True, cache=False, error_model="numpy")
def shallow_prune(c, ply, depth, alpha, beta, static, improving):
    """Two ways to leave a node without searching a move. Returns (value, cut)."""
    # Reverse futility: so far ahead that handing back a depth-scaled margin still beats
    # beta, and no single quiet move is going to cost that much.
    if depth <= 8 and static - (66 * depth - 42 * improving) >= beta:
        return (static + beta) // 2, True
    # Razoring: so far behind that only a capture sequence could rescue this, so ask
    # quiescence directly rather than spend a full-width search finding that out.
    if depth <= 3 and static + 180 * depth < alpha:
        v = qsearch(c, ply, alpha - 1, alpha)
        if v < alpha:
            return v, True
    return 0, False


@njit(nogil=True, cache=False, error_model="numpy")
def skip_move(c, ply, m, depth, alpha, static, played, lmp_limit, noisy, checked,
              hist_score):
    """Shallow prunings inside the move loop. True means do not search this move."""
    if not noisy:
        if depth <= 11 and played >= lmp_limit:
            return True
        if depth <= 8 and not checked and static != NO_EVAL \
                and static + 100 + 112 * depth <= alpha:
            return True
        if depth <= 6 and hist_score < -2200 * depth:
            return True
    margin = -82 * depth if noisy else -18 * depth * depth
    return depth <= 8 and not see_ge(c.bb, c.sq, c.st, m, margin, c.gain)


@njit(nogil=True, cache=False, error_model="numpy")
def reduction(depth, played, pv_node, improving, cut_node, checked, hist_score):
    r = LMR[depth if depth < 64 else 63, played if played < 64 else 63]
    if pv_node:
        r -= 1
    if improving:
        r -= 1
    if cut_node:
        r += 1
    if checked:
        r -= 1
    return r - hist_score // 6000


# ---- main search ------------------------------------------------------------------------------


@njit(nogil=True, cache=False, error_model="numpy")
def negamax(c, ply, depth, alpha, beta, cut_node):
    pv_node = beta - alpha > 1
    c.pvlen[ply] = 0
    if depth <= 0:
        return qsearch(c, ply, alpha, beta)
    c.ctl[C_NODES] += 1
    if c.ctl[C_NODES] >= c.ctl[C_LIMIT]:
        c.ctl[C_STOP] = 1
    if c.ctl[C_STOP] != 0:
        return 0

    key = c.st[ST_KEY]
    c.gkeys[c.ctl[C_GLEN] + ply] = key
    if c.st[ST_R50] >= 100 or is_repetition(c, ply) or insufficient_material(c.bb):
        return draw_score(c)
    if ply >= MAXPLY - 4:
        return evaluate(c.bb, c.sq, c.st, c.P, c.sc, c.ptt)

    # mate distance pruning: a shorter mate is known, so stop looking for a longer one
    if alpha < -MATE + ply:
        alpha = -MATE + ply
    if beta > MATE - ply - 1:
        beta = MATE - ply - 1
    if alpha >= beta:
        return alpha

    cut, tt_value, tt_move, hit, tt_bound, tt_eval = probe(c, ply, depth, alpha, beta,
                                                           pv_node)
    if cut:
        return tt_value

    checked = in_check(c.bb, c.st)
    static, raw_eval = node_static(c, checked, hit, tt_eval, tt_value, tt_bound)
    c.stack[ply, SK_EVAL] = raw_eval

    improving = (not checked and ply >= 2 and c.stack[ply - 2, SK_EVAL] != NO_EVAL
                 and raw_eval > c.stack[ply - 2, SK_EVAL])

    if not pv_node and not checked and abs(beta) < MATE_IN_MAX:
        v, cut = shallow_prune(c, ply, depth, alpha, beta, static, improving)
        if cut:
            return v
        # Null move: hand the opponent a free move. Still above beta afterwards means
        # the position is good enough that a real move will be too.
        after_null = ply > 0 and c.stack[ply - 1, SK_MOVE] == 0
        if depth >= 3 and static >= beta and not after_null and has_pieces(c, c.st[ST_SIDE]):
            adv = (static - beta) // 190
            r = 3 + depth // 4 + (adv if adv < 3 else 3)
            c.stack[ply, SK_PIECE] = NO_PIECE
            c.stack[ply, SK_TO] = 0
            c.stack[ply, SK_MOVE] = 0
            do_null(c.bb, c.sq, c.st, c.und, ply)
            v = -negamax(c, ply + 1, depth - r, -beta, -beta + 1, not cut_node)
            undo_null(c.bb, c.sq, c.st, c.und, ply)
            if c.ctl[C_STOP] != 0:
                return 0
            if v >= beta:
                return beta if v >= MATE_IN_MAX else v

    # No table move at this depth means the ordering here is guesswork. A shallower
    # search is a cheaper way to find a first move than searching this one blind.
    if depth >= 4 and tt_move == 0 and not checked:
        depth -= 1

    n = gen_moves(c.bb, c.sq, c.st, c.ml, ply, I0)
    if n == 0:
        return -MATE + ply if checked else draw_score(c)
    score_moves(c, ply, n, tt_move)

    best = -INF
    best_move = 0
    played = I0
    nquiet = 0
    old_alpha = alpha
    lmp_limit = LMP[1 if improving else 0, depth if depth < 12 else 11]

    for i in range(n):
        m = pick_move(c, ply, i, n)
        noisy = is_noisy(c, m)
        frm = m_from(m)
        to = m_to(m)
        hist_score = I0 if noisy else np.int64(c.hist[c.st[ST_SIDE], frm, to])

        if not pv_node and played > 0 and best > -MATE_IN_MAX and \
                skip_move(c, ply, m, depth, alpha, static, played, lmp_limit, noisy,
                          checked, hist_score):
            continue

        c.stack[ply, SK_PIECE] = c.sq[frm]
        c.stack[ply, SK_TO] = to
        c.stack[ply, SK_MOVE] = m
        do_move(c.bb, c.sq, c.st, c.und, ply, m)
        played += 1
        if not noisy and nquiet < 63:
            c.qbuf[ply, nquiet] = m
            nquiet += 1

        checks = in_check(c.bb, c.st)
        new_depth = depth - 1
        if checks and depth <= 10:
            new_depth += 1

        # The first move of a principal variation gets a full window at full depth.
        # Everything else is offered a reduced zero window first and only re-searched
        # if it beats alpha, which is what makes alpha-beta cheap.
        r = 0
        if depth >= 3 and played > (2 if pv_node else 1) and not noisy:
            r = reduction(depth, played, pv_node, improving, cut_node, checks, hist_score)
        d = new_depth - r
        if d < 1:
            d = 1
        elif d > new_depth:
            d = new_depth

        if pv_node and played == 1:
            v = -negamax(c, ply + 1, new_depth, -beta, -alpha, FALSE)
        else:
            v = -negamax(c, ply + 1, d, -alpha - 1, -alpha, TRUE if r > 0 else not cut_node)
            if v > alpha and (d < new_depth or (pv_node and v < beta)):
                lo = -beta if pv_node else -alpha - 1
                v = -negamax(c, ply + 1, new_depth, lo, -alpha, FALSE)

        undo_move(c.bb, c.sq, c.st, c.und, ply)
        if c.ctl[C_STOP] != 0:
            return 0

        if v > best:
            best = v
            best_move = m
            if v > alpha:
                alpha = v
                if pv_node:
                    copy_pv(c, ply, m)
                if v >= beta:
                    if not noisy:
                        update_quiet_history(c, ply, m, depth, nquiet - 1)
                    break

    bound = BOUND_LOWER if best >= beta else (
        BOUND_EXACT if best > old_alpha else BOUND_UPPER)
    tt_store(c, key, best_move, best, raw_eval, depth, bound, ply)
    return best


@njit(nogil=True, cache=False, error_model="numpy")
def search_root(c, depth, alpha, beta, n):
    """One iteration over the root move list, which stays ordered between iterations."""
    best = -INF
    best_move = c.ml[0, 0]
    c.pvlen[I0] = 0
    checked = in_check(c.bb, c.st)

    for i in range(n):
        m = c.ml[0, i]
        c.stack[I0, SK_PIECE] = c.sq[m_from(m)]
        c.stack[I0, SK_TO] = m_to(m)
        c.stack[I0, SK_MOVE] = m
        before = c.ctl[C_NODES]
        do_move(c.bb, c.sq, c.st, c.und, I0, m)
        new_depth = depth - 1
        if not checked and depth <= 10 and in_check(c.bb, c.st):
            new_depth += 1
        if i == 0:
            v = -negamax(c, I1, new_depth, -beta, -alpha, FALSE)
        else:
            r = 0
            if depth >= 3 and i >= 3 and not is_noisy(c, m):
                r = LMR[depth if depth < 64 else 63, i if i < 64 else 63] - 1
                if r < 0:
                    r = 0
            v = -negamax(c, I1, new_depth - r, -alpha - 1, -alpha, TRUE)
            if v > alpha:
                v = -negamax(c, I1, new_depth, -beta, -alpha, FALSE)
        undo_move(c.bb, c.sq, c.st, c.und, I0)

        if c.ctl[C_STOP] != 0:
            break

        # For the next iteration, order by how much work each move cost. A move that
        # needed a large subtree is one the opponent has resources against.
        spent = (c.ctl[C_NODES] - before) // 64
        c.ms[0, i] = spent if spent < 1_000_000_000 else 1_000_000_000

        if v > best:
            best = v
            best_move = m
            copy_pv(c, I0, m)
            if v > alpha:
                alpha = v
                if v >= beta:
                    break

    if c.ctl[C_STOP] == 0:
        c.ctl[C_BESTMOVE] = best_move
        c.ctl[C_BESTSCORE] = best
        for i in range(n):
            if c.ml[0, i] == best_move:
                c.ms[0, i] = 2_000_000_000
        for i in range(n):
            pick_move(c, I0, i, n)
    return best


@njit(nogil=True, cache=False, error_model="numpy")
def root_moves(c):
    return gen_moves(c.bb, c.sq, c.st, c.ml, I0, I0)


@njit(nogil=True, cache=False, error_model="numpy")
def new_search(c):
    """Between our own moves: keep the table, halve the history, clear the killers."""
    c.ctl[C_NODES] = 0
    c.ctl[C_STOP] = 0
    c.ctl[C_SOFT] = 0
    c.ctl[C_SELDEPTH] = 0
    c.ctl[C_LIMIT] = np.int64(1) << np.int64(60)
    c.ctl[C_ROOTSIDE] = c.st[ST_SIDE]
    c.ctl[C_AGE] = (c.ctl[C_AGE] + 1) & 63
    for i in range(MAXPLY + 8):
        c.killers[i, 0] = 0
        c.killers[i, 1] = 0
        c.stack[i, SK_EVAL] = NO_EVAL
        c.stack[i, SK_PIECE] = NO_PIECE
        c.stack[i, SK_TO] = 0
        c.stack[i, SK_MOVE] = 0
    for s in range(2):
        for a in range(64):
            for b in range(64):
                c.hist[s, a, b] //= 2


@njit(nogil=True, cache=False, error_model="numpy")
def static_eval(c):
    return evaluate(c.bb, c.sq, c.st, c.P, c.sc, c.ptt)


@njit(nogil=True, cache=False, error_model="numpy")
def bench_eval(c, reps):
    total = 0
    for _ in range(reps):
        total += evaluate(c.bb, c.sq, c.st, c.P, c.sc, c.ptt)
    return total


@njit(nogil=True, cache=False, error_model="numpy")
def bench_movegen(c, reps):
    total = 0
    for _ in range(reps):
        total += gen_moves(c.bb, c.sq, c.st, c.ml, I0, I0)
    return total
