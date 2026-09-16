"""Tapered evaluation: material, piece-square tables, mobility, pawn structure, king
safety, threats, and endgame scaling.

Every weight comes out of the flat vector in nx_params and the function is otherwise pure,
so scripts/tune.py can fit the whole thing without the evaluation knowing a tuner exists.

The evaluation is deliberately split into one function per term group rather than written
as one long routine. numba's compile time grows much faster than linearly with the size of
a single function, and the whole engine has to compile inside a 90 second init budget.

Scratch space is passed in rather than allocated. This runs millions of times a move, and
a single np.zeros in here would cost more than everything else put together.
"""

import numpy as np
from numba import njit

from nx_core import (
    BISHOP, BLACK, DIST, FILE_BB, KING, KING_ATT, KNIGHT, KNIGHT_ATT, OCC_A, OCC_W, PAWN,
    PAWN_ATT, QUEEN, RANK_BB, ROOK, ST_PKEY, ST_SIDE, WHITE, bishop_attacks, bit, lsb,
    popcount, queen_attacks, rook_attacks, shr,
)
from nx_params import (
    I_BACKWARD, I_BAD_BISHOP, I_BISHOP_PAIR, I_CONNECTED, I_DOUBLED, I_ISOLATED,
    I_KING_ATT, I_KING_OPEN, I_KING_QUAD, I_MATERIAL, I_MINOR_SHIELD, I_MOBILITY,
    I_OUTPOST_B, I_OUTPOST_N, I_PASSED, I_PASSED_FREE, I_PASSED_KING, I_PASSED_PROT,
    I_PSQT, I_ROOK_OPEN, I_ROOK_SEMI, I_ROOK_SEVENTH, I_SAFE_CHECK, I_SHELTER, I_STORM,
    I_TEMPO, I_THREAT_HANGING, I_THREAT_KING, I_THREAT_MINOR_MAJOR, I_THREAT_PAWN_MAJOR,
    I_THREAT_PAWN_MINOR, I_THREAT_ROOK_QUEEN,
)

# ---- geometry tables -------------------------------------------------------------------

_adj = np.zeros(8, dtype=np.int64)
for _f in range(8):
    if _f > 0:
        _adj[_f] |= FILE_BB[_f - 1]
    if _f < 7:
        _adj[_f] |= FILE_BB[_f + 1]
ADJ_FILES = _adj

_fwd = np.zeros((2, 8), dtype=np.int64)
for _r in range(8):
    for _rr in range(8):
        if _rr > _r:
            _fwd[0, _r] |= RANK_BB[_rr]
        if _rr < _r:
            _fwd[1, _r] |= RANK_BB[_rr]
FORWARD_RANKS = _fwd

_front = np.zeros((2, 64), dtype=np.int64)
_passed = np.zeros((2, 64), dtype=np.int64)
_kz = np.zeros((2, 64), dtype=np.int64)
for _c in range(2):
    for _sq in range(64):
        _f, _r = _sq & 7, _sq >> 3
        _front[_c, _sq] = FORWARD_RANKS[_c, _r] & FILE_BB[_f]
        _passed[_c, _sq] = FORWARD_RANKS[_c, _r] & (FILE_BB[_f] | ADJ_FILES[_f])
        _base = int(KING_ATT[_sq]) | (1 << _sq)
        _ahead = (_base << 8) if _c == 0 else (_base >> 8)
        _kz[_c, _sq] = np.int64((_base | _ahead) & 0xFFFFFFFFFFFFFFFF
                                if ((_base | _ahead) & 0xFFFFFFFFFFFFFFFF) < (1 << 63)
                                else ((_base | _ahead) & 0xFFFFFFFFFFFFFFFF) - (1 << 64))
FRONT_FILE = _front
PASSED_MASK = _passed
KING_ZONE = _kz

DARK = np.int64(-6172840429334713771)   # 0xAA55AA55AA55AA55 with the sign bit taken as data
LIGHT = ~DARK

PHASE_W = np.array([0, 1, 1, 2, 4, 0], dtype=np.int64)
EVAL_CAP = 20000
TOTAL_PHASE = 24

# scratch slots, shared by the term functions
S_ATT = 0        # 12 entries, indexed [side * 6 + piece type]
S_ALL = 12       # 2 entries: every square a side attacks
S_DBL = 14       # 2 entries: squares a side attacks at least twice
S_AREA = 16      # 2 entries: where mobility counts
S_KZONE = 18     # 2 entries
S_KATT = 20      # 2 entries: accumulated danger around each king
S_KCNT = 22      # 2 entries: how many enemy pieces join the attack
S_N = 24


@njit(cache=False, error_model="numpy")
def eval_pawns(bb, P):
    """White minus black pawn structure, and each side's passed pawns.

    Depends only on the pawns, so the caller memoises it on the pawn hash key.
    """
    mg = 0
    eg = 0
    pw = np.int64(0)
    pb = np.int64(0)
    for side in range(2):
        sign = 1 if side == WHITE else -1
        own = bb[side * 6 + PAWN]
        opp = bb[(1 - side) * 6 + PAWN]
        pawns = own
        while pawns != 0:
            sq = lsb(pawns)
            pawns &= pawns - 1
            f = sq & 7
            r = sq >> 3
            rr = r if side == WHITE else 7 - r
            neighbours = own & ADJ_FILES[f]
            phalanx = neighbours & RANK_BB[r]
            support = own & PAWN_ATT[1 - side, sq]
            ahead_own = own & FRONT_FILE[side, sq]

            if neighbours == 0:
                mg += sign * P[I_ISOLATED]
                eg += sign * P[I_ISOLATED + 1]
            elif support == 0 and phalanx == 0:
                stop = sq + 8 if side == WHITE else sq - 8
                if (neighbours & ~FORWARD_RANKS[side, r]) == 0 and 0 <= stop < 64 \
                        and (PAWN_ATT[side, stop] & opp) != 0:
                    mg += sign * P[I_BACKWARD]
                    eg += sign * P[I_BACKWARD + 1]

            if ahead_own != 0:
                mg += sign * P[I_DOUBLED]
                eg += sign * P[I_DOUBLED + 1]

            if support != 0 or phalanx != 0:
                mg += sign * P[I_CONNECTED + rr * 2]
                eg += sign * P[I_CONNECTED + rr * 2 + 1]

            if (PASSED_MASK[side, sq] & opp) == 0 and ahead_own == 0:
                if side == WHITE:
                    pw |= bit(sq)
                else:
                    pb |= bit(sq)
                mg += sign * P[I_PASSED + rr * 2]
                eg += sign * P[I_PASSED + rr * 2 + 1]
                if support != 0:
                    mg += sign * P[I_PASSED_PROT]
                    eg += sign * P[I_PASSED_PROT + 1]
    return mg, eg, pw, pb


@njit(cache=False, error_model="numpy")
def _material_psqt(bb, P):
    mg = 0
    eg = 0
    phase = 0
    for side in range(2):
        sign = 1 if side == WHITE else -1
        for pt in range(6):
            pieces = bb[side * 6 + pt]
            phase += PHASE_W[pt] * popcount(pieces)
            while pieces != 0:
                sq = lsb(pieces)
                pieces &= pieces - 1
                rsq = sq if side == WHITE else sq ^ 56
                mg += sign * P[I_PSQT + (pt * 64 + rsq) * 2]
                eg += sign * P[I_PSQT + (pt * 64 + rsq) * 2 + 1]
                if pt != KING:
                    mg += sign * P[I_MATERIAL + pt * 2]
                    eg += sign * P[I_MATERIAL + pt * 2 + 1]
    return mg, eg, phase if phase < TOTAL_PHASE else TOTAL_PHASE


@njit(cache=False, error_model="numpy")
def _seed_attacks(bb, sc, wk, bk):
    """Pawn and king attack sets, which everything else is measured against."""
    for i in range(S_N):
        sc[i] = 0
    for side in range(2):
        pawns = bb[side * 6 + PAWN]
        if side == WHITE:
            left = (pawns & ~FILE_BB[0]) << 7
            right = (pawns & ~FILE_BB[7]) << 9
        else:
            left = shr(pawns & ~FILE_BB[0], 9)
            right = shr(pawns & ~FILE_BB[7], 7)
        ksq = wk if side == WHITE else bk
        sc[S_ATT + side * 6 + PAWN] = left | right
        sc[S_ATT + side * 6 + KING] = KING_ATT[ksq]
        sc[S_DBL + side] = (left & right) | ((left | right) & KING_ATT[ksq])
        sc[S_ALL + side] = left | right | KING_ATT[ksq]
        sc[S_KZONE + side] = KING_ZONE[side, ksq]


@njit(cache=False, error_model="numpy")
def _pieces(bb, P, sc, occ):
    """Mobility, attack maps, and the per-piece positional terms."""
    mg = 0
    eg = 0
    for side in range(2):
        sign = 1 if side == WHITE else -1
        them = 1 - side
        own_pawns = bb[side * 6 + PAWN]
        blocked = own_pawns & (shr(occ, 8) if side == WHITE else (occ << 8))
        sc[S_AREA + side] = ~(sc[S_ATT + them * 6 + PAWN] | blocked | bb[side * 6 + KING])
        ekz = sc[S_KZONE + them]
        for pt in range(KNIGHT, KING):
            pieces = bb[side * 6 + pt]
            while pieces != 0:
                sq = lsb(pieces)
                pieces &= pieces - 1
                if pt == KNIGHT:
                    att = KNIGHT_ATT[sq]
                elif pt == BISHOP:
                    att = bishop_attacks(sq, occ ^ bb[side * 6 + QUEEN] ^ bb[side * 6 + BISHOP])
                elif pt == ROOK:
                    att = rook_attacks(sq, occ ^ bb[side * 6 + QUEEN] ^ bb[side * 6 + ROOK])
                else:
                    att = queen_attacks(sq, occ)
                sc[S_DBL + side] |= sc[S_ALL + side] & att
                sc[S_ALL + side] |= att
                sc[S_ATT + side * 6 + pt] |= att
                m = popcount(att & sc[S_AREA + side])
                mg += sign * P[I_MOBILITY + ((pt - 1) * 28 + m) * 2]
                eg += sign * P[I_MOBILITY + ((pt - 1) * 28 + m) * 2 + 1]
                if (att & ekz) != 0:
                    sc[S_KATT + them] += P[I_KING_ATT + pt] * popcount(att & ekz)
                    sc[S_KCNT + them] += 1

                if pt == ROOK:
                    f = sq & 7
                    if (own_pawns & FILE_BB[f]) == 0:
                        idx = I_ROOK_OPEN if (bb[them * 6 + PAWN] & FILE_BB[f]) == 0 \
                            else I_ROOK_SEMI
                        mg += sign * P[idx]
                        eg += sign * P[idx + 1]
                    if (sq >> 3) == (6 if side == WHITE else 1):
                        mg += sign * P[I_ROOK_SEVENTH]
                        eg += sign * P[I_ROOK_SEVENTH + 1]
                elif pt == KNIGHT or pt == BISHOP:
                    rr = (sq >> 3) if side == WHITE else 7 - (sq >> 3)
                    if 3 <= rr <= 5 and (sc[S_ATT + side * 6 + PAWN] & bit(sq)) != 0 \
                            and (PASSED_MASK[side, sq] & ~FILE_BB[sq & 7]
                                 & bb[them * 6 + PAWN]) == 0:
                        idx = I_OUTPOST_N if pt == KNIGHT else I_OUTPOST_B
                        mg += sign * P[idx]
                        eg += sign * P[idx + 1]
                    ahead = sq + 8 if side == WHITE else sq - 8
                    if 0 <= ahead < 64 and (own_pawns & bit(ahead)) != 0:
                        mg += sign * P[I_MINOR_SHIELD]
                        eg += sign * P[I_MINOR_SHIELD + 1]
                    if pt == BISHOP:
                        n = popcount(own_pawns & (DARK if (bit(sq) & DARK) != 0 else LIGHT))
                        mg += sign * P[I_BAD_BISHOP] * n
                        eg += sign * P[I_BAD_BISHOP + 1] * n

        if popcount(bb[side * 6 + BISHOP]) >= 2:
            mg += sign * P[I_BISHOP_PAIR]
            eg += sign * P[I_BISHOP_PAIR + 1]
    return mg, eg


@njit(cache=False, error_model="numpy")
def _shelter(bb, P, side, ksq):
    """Pawn cover in front of the king, and enemy pawns rolling towards it."""
    own = bb[side * 6 + PAWN]
    opp = bb[(1 - side) * 6 + PAWN]
    kf = ksq & 7
    kr = ksq >> 3
    total = 0
    for f in range(max(0, kf - 1), min(7, kf + 1) + 1):
        d = abs(f - kf)
        for who in range(2):
            on_file = (own if who == 0 else opp) & FILE_BB[f] & FORWARD_RANKS[side, kr]
            if on_file != 0:
                near = lsb(on_file) if side == WHITE else 63 - lsb(_reverse(on_file))
                gap = abs((near >> 3) - kr)
                if gap > 7:
                    gap = 7
            else:
                gap = 7
            total += P[(I_SHELTER if who == 0 else I_STORM) + d * 8 + gap]
    return total


@njit(cache=False, error_model="numpy")
def _reverse(b):
    """Byte-reverse a bitboard so lsb finds the square nearest black's king."""
    b = ((b & 0x00FF00FF00FF00FF) << 8) | (shr(b, 8) & 0x00FF00FF00FF00FF)
    b = ((b & 0x0000FFFF0000FFFF) << 16) | (shr(b, 16) & 0x0000FFFF0000FFFF)
    return (b << 32) | shr(b, 32)


@njit(cache=False, error_model="numpy")
def _king_safety(bb, P, sc, occ, wk, bk):
    mg = 0
    eg = 0
    for side in range(2):
        sign = 1 if side == WHITE else -1
        them = 1 - side
        ksq = wk if side == WHITE else bk
        mg += sign * _shelter(bb, P, side, ksq)
        if sc[S_KCNT + side] < 2 or bb[them * 6 + QUEEN] == 0:
            continue
        danger = sc[S_KATT + side]
        weak = sc[S_ALL + them] & ~sc[S_DBL + side] & (
            ~sc[S_ALL + side] | sc[S_ATT + side * 6 + QUEEN] | sc[S_ATT + side * 6 + KING])
        safe = ~bb[OCC_W + them] & (~sc[S_ALL + side] | (weak & sc[S_DBL + them]))
        rk = rook_attacks(ksq, occ)
        bk_ = bishop_attacks(ksq, occ)
        if (rk & sc[S_ATT + them * 6 + ROOK] & safe) != 0:
            danger += P[I_SAFE_CHECK + 2]
        if ((rk | bk_) & sc[S_ATT + them * 6 + QUEEN] & safe) != 0:
            danger += P[I_SAFE_CHECK + 3]
        if (bk_ & sc[S_ATT + them * 6 + BISHOP] & safe) != 0:
            danger += P[I_SAFE_CHECK + 1]
        if (KNIGHT_ATT[ksq] & sc[S_ATT + them * 6 + KNIGHT] & safe) != 0:
            danger += P[I_SAFE_CHECK]
        kf = ksq & 7
        for f in range(max(0, kf - 1), min(7, kf + 1) + 1):
            if (bb[side * 6 + PAWN] & FILE_BB[f]) == 0:
                danger -= P[I_KING_OPEN if (bb[them * 6 + PAWN] & FILE_BB[f]) == 0
                            else I_KING_OPEN + 2]
        if danger > 0:
            mg -= sign * (P[I_KING_QUAD] * danger // 16
                          + P[I_KING_QUAD + 1] * danger * danger // 2048)
            eg -= sign * (P[I_KING_QUAD] * danger // 48)
    return mg, eg


@njit(cache=False, error_model="numpy")
def _threats(bb, P, sc):
    mg = 0
    eg = 0
    for side in range(2):
        sign = 1 if side == WHITE else -1
        them = 1 - side
        minors = bb[them * 6 + KNIGHT] | bb[them * 6 + BISHOP]
        majors = bb[them * 6 + ROOK] | bb[them * 6 + QUEEN]
        pawn_att = sc[S_ATT + side * 6 + PAWN]
        minor_att = sc[S_ATT + side * 6 + KNIGHT] | sc[S_ATT + side * 6 + BISHOP]
        hanging = (bb[OCC_W + them] & ~bb[them * 6 + PAWN]
                   & sc[S_ALL + side] & ~sc[S_ALL + them])
        for idx, n in ((I_THREAT_PAWN_MINOR, popcount(pawn_att & minors)),
                       (I_THREAT_PAWN_MAJOR, popcount(pawn_att & majors)),
                       (I_THREAT_MINOR_MAJOR, popcount(minor_att & majors)),
                       (I_THREAT_ROOK_QUEEN,
                        popcount(sc[S_ATT + side * 6 + ROOK] & bb[them * 6 + QUEEN])),
                       (I_THREAT_HANGING, popcount(hanging)),
                       (I_THREAT_KING, popcount(sc[S_ATT + side * 6 + KING]
                                                & bb[OCC_W + them] & ~sc[S_ALL + them]))):
            mg += sign * P[idx] * n
            eg += sign * P[idx + 1] * n
    return mg, eg


@njit(cache=False, error_model="numpy")
def _passers(P, occ, pw, pb, wk, bk):
    """The part of a passed pawn's value that depends on where the pieces are."""
    mg = 0
    eg = 0
    for side in range(2):
        sign = 1 if side == WHITE else -1
        p = pw if side == WHITE else pb
        own_k = wk if side == WHITE else bk
        their_k = bk if side == WHITE else wk
        while p != 0:
            sq = lsb(p)
            p &= p - 1
            rr = (sq >> 3) if side == WHITE else 7 - (sq >> 3)
            if (FRONT_FILE[side, sq] & occ) == 0:
                mg += sign * P[I_PASSED_FREE] * rr // 6
                eg += sign * P[I_PASSED_FREE + 1] * rr // 6
            stop = sq + 8 if side == WHITE else sq - 8
            if 0 <= stop < 64:
                eg += sign * P[I_PASSED_KING + 1] * rr * (
                    DIST[their_k, stop] - DIST[own_k, stop]) // 6
    return mg, eg


@njit(cache=False, error_model="numpy")
def _scale_factor(bb, eg):
    """Pull scores towards a draw when the stronger side cannot force a win."""
    strong = WHITE if eg > 0 else BLACK
    weak = 1 - strong
    if bb[strong * 6 + PAWN] == 0:
        diff = (popcount(bb[strong * 6 + KNIGHT]) * 3 + popcount(bb[strong * 6 + BISHOP]) * 3
                + popcount(bb[strong * 6 + ROOK]) * 5 + popcount(bb[strong * 6 + QUEEN]) * 9
                - popcount(bb[weak * 6 + KNIGHT]) * 3 - popcount(bb[weak * 6 + BISHOP]) * 3
                - popcount(bb[weak * 6 + ROOK]) * 5 - popcount(bb[weak * 6 + QUEEN]) * 9)
        if diff <= 1:
            return 4
        if diff <= 3:
            return 20
    if (popcount(bb[WHITE * 6 + BISHOP]) == 1 and popcount(bb[BLACK * 6 + BISHOP]) == 1
            and ((bb[WHITE * 6 + BISHOP] & DARK) != 0)
            != ((bb[BLACK * 6 + BISHOP] & DARK) != 0)
            and (bb[WHITE * 6 + ROOK] | bb[BLACK * 6 + ROOK] | bb[WHITE * 6 + QUEEN]
                 | bb[BLACK * 6 + QUEEN] | bb[WHITE * 6 + KNIGHT]
                 | bb[BLACK * 6 + KNIGHT]) == 0):
        return 24
    return 64


@njit(cache=False, error_model="numpy")
def evaluate(bb, sq_of, st, P, sc, ptt):
    """Score in centipawns from the point of view of the side to move."""
    wk = lsb(bb[WHITE * 6 + KING])
    bk = lsb(bb[BLACK * 6 + KING])
    occ = bb[OCC_A]

    mg, eg, phase = _material_psqt(bb, P)

    # An empty slot reads as key 0, which is exactly the key of a position with no pawns,
    # and the zeroed payload is the right answer for one. So no validity flag is needed.
    pkey = st[ST_PKEY]
    slot = (pkey ^ shr(pkey, 32)) & (ptt.shape[0] - 1)
    if ptt[slot, 0] == pkey:
        pmg, peg, pw, pb = ptt[slot, 1], ptt[slot, 2], ptt[slot, 3], ptt[slot, 4]
    else:
        pmg, peg, pw, pb = eval_pawns(bb, P)
        ptt[slot, 0] = pkey
        ptt[slot, 1] = pmg
        ptt[slot, 2] = peg
        ptt[slot, 3] = pw
        ptt[slot, 4] = pb
    mg += pmg
    eg += peg

    _seed_attacks(bb, sc, wk, bk)
    a, b = _pieces(bb, P, sc, occ)
    mg += a
    eg += b
    a, b = _king_safety(bb, P, sc, occ, wk, bk)
    mg += a
    eg += b
    a, b = _threats(bb, P, sc)
    mg += a
    eg += b
    a, b = _passers(P, occ, pw, pb, wk, bk)
    mg += a
    eg += b

    score = (mg * phase + eg * (TOTAL_PHASE - phase)) // TOTAL_PHASE
    score += P[I_TEMPO] if st[ST_SIDE] == WHITE else -P[I_TEMPO]
    scale = _scale_factor(bb, eg)
    if scale != 64:
        score = score * scale // 64
    # The transposition table packs a score into 16 bits, and a position with several
    # promoted queens on it can otherwise run past that and corrupt the entry.
    if score > EVAL_CAP:
        score = EVAL_CAP
    elif score < -EVAL_CAP:
        score = -EVAL_CAP
    return score if st[ST_SIDE] == WHITE else -score


@njit(cache=False, error_model="numpy")
def insufficient_material(bb):
    """Positions FIDE calls dead drawn, so the search never tries to win them."""
    if bb[PAWN] != 0 or bb[6 + PAWN] != 0:
        return False
    if bb[ROOK] != 0 or bb[6 + ROOK] != 0 or bb[QUEEN] != 0 or bb[6 + QUEEN] != 0:
        return False
    knights = bb[KNIGHT] | bb[6 + KNIGHT]
    bishops = bb[BISHOP] | bb[6 + BISHOP]
    if knights != 0:
        return bishops == 0 and popcount(knights) <= 1
    # Any number of bishops is dead only when every bishop is confined to the
    # same colour complex. Opposite-coloured bishops can help construct a mate.
    return bishops == 0 or (bishops & DARK) == 0 or (bishops & LIGHT) == 0
