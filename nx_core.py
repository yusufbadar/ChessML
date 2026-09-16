"""Bitboard core: tables, position state, legal move generation, make and unmake.

Everything here is compiled by numba in nopython mode, so the style is deliberately flat:
integers and numpy arrays, no objects, no exceptions on the hot path.

Board geometry follows python-chess, so square 0 is a1 and square 63 is h8. Bitboards are
int64 read as unsigned, which means every logical right shift goes through shr(): numba's
>> on a signed integer propagates the sign bit, and shr masks that off.
"""

import numpy as np
from numba import njit

import nx_magic

# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------

WHITE, BLACK = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5
NO_PIECE = 12
OCC_W, OCC_B, OCC_A = 12, 13, 14

MAXPLY = 128
MAXMOVES = 256

# move flags, held in bits 14-15 of a move
F_NORMAL, F_PROMO, F_EP, F_CASTLE = 0, 1, 2, 3

# slots of the scalar state vector
ST_SIDE, ST_CASTLE, ST_EP, ST_R50, ST_KEY, ST_PKEY, ST_GPLY, ST_GLEN = 0, 1, 2, 3, 4, 5, 6, 7
ST_N = 8

# slots of one undo record
U_CASTLE, U_EP, U_R50, U_KEY, U_PKEY, U_CAPT, U_MOVED, U_MOVE = 0, 1, 2, 3, 4, 5, 6, 7
U_N = 8

# castling rights bits
CR_WK, CR_WQ, CR_BK, CR_BQ = 1, 2, 4, 8

# --------------------------------------------------------------------------------------
# bit helpers
# --------------------------------------------------------------------------------------

_r = np.zeros(65, dtype=np.int64)
_r[0] = -1
for _n in range(1, 64):
    _r[_n] = (1 << (64 - _n)) - 1
RMASK = _r  # RMASK[n] keeps the bits that survive a logical right shift by n

DEBRUIJN = 0x03F79D71B4CB0A89
_db = np.zeros(64, dtype=np.int64)
for _i in range(64):
    _db[((1 << _i) * DEBRUIJN & 0xFFFFFFFFFFFFFFFF) >> 58] = _i
DEBRUIJN_IDX = _db


def _s64(v):
    """Reinterpret an unsigned 64-bit literal as the signed integer with the same bits."""
    return v - (1 << 64) if v >> 63 else v


@njit(inline="always", cache=False, error_model="numpy")
def shr(b, n):
    """Logical right shift of a bitboard held in a signed 64-bit integer."""
    return (b >> n) & RMASK[n]


@njit(inline="always", cache=False, error_model="numpy")
def popcount(b):
    b = b - (shr(b, 1) & 0x5555555555555555)
    b = (b & 0x3333333333333333) + (shr(b, 2) & 0x3333333333333333)
    b = (b + shr(b, 4)) & 0x0F0F0F0F0F0F0F0F
    return shr(b * 0x0101010101010101, 56) & 0xFF


@njit(inline="always", cache=False, error_model="numpy")
def lsb(b):
    """Index of the least significant set bit. Undefined for an empty board."""
    return DEBRUIJN_IDX[shr((b & -b) * DEBRUIJN, 58)]


@njit(inline="always", cache=False, error_model="numpy")
def bit(sq):
    return np.int64(1) << sq


@njit(inline="always", cache=False, error_model="numpy")
def more_than_one(b):
    return (b & (b - 1)) != 0


# --------------------------------------------------------------------------------------
# move encoding
# --------------------------------------------------------------------------------------


@njit(inline="always", cache=False, error_model="numpy")
def mk_move(frm, to, promo, flag):
    return frm | (to << 6) | (promo << 12) | (flag << 14)


@njit(inline="always", cache=False, error_model="numpy")
def m_from(m):
    return m & 63


@njit(inline="always", cache=False, error_model="numpy")
def m_to(m):
    return (m >> 6) & 63


@njit(inline="always", cache=False, error_model="numpy")
def m_promo(m):
    """0..3 meaning knight, bishop, rook, queen. Meaningful only when the flag is F_PROMO."""
    return (m >> 12) & 3


@njit(inline="always", cache=False, error_model="numpy")
def m_flag(m):
    return (m >> 14) & 3


# --------------------------------------------------------------------------------------
# static tables
# --------------------------------------------------------------------------------------

FILE_BB = np.array([_s64(0x0101010101010101 << f) for f in range(8)], dtype=np.int64)
RANK_BB = np.array([_s64(0xFF << (8 * r)) for r in range(8)], dtype=np.int64)

KNIGHT_ATT = np.zeros(64, dtype=np.int64)
KING_ATT = np.zeros(64, dtype=np.int64)
PAWN_ATT = np.zeros((2, 64), dtype=np.int64)
BETWEEN = np.zeros((64, 64), dtype=np.int64)
LINE = np.zeros((64, 64), dtype=np.int64)
DIST = np.zeros((64, 64), dtype=np.int64)

ROOK_MASK = np.array(nx_magic.ROOK_MASK, dtype=np.int64)
ROOK_MAGIC = np.array(nx_magic.ROOK_MAGIC, dtype=np.int64)
ROOK_SHIFT = np.array([64 - b for b in nx_magic.ROOK_BITS], dtype=np.int64)
ROOK_OFFSET = np.array(nx_magic.ROOK_OFFSET, dtype=np.int64)
BISHOP_MASK = np.array(nx_magic.BISHOP_MASK, dtype=np.int64)
BISHOP_MAGIC = np.array(nx_magic.BISHOP_MAGIC, dtype=np.int64)
BISHOP_SHIFT = np.array([64 - b for b in nx_magic.BISHOP_BITS], dtype=np.int64)
BISHOP_OFFSET = np.array(nx_magic.BISHOP_OFFSET, dtype=np.int64)
ROOK_ATT = np.zeros(102400, dtype=np.int64)
BISHOP_ATT = np.zeros(5248, dtype=np.int64)

DR = np.array([1, -1, 0, 0, 1, 1, -1, -1], dtype=np.int64)
DF = np.array([0, 0, 1, -1, 1, -1, 1, -1], dtype=np.int64)


@njit(cache=False, error_model="numpy")
def _slider_attacks(sq, occ, rook, dr, df):
    """Attacks from sq along four rays, stopping on and including the first blocker."""
    res = np.int64(0)
    r0 = sq >> 3
    f0 = sq & 7
    lo = 0 if rook else 4
    for d in range(lo, lo + 4):
        r = r0 + dr[d]
        f = f0 + df[d]
        while 0 <= r < 8 and 0 <= f < 8:
            t = r * 8 + f
            res |= bit(t)
            if (occ >> t) & 1:
                break
            r += dr[d]
            f += df[d]
    return res


@njit(cache=False, error_model="numpy")
def _build_tables(knight, king, pawn, between, line, dist, dr, df,
                  rmask, rmagic, rshift, roff, rtab,
                  bmask, bmagic, bshift, boff, btab):
    km = np.array([17, 15, 10, 6, -6, -10, -15, -17], dtype=np.int64)
    kd = np.array([1, 1, 2, 2, 2, 2, 1, 1], dtype=np.int64)
    gm = np.array([8, 9, 1, -7, -8, -9, -1, 7], dtype=np.int64)
    gd = np.array([0, 1, 1, 1, 0, 1, 1, 1], dtype=np.int64)
    for sq in range(64):
        f0 = sq & 7
        for i in range(8):
            t = sq + km[i]
            if 0 <= t < 64 and abs((t & 7) - f0) == kd[i]:
                knight[sq] |= bit(t)
            t = sq + gm[i]
            if 0 <= t < 64 and abs((t & 7) - f0) == gd[i]:
                king[sq] |= bit(t)
        r0 = sq >> 3
        if r0 < 7:
            if f0 > 0:
                pawn[0, sq] |= bit(sq + 7)
            if f0 < 7:
                pawn[0, sq] |= bit(sq + 9)
        if r0 > 0:
            if f0 > 0:
                pawn[1, sq] |= bit(sq - 9)
            if f0 < 7:
                pawn[1, sq] |= bit(sq - 7)

    for a in range(64):
        for b in range(64):
            dist[a, b] = max(abs((a >> 3) - (b >> 3)), abs((a & 7) - (b & 7)))
        for rook in range(2):
            empty = _slider_attacks(a, np.int64(0), rook == 1, dr, df)
            walk = empty
            while walk != 0:
                b = lsb(walk)
                walk &= walk - 1
                line[a, b] = (empty & _slider_attacks(b, np.int64(0), rook == 1, dr, df)) \
                    | bit(a) | bit(b)
                between[a, b] = (_slider_attacks(a, bit(b), rook == 1, dr, df)
                                 & _slider_attacks(b, bit(a), rook == 1, dr, df))

    for sq in range(64):
        for rook in range(2):
            mask = rmask[sq] if rook else bmask[sq]
            magic = rmagic[sq] if rook else bmagic[sq]
            shift = rshift[sq] if rook else bshift[sq]
            off = roff[sq] if rook else boff[sq]
            sub = np.int64(0)
            while True:
                att = _slider_attacks(sq, sub, rook == 1, dr, df)
                idx = shr(sub * magic, shift)
                if rook:
                    rtab[off + idx] = att
                else:
                    btab[off + idx] = att
                sub = (sub - mask) & mask
                if sub == 0:
                    break


_build_tables(KNIGHT_ATT, KING_ATT, PAWN_ATT, BETWEEN, LINE, DIST, DR, DF,
              ROOK_MASK, ROOK_MAGIC, ROOK_SHIFT, ROOK_OFFSET, ROOK_ATT,
              BISHOP_MASK, BISHOP_MAGIC, BISHOP_SHIFT, BISHOP_OFFSET, BISHOP_ATT)


@njit(inline="always", cache=False, error_model="numpy")
def rook_attacks(sq, occ):
    return ROOK_ATT[ROOK_OFFSET[sq] + shr((occ & ROOK_MASK[sq]) * ROOK_MAGIC[sq], ROOK_SHIFT[sq])]


@njit(inline="always", cache=False, error_model="numpy")
def bishop_attacks(sq, occ):
    return BISHOP_ATT[
        BISHOP_OFFSET[sq] + shr((occ & BISHOP_MASK[sq]) * BISHOP_MAGIC[sq], BISHOP_SHIFT[sq])
    ]


@njit(inline="always", cache=False, error_model="numpy")
def queen_attacks(sq, occ):
    return rook_attacks(sq, occ) | bishop_attacks(sq, occ)


# --------------------------------------------------------------------------------------
# zobrist keys, from our own seeded generator so nothing needs shipping as data
# --------------------------------------------------------------------------------------


def _xorshift(seed):
    x = seed
    while True:
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        yield _s64(x)


_rng = _xorshift(0x9E3779B97F4A7C15)
ZOB_PIECE = np.array([[next(_rng) for _ in range(64)] for _ in range(12)], dtype=np.int64)
ZOB_CASTLE = np.array([next(_rng) for _ in range(16)], dtype=np.int64)
ZOB_EP = np.array([next(_rng) for _ in range(8)], dtype=np.int64)
ZOB_SIDE = np.int64(next(_rng))

# castling: king origin, king target, rook origin, rook target, squares that must be empty
CASTLE_KING_TO = np.array([6, 2, 62, 58], dtype=np.int64)
CASTLE_ROOK_FROM = np.array([7, 0, 63, 56], dtype=np.int64)
CASTLE_ROOK_TO = np.array([5, 3, 61, 59], dtype=np.int64)
CASTLE_EMPTY = np.array([
    _s64((1 << 5) | (1 << 6)),
    _s64((1 << 1) | (1 << 2) | (1 << 3)),
    _s64((1 << 61) | (1 << 62)),
    _s64((1 << 57) | (1 << 58) | (1 << 59)),
], dtype=np.int64)
CASTLE_SAFE = np.array([
    _s64((1 << 4) | (1 << 5) | (1 << 6)),
    _s64((1 << 4) | (1 << 3) | (1 << 2)),
    _s64((1 << 60) | (1 << 61) | (1 << 62)),
    _s64((1 << 60) | (1 << 59) | (1 << 58)),
], dtype=np.int64)

# a move touching one of these squares clears the rights in CASTLE_CLEAR
_cc = np.full(64, 15, dtype=np.int64)
_cc[4] = 15 & ~(CR_WK | CR_WQ)
_cc[0] = 15 & ~CR_WQ
_cc[7] = 15 & ~CR_WK
_cc[60] = 15 & ~(CR_BK | CR_BQ)
_cc[56] = 15 & ~CR_BQ
_cc[63] = 15 & ~CR_BK
CASTLE_CLEAR = _cc

SEE_VALUE = np.array([100, 320, 330, 500, 950, 20000, 0], dtype=np.int64)


# --------------------------------------------------------------------------------------
# position queries
# --------------------------------------------------------------------------------------


@njit(cache=False, error_model="numpy")
def attackers_to(bb, sq, occ):
    """Every piece of either colour that attacks sq under the given occupancy."""
    return ((PAWN_ATT[WHITE, sq] & bb[BLACK * 6 + PAWN])
            | (PAWN_ATT[BLACK, sq] & bb[WHITE * 6 + PAWN])
            | (KNIGHT_ATT[sq] & (bb[KNIGHT] | bb[6 + KNIGHT]))
            | (KING_ATT[sq] & (bb[KING] | bb[6 + KING]))
            | (bishop_attacks(sq, occ) & (bb[BISHOP] | bb[6 + BISHOP] | bb[QUEEN] | bb[6 + QUEEN]))
            | (rook_attacks(sq, occ) & (bb[ROOK] | bb[6 + ROOK] | bb[QUEEN] | bb[6 + QUEEN])))


@njit(cache=False, error_model="numpy")
def attacked_by(bb, sq, by, occ):
    """True if the side `by` attacks sq. Cheaper than attackers_to because it short-circuits."""
    o = by * 6
    if PAWN_ATT[1 - by, sq] & bb[o + PAWN]:
        return True
    if KNIGHT_ATT[sq] & bb[o + KNIGHT]:
        return True
    if KING_ATT[sq] & bb[o + KING]:
        return True
    if bishop_attacks(sq, occ) & (bb[o + BISHOP] | bb[o + QUEEN]):
        return True
    if rook_attacks(sq, occ) & (bb[o + ROOK] | bb[o + QUEEN]):
        return True
    return False


@njit(cache=False, error_model="numpy")
def pinned_pieces(bb, ksq, us):
    """Own pieces that stand alone between our king and an enemy slider."""
    them = 1 - us
    o = them * 6
    occ = bb[OCC_A]
    ours = bb[OCC_W + us]
    res = np.int64(0)
    snipers = ((rook_attacks(ksq, np.int64(0)) & (bb[o + ROOK] | bb[o + QUEEN]))
               | (bishop_attacks(ksq, np.int64(0)) & (bb[o + BISHOP] | bb[o + QUEEN])))
    while snipers != 0:
        s = lsb(snipers)
        snipers &= snipers - 1
        blockers = BETWEEN[ksq, s] & occ
        if blockers != 0 and not more_than_one(blockers):
            res |= blockers & ours
    return res


@njit(cache=False, error_model="numpy")
def compute_key(bb, sq_of, st):
    key = np.int64(0)
    pkey = np.int64(0)
    for s in range(64):
        p = sq_of[s]
        if p != NO_PIECE:
            key ^= ZOB_PIECE[p, s]
            if p == PAWN or p == 6 + PAWN:
                pkey ^= ZOB_PIECE[p, s]
    key ^= ZOB_CASTLE[st[ST_CASTLE]]
    if st[ST_EP] >= 0:
        key ^= ZOB_EP[st[ST_EP] & 7]
    if st[ST_SIDE] == BLACK:
        key ^= ZOB_SIDE
    st[ST_KEY] = key
    st[ST_PKEY] = pkey
    return key


# --------------------------------------------------------------------------------------
# move generation
# --------------------------------------------------------------------------------------


@njit(cache=False, error_model="numpy")
def ep_is_legal(bb, frm, to, us, ksq):
    """En passant is the one move that can expose the king along a rank it never left."""
    cap = to - 8 if us == WHITE else to + 8
    occ = (bb[OCC_A] ^ bit(frm) ^ bit(cap)) | bit(to)
    o = (1 - us) * 6
    if rook_attacks(ksq, occ) & (bb[o + ROOK] | bb[o + QUEEN]):
        return False
    if bishop_attacks(ksq, occ) & (bb[o + BISHOP] | bb[o + QUEEN]):
        return False
    return True


@njit(inline="always", cache=False, error_model="numpy")
def _forward(b, us):
    """Shift a set of squares one rank towards the enemy back rank."""
    return (b << 8) if us == WHITE else shr(b, 8)


@njit(cache=False, error_model="numpy")
def _gen_king(bb, ml, ply, n, us, ksq, occ, ours, theirs, captures_only):
    """King moves, tested against the enemy attack set with the king lifted off the
    board so that sliders x-ray through the square it is leaving."""
    them = 1 - us
    occ_nk = occ ^ bit(ksq)
    moves = KING_ATT[ksq] & ~ours
    if captures_only != 0:
        moves &= theirs
    while moves != 0:
        t = lsb(moves)
        moves &= moves - 1
        if not attacked_by(bb, t, them, occ_nk):
            ml[ply, n] = mk_move(ksq, t, 0, F_NORMAL)
            n += 1
    return n


@njit(cache=False, error_model="numpy")
def _gen_pawns(bb, st, ml, ply, n, us, ksq, occ, theirs, legal, pinned, captures_only):
    """Pushes, double pushes, captures, promotions and en passant.

    Promotions are emitted even in a captures-only pass: a pawn reaching the last rank
    changes material as violently as any capture does.
    """
    empty = ~occ
    pawns = bb[us * 6 + PAWN]
    up = 8 if us == WHITE else -8
    rank3 = RANK_BB[2] if us == WHITE else RANK_BB[5]
    last = RANK_BB[7] if us == WHITE else RANK_BB[0]
    cap_mask = legal & theirs
    quiet_mask = legal & empty

    push1 = _forward(pawns, us) & empty
    if captures_only == 0:
        push2 = _forward(push1 & rank3, us) & quiet_mask
        while push2 != 0:
            t = lsb(push2)
            push2 &= push2 - 1
            f = t - 2 * up
            if not (pinned >> f) & 1 or (LINE[ksq, f] >> t) & 1:
                ml[ply, n] = mk_move(f, t, 0, F_NORMAL)
                n += 1
        quiet = push1 & quiet_mask & ~last
        while quiet != 0:
            t = lsb(quiet)
            quiet &= quiet - 1
            f = t - up
            if not (pinned >> f) & 1 or (LINE[ksq, f] >> t) & 1:
                ml[ply, n] = mk_move(f, t, 0, F_NORMAL)
                n += 1

    promo = push1 & quiet_mask & last
    while promo != 0:
        t = lsb(promo)
        promo &= promo - 1
        f = t - up
        if not (pinned >> f) & 1 or (LINE[ksq, f] >> t) & 1:
            for pr in range(4):
                ml[ply, n] = mk_move(f, t, pr, F_PROMO)
                n += 1

    for side in range(2):
        if side == 0:
            src = pawns & ~FILE_BB[0]
            d = 7 if us == WHITE else -9      # towards a lower file
        else:
            src = pawns & ~FILE_BB[7]
            d = 9 if us == WHITE else -7      # towards a higher file
        caps = ((src << d) if d > 0 else shr(src, -d)) & cap_mask
        pcaps = caps & last
        caps &= ~last
        while caps != 0:
            t = lsb(caps)
            caps &= caps - 1
            f = t - d
            if not (pinned >> f) & 1 or (LINE[ksq, f] >> t) & 1:
                ml[ply, n] = mk_move(f, t, 0, F_NORMAL)
                n += 1
        while pcaps != 0:
            t = lsb(pcaps)
            pcaps &= pcaps - 1
            f = t - d
            if not (pinned >> f) & 1 or (LINE[ksq, f] >> t) & 1:
                for pr in range(4):
                    ml[ply, n] = mk_move(f, t, pr, F_PROMO)
                    n += 1

    ep = st[ST_EP]
    if ep >= 0:
        capsq = ep - up
        # In check, en passant is only legal if it takes the checking pawn or blocks.
        if ((legal >> ep) & 1) != 0 or ((legal >> capsq) & 1) != 0:
            takers = PAWN_ATT[1 - us, ep] & pawns
            while takers != 0:
                f = lsb(takers)
                takers &= takers - 1
                if ((pinned >> f) & 1) != 0 and ((LINE[ksq, f] >> ep) & 1) == 0:
                    continue
                if ep_is_legal(bb, f, ep, us, ksq):
                    ml[ply, n] = mk_move(f, ep, 0, F_EP)
                    n += 1
    return n


@njit(cache=False, error_model="numpy")
def _gen_pieces(bb, ml, ply, n, us, ksq, occ, target, pinned):
    """Knights, bishops, rooks and queens, filtered against the pin mask."""
    for pt in range(KNIGHT, KING):
        pieces = bb[us * 6 + pt]
        while pieces != 0:
            f = lsb(pieces)
            pieces &= pieces - 1
            if pt == KNIGHT:
                att = KNIGHT_ATT[f]
            elif pt == BISHOP:
                att = bishop_attacks(f, occ)
            elif pt == ROOK:
                att = rook_attacks(f, occ)
            else:
                att = queen_attacks(f, occ)
            att &= target
            if (pinned >> f) & 1:
                att &= LINE[ksq, f]
            while att != 0:
                t = lsb(att)
                att &= att - 1
                ml[ply, n] = mk_move(f, t, 0, F_NORMAL)
                n += 1
    return n


@njit(cache=False, error_model="numpy")
def _gen_castling(bb, st, ml, ply, n, us, ksq, occ):
    them = 1 - us
    occ_nk = occ ^ bit(ksq)
    rights = st[ST_CASTLE]
    base = 0 if us == WHITE else 2
    for i in range(base, base + 2):
        if not (rights >> i) & 1:
            continue
        if occ & CASTLE_EMPTY[i]:
            continue
        path = CASTLE_SAFE[i]
        ok = True
        while path != 0:
            s = lsb(path)
            path &= path - 1
            if attacked_by(bb, s, them, occ_nk):
                ok = False
                break
        if ok:
            ml[ply, n] = mk_move(ksq, CASTLE_KING_TO[i], 0, F_CASTLE)
            n += 1
    return n


@njit(cache=False, error_model="numpy")
def gen_moves(bb, sq_of, st, ml, ply, captures_only):
    """Write every legal move for the side to move into ml[ply]. Returns how many.

    Generation is legal, not pseudo-legal: king destinations are tested against the
    enemy attack set, other pieces are filtered against the pin mask, and in check the
    target set is restricted to blocks and captures of the checker. Nothing downstream
    ever has to make a move to find out it was illegal.
    """
    us = st[ST_SIDE]
    them = 1 - us
    occ = bb[OCC_A]
    ours = bb[OCC_W + us]
    theirs = bb[OCC_W + them]
    ksq = lsb(bb[us * 6 + KING])
    checkers = attackers_to(bb, ksq, occ) & theirs

    n = _gen_king(bb, ml, ply, 0, us, ksq, occ, ours, theirs, captures_only)
    if more_than_one(checkers):
        return n                      # double check: only the king may move

    evade = (BETWEEN[ksq, lsb(checkers)] | checkers) if checkers != 0 else np.int64(-1)
    legal = evade & ~ours
    pinned = pinned_pieces(bb, ksq, us)
    n = _gen_pawns(bb, st, ml, ply, n, us, ksq, occ, theirs, legal, pinned, captures_only)
    target = (legal & theirs) if captures_only != 0 else legal
    n = _gen_pieces(bb, ml, ply, n, us, ksq, occ, target, pinned)
    if captures_only == 0 and checkers == 0:
        n = _gen_castling(bb, st, ml, ply, n, us, ksq, occ)
    return n


# --------------------------------------------------------------------------------------
# make and unmake
# --------------------------------------------------------------------------------------


@njit(inline="always", cache=False, error_model="numpy")
def _put(bb, sq_of, p, s):
    bb[p] |= bit(s)
    bb[OCC_W + p // 6] |= bit(s)
    bb[OCC_A] |= bit(s)
    sq_of[s] = p


@njit(inline="always", cache=False, error_model="numpy")
def _clear(bb, sq_of, p, s):
    b = ~bit(s)
    bb[p] &= b
    bb[OCC_W + p // 6] &= b
    bb[OCC_A] &= b
    sq_of[s] = NO_PIECE


@njit(cache=False, error_model="numpy")
def do_move(bb, sq_of, st, und, ply, m):
    frm = m_from(m)
    to = m_to(m)
    flag = m_flag(m)
    us = st[ST_SIDE]
    them = 1 - us
    moved = sq_of[frm]
    captured = sq_of[to]
    key = st[ST_KEY]
    pkey = st[ST_PKEY]

    und[ply, U_CASTLE] = st[ST_CASTLE]
    und[ply, U_EP] = st[ST_EP]
    und[ply, U_R50] = st[ST_R50]
    und[ply, U_KEY] = key
    und[ply, U_PKEY] = pkey
    und[ply, U_MOVED] = moved
    und[ply, U_MOVE] = m

    if st[ST_EP] >= 0:
        key ^= ZOB_EP[st[ST_EP] & 7]
    key ^= ZOB_CASTLE[st[ST_CASTLE]]

    st[ST_R50] += 1
    st[ST_EP] = -1

    if flag == F_EP:
        capsq = to - 8 if us == WHITE else to + 8
        cp = them * 6 + PAWN
        _clear(bb, sq_of, cp, capsq)
        key ^= ZOB_PIECE[cp, capsq]
        pkey ^= ZOB_PIECE[cp, capsq]
        und[ply, U_CAPT] = NO_PIECE
        st[ST_R50] = 0
    elif captured != NO_PIECE:
        _clear(bb, sq_of, captured, to)
        key ^= ZOB_PIECE[captured, to]
        if captured == PAWN or captured == 6 + PAWN:
            pkey ^= ZOB_PIECE[captured, to]
        und[ply, U_CAPT] = captured
        st[ST_R50] = 0
    else:
        und[ply, U_CAPT] = NO_PIECE

    _clear(bb, sq_of, moved, frm)
    key ^= ZOB_PIECE[moved, frm]
    if moved == PAWN or moved == 6 + PAWN:
        pkey ^= ZOB_PIECE[moved, frm]
        st[ST_R50] = 0
        if (to - frm) * (to - frm) == 256:  # a double push, 16 squares either way
            st[ST_EP] = (frm + to) // 2

    if flag == F_PROMO:
        np_ = us * 6 + 1 + m_promo(m)
        _put(bb, sq_of, np_, to)
        key ^= ZOB_PIECE[np_, to]
    else:
        _put(bb, sq_of, moved, to)
        key ^= ZOB_PIECE[moved, to]
        if moved == PAWN or moved == 6 + PAWN:
            pkey ^= ZOB_PIECE[moved, to]

    if flag == F_CASTLE:
        idx = 0
        if to == 2:
            idx = 1
        elif to == 62:
            idx = 2
        elif to == 58:
            idx = 3
        rf = CASTLE_ROOK_FROM[idx]
        rt = CASTLE_ROOK_TO[idx]
        rp = us * 6 + ROOK
        _clear(bb, sq_of, rp, rf)
        _put(bb, sq_of, rp, rt)
        key ^= ZOB_PIECE[rp, rf] ^ ZOB_PIECE[rp, rt]

    st[ST_CASTLE] &= CASTLE_CLEAR[frm] & CASTLE_CLEAR[to]
    key ^= ZOB_CASTLE[st[ST_CASTLE]]
    if st[ST_EP] >= 0:
        key ^= ZOB_EP[st[ST_EP] & 7]
    key ^= ZOB_SIDE
    st[ST_SIDE] = them
    st[ST_KEY] = key
    st[ST_PKEY] = pkey


@njit(cache=False, error_model="numpy")
def undo_move(bb, sq_of, st, und, ply):
    m = und[ply, U_MOVE]
    frm = m_from(m)
    to = m_to(m)
    flag = m_flag(m)
    them = st[ST_SIDE]
    us = 1 - them
    moved = und[ply, U_MOVED]

    if flag == F_PROMO:
        _clear(bb, sq_of, us * 6 + 1 + m_promo(m), to)
    else:
        _clear(bb, sq_of, sq_of[to], to)
    _put(bb, sq_of, moved, frm)

    captured = und[ply, U_CAPT]
    if flag == F_EP:
        capsq = to - 8 if us == WHITE else to + 8
        _put(bb, sq_of, them * 6 + PAWN, capsq)
    elif captured != NO_PIECE:
        _put(bb, sq_of, captured, to)

    if flag == F_CASTLE:
        idx = 0
        if to == 2:
            idx = 1
        elif to == 62:
            idx = 2
        elif to == 58:
            idx = 3
        rp = us * 6 + ROOK
        _clear(bb, sq_of, rp, CASTLE_ROOK_TO[idx])
        _put(bb, sq_of, rp, CASTLE_ROOK_FROM[idx])

    st[ST_SIDE] = us
    st[ST_CASTLE] = und[ply, U_CASTLE]
    st[ST_EP] = und[ply, U_EP]
    st[ST_R50] = und[ply, U_R50]
    st[ST_KEY] = und[ply, U_KEY]
    st[ST_PKEY] = und[ply, U_PKEY]


@njit(cache=False, error_model="numpy")
def do_null(bb, sq_of, st, und, ply):
    und[ply, U_CASTLE] = st[ST_CASTLE]
    und[ply, U_EP] = st[ST_EP]
    und[ply, U_R50] = st[ST_R50]
    und[ply, U_KEY] = st[ST_KEY]
    und[ply, U_PKEY] = st[ST_PKEY]
    und[ply, U_MOVE] = 0
    und[ply, U_MOVED] = NO_PIECE
    und[ply, U_CAPT] = NO_PIECE
    key = st[ST_KEY]
    if st[ST_EP] >= 0:
        key ^= ZOB_EP[st[ST_EP] & 7]
    st[ST_EP] = -1
    key ^= ZOB_SIDE
    st[ST_KEY] = key
    st[ST_SIDE] = 1 - st[ST_SIDE]
    st[ST_R50] += 1


@njit(cache=False, error_model="numpy")
def undo_null(bb, sq_of, st, und, ply):
    st[ST_SIDE] = 1 - st[ST_SIDE]
    st[ST_CASTLE] = und[ply, U_CASTLE]
    st[ST_EP] = und[ply, U_EP]
    st[ST_R50] = und[ply, U_R50]
    st[ST_KEY] = und[ply, U_KEY]
    st[ST_PKEY] = und[ply, U_PKEY]


@njit(cache=False, error_model="numpy")
def in_check(bb, st):
    us = st[ST_SIDE]
    return attacked_by(bb, lsb(bb[us * 6 + KING]), 1 - us, bb[OCC_A])


@njit(cache=False, error_model="numpy")
def gives_check(bb, sq_of, st, m):
    """Cheap test used for extensions and pruning; exact for the cases that matter."""
    them = 1 - st[ST_SIDE]
    ksq = lsb(bb[them * 6 + KING])
    frm = m_from(m)
    to = m_to(m)
    flag = m_flag(m)
    pt = sq_of[frm] % 6
    occ = (bb[OCC_A] ^ bit(frm)) | bit(to)
    if flag == F_PROMO:
        pt = 1 + m_promo(m)
    if pt == PAWN:
        if PAWN_ATT[st[ST_SIDE], to] & bit(ksq):
            return True
    elif pt == KNIGHT:
        if KNIGHT_ATT[to] & bit(ksq):
            return True
    elif pt == BISHOP:
        if bishop_attacks(to, occ) & bit(ksq):
            return True
    elif pt == ROOK:
        if rook_attacks(to, occ) & bit(ksq):
            return True
    elif pt == QUEEN:
        if queen_attacks(to, occ) & bit(ksq):
            return True
    # discovered check
    o = st[ST_SIDE] * 6
    if rook_attacks(ksq, occ) & (bb[o + ROOK] | bb[o + QUEEN]) & ~bit(frm):
        return True
    if bishop_attacks(ksq, occ) & (bb[o + BISHOP] | bb[o + QUEEN]) & ~bit(frm):
        return True
    return False


# --------------------------------------------------------------------------------------
# static exchange evaluation
# --------------------------------------------------------------------------------------


@njit(cache=False, error_model="numpy")
def see_value(bb, sq_of, st, m, gain):
    """Value of the capture sequence on the destination square, in centipawns.

    The classic swap list: play out the exchange always recapturing with the cheapest
    attacker, recording the running balance, then fold it back with a negamax minimum
    because either side may stop capturing whenever continuing would lose material.
    `gain` is scratch space owned by the caller so the hot path allocates nothing.
    """
    frm = m_from(m)
    to = m_to(m)
    flag = m_flag(m)
    if flag == F_CASTLE:
        return 0
    us = st[ST_SIDE]
    occ = bb[OCC_A]

    victim = sq_of[to]
    if flag == F_EP:
        occ ^= bit(to - 8 if us == WHITE else to + 8)
        balance = SEE_VALUE[PAWN]
    else:
        balance = SEE_VALUE[victim % 6] if victim != NO_PIECE else 0

    a_pt = sq_of[frm] % 6
    if flag == F_PROMO:
        a_pt = 1 + m_promo(m)
        balance += SEE_VALUE[a_pt] - SEE_VALUE[PAWN]

    gain[0] = balance
    a_sq = frm
    side = us
    d = 0
    attackers = attackers_to(bb, to, occ) & occ
    sliders_b = bb[BISHOP] | bb[6 + BISHOP] | bb[QUEEN] | bb[6 + QUEEN]
    sliders_r = bb[ROOK] | bb[6 + ROOK] | bb[QUEEN] | bb[6 + QUEEN]

    while True:
        d += 1
        gain[d] = SEE_VALUE[a_pt] - gain[d - 1]
        if max(-gain[d - 1], gain[d]) < 0:
            break
        occ ^= bit(a_sq)
        attackers &= ~bit(a_sq)
        attackers |= ((bishop_attacks(to, occ) & sliders_b)
                      | (rook_attacks(to, occ) & sliders_r))
        attackers &= occ
        side = 1 - side
        mine = attackers & bb[OCC_W + side]
        if mine == 0:
            break
        a_pt = KING
        picked = np.int64(0)
        for t in range(PAWN, KING + 1):
            sel = mine & bb[side * 6 + t]
            if sel != 0:
                a_pt = t
                picked = sel
                break
        if a_pt == KING and (attackers & bb[OCC_W + (1 - side)]) != 0:
            break  # the king may not walk into a square the other side still attacks
        a_sq = lsb(picked)

    while d > 0:
        gain[d - 1] = -max(-gain[d - 1], gain[d])
        d -= 1
    return gain[0]


@njit(cache=False, error_model="numpy")
def see_ge(bb, sq_of, st, m, threshold, gain):
    return see_value(bb, sq_of, st, m, gain) >= threshold


# --------------------------------------------------------------------------------------
# plain-python glue: only ever runs once per move, so it does not need to be fast
# --------------------------------------------------------------------------------------

FILES = "abcdefgh"
PROMO_CHAR = "nbrq"


def new_position():
    return (np.zeros(15, dtype=np.int64),
            np.full(64, NO_PIECE, dtype=np.int8),
            np.zeros(ST_N, dtype=np.int64))


def set_from_board(bb, sq_of, st, board):
    """Load a python-chess board into our arrays. python-chess owns FEN parsing."""
    bb[:] = 0
    sq_of[:] = NO_PIECE
    for square, piece in board.piece_map().items():
        p = (0 if piece.color else 6) + (piece.piece_type - 1)
        sq_of[square] = p
        b = np.int64(1) << np.int64(square)
        bb[p] |= b
        bb[OCC_W + (0 if piece.color else 1)] |= b
        bb[OCC_A] |= b
    rights = 0
    cr = board.castling_rights
    if cr & (1 << 7):
        rights |= CR_WK
    if cr & 1:
        rights |= CR_WQ
    if cr & (1 << 63):
        rights |= CR_BK
    if cr & (1 << 56):
        rights |= CR_BQ
    st[ST_SIDE] = WHITE if board.turn else BLACK
    st[ST_CASTLE] = rights
    st[ST_EP] = board.ep_square if board.ep_square is not None else -1
    st[ST_R50] = board.halfmove_clock
    st[ST_GPLY] = 0
    st[ST_GLEN] = 0
    compute_key(bb, sq_of, st)


def move_to_uci(m):
    frm, to = m & 63, (m >> 6) & 63
    text = FILES[frm & 7] + str((frm >> 3) + 1) + FILES[to & 7] + str((to >> 3) + 1)
    if (m >> 14) & 3 == F_PROMO:
        text += PROMO_CHAR[(m >> 12) & 3]
    return text


def uci_to_move(uci, ml, count):
    """Match a UCI string against a generated move list. Returns -1 if it is not there."""
    for i in range(count):
        if move_to_uci(ml[i]) == uci:
            return ml[i]
    return -1


# --------------------------------------------------------------------------------------
# perft, used only to prove the generator correct
# --------------------------------------------------------------------------------------


@njit(cache=False, error_model="numpy")
def perft(bb, sq_of, st, und, ml, ply, depth):
    if depth == 0:
        return 1
    n = gen_moves(bb, sq_of, st, ml, ply, np.int64(0))
    if depth == 1:
        return n
    total = 0
    for i in range(n):
        m = ml[ply, i]
        do_move(bb, sq_of, st, und, ply, m)
        total += perft(bb, sq_of, st, und, ml, ply + 1, depth - 1)
        undo_move(bb, sq_of, st, und, ply)
    return total
