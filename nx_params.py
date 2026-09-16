"""Every evaluation weight the engine has, laid out as one flat vector.

Two reasons for the flat vector. It keeps the evaluation a pure function of (position,
weights), which is what a tuner needs, and it means scripts/tune.py can rewrite the whole
evaluation by writing a single tuple back into this file. TUNED below is that tuple.

The defaults are built, not copied. Piece-square tables come from geometric shape
functions -- centrality, advancement, file preference -- so that the starting point is
explainable rather than borrowed, and everything after that is ordinary chess knowledge
stated as a number: a bishop pair is worth about half a pawn, an isolated pawn costs
about a sixth of one. The tuner moves them from there.
"""

import numpy as np

MG, EG = 0, 1

# ---- layout ---------------------------------------------------------------------------
# Every block is a (offset, size) pair. Sizes are doubled because each weight is a
# (midgame, endgame) pair stored adjacently.

I_MATERIAL = 0            # 5 piece types, no king
N_MATERIAL = 5 * 2

I_PSQT = I_MATERIAL + N_MATERIAL          # [piece][square][phase]
N_PSQT = 6 * 64 * 2

I_MOBILITY = I_PSQT + N_PSQT              # [piece-1][count][phase] for N B R Q
N_MOBILITY = 4 * 28 * 2

I_PASSED = I_MOBILITY + N_MOBILITY        # [rank][phase]
N_PASSED = 8 * 2
I_PASSED_PROT = I_PASSED + N_PASSED       # defended by a friendly pawn
I_PASSED_FREE = I_PASSED_PROT + 2         # path to promotion is empty
I_PASSED_KING = I_PASSED_FREE + 2         # endgame: our king near it, their king far
I_ISOLATED = I_PASSED_KING + 2
I_DOUBLED = I_ISOLATED + 2
I_BACKWARD = I_DOUBLED + 2
I_CONNECTED = I_BACKWARD + 2              # [rank][phase]
N_CONNECTED = 8 * 2
I_PAWN_END = I_CONNECTED + N_CONNECTED

I_BISHOP_PAIR = I_PAWN_END
I_ROOK_OPEN = I_BISHOP_PAIR + 2
I_ROOK_SEMI = I_ROOK_OPEN + 2
I_ROOK_SEVENTH = I_ROOK_SEMI + 2
I_OUTPOST_N = I_ROOK_SEVENTH + 2
I_OUTPOST_B = I_OUTPOST_N + 2
I_BAD_BISHOP = I_OUTPOST_B + 2            # per own pawn on the bishop's colour
I_MINOR_SHIELD = I_BAD_BISHOP + 2         # minor with a pawn directly in front
I_TEMPO = I_MINOR_SHIELD + 2
I_PIECE_END = I_TEMPO + 2

I_SHELTER = I_PIECE_END                   # [edge-distance 0..3][own pawn rank 0..7]
N_SHELTER = 4 * 8
I_STORM = I_SHELTER + N_SHELTER           # [edge-distance 0..3][enemy pawn rank 0..7]
N_STORM = 4 * 8
I_KING_ATT = I_STORM + N_STORM            # attack weight per attacking piece type
N_KING_ATT = 6
I_SAFE_CHECK = I_KING_ATT + N_KING_ATT    # knight, bishop, rook, queen
N_SAFE_CHECK = 4
I_KING_OPEN = I_SAFE_CHECK + N_SAFE_CHECK  # open / semi-open file beside our king
I_KING_QUAD = I_KING_OPEN + 4             # danger scaling: linear, quadratic
I_KING_END = I_KING_QUAD + 2

I_THREAT_PAWN_MINOR = I_KING_END
I_THREAT_PAWN_MAJOR = I_THREAT_PAWN_MINOR + 2
I_THREAT_MINOR_MAJOR = I_THREAT_PAWN_MAJOR + 2
I_THREAT_ROOK_QUEEN = I_THREAT_MINOR_MAJOR + 2
I_THREAT_HANGING = I_THREAT_ROOK_QUEEN + 2
I_THREAT_KING = I_THREAT_HANGING + 2
I_END = I_THREAT_KING + 2

N_PARAMS = I_END


# ---- shape functions used to build the piece-square tables -----------------------------

def _centre(f, r):
    """1.0 in the middle four squares, 0.0 at the corners."""
    return (1.0 - abs(f - 3.5) / 3.5) * (1.0 - abs(r - 3.5) / 3.5)


def _file_centre(f):
    return 1.0 - abs(f - 3.5) / 3.5


def _psqt_defaults():
    """Piece-square tables from geometry: centrality, advancement, home-rank penalties."""
    table = np.zeros((6, 64, 2), dtype=np.float64)
    for sq in range(64):
        f, r = sq & 7, sq >> 3
        c = _centre(f, r)
        home = 1.0 if r == 0 else 0.0

        # Pawns: reward advancing, reward holding the centre early, and nothing on the
        # first or last rank where a pawn cannot stand.
        if 0 < r < 7:
            adv_mg = (0.0, 0.0, 2.0, 8.0, 20.0, 40.0, 68.0, 0.0)[r]
            adv_eg = (0.0, 4.0, 8.0, 16.0, 32.0, 62.0, 110.0, 0.0)[r]
            centre_pull = (-6.0, -3.0, 1.0, 9.0, 9.0, 1.0, -3.0, -6.0)[f]
            table[0, sq, MG] = adv_mg + centre_pull * (1.0 if r <= 3 else 0.4)
            table[0, sq, EG] = adv_eg
            if r == 1 and f in (3, 4):
                table[0, sq, MG] -= 8.0   # an unmoved d/e pawn blocks its own pieces

        # Knights live on centralised, advanced squares and hate the rim.
        table[1, sq, MG] = -38.0 + 76.0 * c + 14.0 * (r / 7.0) - 12.0 * home
        table[1, sq, EG] = -32.0 + 62.0 * c

        # Bishops: mild centralisation, long diagonals, off the back rank.
        long_diag = 6.0 if (f == r or f + r == 7) else 0.0
        table[2, sq, MG] = -16.0 + 30.0 * c + long_diag + 6.0 * (r / 7.0) - 10.0 * home
        table[2, sq, EG] = -12.0 + 24.0 * c + long_diag * 0.5

        # Rooks want central files and the seventh rank; the file term does most of it.
        seventh = (0.0, -2.0, -2.0, 0.0, 2.0, 6.0, 18.0, 4.0)[r]
        table[3, sq, MG] = 8.0 * _file_centre(f) + seventh
        table[3, sq, EG] = 4.0 + 6.0 * (r / 7.0)

        # Queens: slight centralisation, and no reason to come out early.
        table[4, sq, MG] = -14.0 + 22.0 * c - 6.0 * home
        table[4, sq, EG] = -26.0 + 46.0 * c

        # King: tucked away in the middlegame, marching in the endgame.
        shelter = 26.0 * (1.0 - _file_centre(f)) - 34.0 * (r / 7.0)
        table[5, sq, MG] = shelter + (10.0 if r == 0 and f in (1, 2, 6) else 0.0)
        table[5, sq, EG] = -48.0 + 96.0 * c
    return table


def _mobility_defaults():
    """Mobility is worth a lot per square when a piece has few, and saturates."""
    table = np.zeros((4, 28, 2), dtype=np.float64)
    #             knight bishop rook  queen
    step_mg = (5.0, 4.4, 2.6, 1.4)
    step_eg = (5.6, 5.2, 4.6, 2.4)
    centre = (4.0, 6.5, 7.0, 13.0)
    for p in range(4):
        for c in range(28):
            # A square-root shape: the first legal squares matter most.
            x = (c - centre[p])
            shaped = (x / (1.0 + abs(x) / 8.0))
            table[p, c, MG] = step_mg[p] * shaped
            table[p, c, EG] = step_eg[p] * shaped
    return table


def build_default():
    p = np.zeros(N_PARAMS, dtype=np.float64)

    # Material. Classical values with the bishop a shade above the knight and the
    # endgame values reflecting that rooks and pawns gain as the board empties.
    for i, (mg, eg) in enumerate([(88, 118), (330, 300), (350, 320), (470, 560), (960, 1040)]):
        p[I_MATERIAL + i * 2 + MG] = mg
        p[I_MATERIAL + i * 2 + EG] = eg

    psqt = _psqt_defaults()
    for pt in range(6):
        for sq in range(64):
            for ph in range(2):
                p[I_PSQT + (pt * 64 + sq) * 2 + ph] = psqt[pt, sq, ph]

    mob = _mobility_defaults()
    for pt in range(4):
        for c in range(28):
            for ph in range(2):
                p[I_MOBILITY + (pt * 28 + c) * 2 + ph] = mob[pt, c, ph]

    # Passed pawns, by the rank they stand on. The endgame value grows fast.
    for r, (mg, eg) in enumerate([(0, 0), (2, 8), (6, 18), (14, 38), (32, 70),
                                  (58, 118), (94, 180), (0, 0)]):
        p[I_PASSED + r * 2 + MG] = mg
        p[I_PASSED + r * 2 + EG] = eg
    p[I_PASSED_PROT + MG], p[I_PASSED_PROT + EG] = 8, 16
    p[I_PASSED_FREE + MG], p[I_PASSED_FREE + EG] = 6, 20
    p[I_PASSED_KING + MG], p[I_PASSED_KING + EG] = 0, 12
    p[I_ISOLATED + MG], p[I_ISOLATED + EG] = -12, -16
    p[I_DOUBLED + MG], p[I_DOUBLED + EG] = -8, -22
    p[I_BACKWARD + MG], p[I_BACKWARD + EG] = -10, -8
    for r, (mg, eg) in enumerate([(0, 0), (4, 2), (6, 3), (10, 6), (18, 14),
                                  (32, 30), (54, 58), (0, 0)]):
        p[I_CONNECTED + r * 2 + MG] = mg
        p[I_CONNECTED + r * 2 + EG] = eg

    p[I_BISHOP_PAIR + MG], p[I_BISHOP_PAIR + EG] = 34, 58
    p[I_ROOK_OPEN + MG], p[I_ROOK_OPEN + EG] = 30, 14
    p[I_ROOK_SEMI + MG], p[I_ROOK_SEMI + EG] = 12, 8
    p[I_ROOK_SEVENTH + MG], p[I_ROOK_SEVENTH + EG] = 12, 20
    p[I_OUTPOST_N + MG], p[I_OUTPOST_N + EG] = 26, 14
    p[I_OUTPOST_B + MG], p[I_OUTPOST_B + EG] = 16, 8
    p[I_BAD_BISHOP + MG], p[I_BAD_BISHOP + EG] = -3, -5
    p[I_MINOR_SHIELD + MG], p[I_MINOR_SHIELD + EG] = 8, 0
    p[I_TEMPO + MG], p[I_TEMPO + EG] = 20, 12

    # King shelter: a friendly pawn close in front of the king is worth a lot, and the
    # value falls off sharply the further away it is. Index 0 is the file the king is on.
    for d in range(4):
        near = (1.0, 0.85, 0.5, 0.0)[d]
        for r in range(8):
            # r is the distance from the king's rank to the nearest own pawn on that file
            p[I_SHELTER + d * 8 + r] = near * (0.0, 34.0, 20.0, 8.0, 2.0, 0.0, 0.0, -12.0)[r]
            p[I_STORM + d * 8 + r] = -near * (0.0, 44.0, 26.0, 12.0, 4.0, 0.0, 0.0, 0.0)[r]

    for i, w in enumerate([0, 32, 34, 46, 82, 0]):  # pawn N B R Q king
        p[I_KING_ATT + i] = w
    for i, w in enumerate([56, 34, 62, 46]):        # safe checks: N B R Q
        p[I_SAFE_CHECK + i] = w
    p[I_KING_OPEN + MG], p[I_KING_OPEN + EG] = -22, 0
    p[I_KING_OPEN + 2 + MG], p[I_KING_OPEN + 2 + EG] = -10, 0
    p[I_KING_QUAD + 0] = 12     # linear part, divided by 16 in the evaluation
    p[I_KING_QUAD + 1] = 10     # quadratic part, divided by 256

    p[I_THREAT_PAWN_MINOR + MG], p[I_THREAT_PAWN_MINOR + EG] = 52, 38
    p[I_THREAT_PAWN_MAJOR + MG], p[I_THREAT_PAWN_MAJOR + EG] = 62, 44
    p[I_THREAT_MINOR_MAJOR + MG], p[I_THREAT_MINOR_MAJOR + EG] = 34, 26
    p[I_THREAT_ROOK_QUEEN + MG], p[I_THREAT_ROOK_QUEEN + EG] = 30, 18
    p[I_THREAT_HANGING + MG], p[I_THREAT_HANGING + EG] = 22, 24
    p[I_THREAT_KING + MG], p[I_THREAT_KING + EG] = 12, 20
    return p


TUNED = (
    70, 176, 530, 668, 600, 720, 480, 1206, 1080, 2220, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, -6, 20, 14, 48, 0, 56, -20, 48, -20, 48,
    0, 56, 14, 48, -6, 20, -4, -8, 6, -18, -4, 2,
    -4, 0, -4, 0, -4, 2, 6, -18, -4, -8, 14, 0,
    0, 10, 14, -6, 18, -28, 18, -28, 14, -6, 0, 10,
    14, 0, 28, 48, 22, 34, 30, 18, 48, -34, 48, -34,
    30, 18, 22, 34, 28, 48, 64, 64, 118, 64, 84, 22,
    98, -30, 98, -30, 84, 22, 118, 64, 64, 64, -122, -20,
    42, 52, 156, 40, 228, -160, 228, -160, 156, 40, 42, 52,
    -122, -20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, -170, -164, -120, -104, -112, -84,
    -122, -44, -122, -44, -112, -84, -120, -104, -170, -164, -110, -44,
    -110, -76, -96, -66, -92, -76, -92, -76, -96, -66, -110, -76,
    -110, -44, -108, -100, -80, -66, -98, -56, -72, -24, -72, -24,
    -98, -56, -80, -66, -108, -100, -68, -72, -44, -20, -68, -32,
    -80, 20, -80, 20, -68, -32, -44, -20, -68, -72, -22, 14,
    -52, 12, -54, 16, -36, 36, -36, 36, -54, 16, -52, 12,
    -22, 14, -12, -50, -28, -94, -28, 12, 8, -18, 8, -18,
    -28, 12, -28, -94, -12, -50, 0, -44, -66, -48, -88, -10,
    -16, -6, -16, -6, -88, -10, -66, -48, 0, -44, -278, -144,
    -68, -34, -288, 50, -72, 22, -72, 22, -288, 50, -68, -34,
    -278, -144, 2, 20, 38, -22, -30, 10, -28, 56, -28, 56,
    -30, 10, 38, -22, 2, 20, 20, -8, 30, -4, 32, -4,
    -14, 28, -14, 28, 32, -4, 30, -4, 20, -8, 34, 6,
    16, 22, 0, 62, 6, 48, 6, 48, 0, 62, 16, 22,
    34, 6, 32, -20, -12, 58, -14, 66, 36, 56, 36, 56,
    -14, 66, -12, 58, 32, -20, -16, 66, 14, 56, 44, 50,
    16, 76, 16, 76, 44, 50, 14, 56, -16, 66, 64, 36,
    96, 24, 28, 52, 56, 20, 56, 20, 28, 52, 96, 24,
    64, 36, -22, 96, -60, 46, -20, 54, 22, 88, 22, 88,
    -20, 54, -60, 46, -22, 96, -118, 78, -68, 72, -230, 114,
    -260, 102, -260, 102, -230, 114, -68, 72, -118, 78, -46, -24,
    -18, -40, -32, -20, -16, -50, -16, -50, -32, -20, -18, -40,
    -46, -24, -56, -22, -32, -38, -32, -48, -26, -54, -26, -54,
    -32, -48, -32, -38, -56, -22, 24, -66, 38, -38, -14, -22,
    -16, -32, -16, -32, -14, -22, 38, -38, 24, -66, -36, 26,
    8, 2, -26, 20, -16, -4, -16, -4, -26, 20, 8, 2,
    -36, 26, 28, 26, 14, 50, 24, 56, 8, 42, 8, 42,
    24, 56, 14, 50, 28, 26, 22, 30, 100, 6, 72, 28,
    66, 42, 66, 42, 72, 28, 100, 6, 22, 30, 16, -34,
    16, -36, 46, -4, 30, 4, 30, 4, 46, -4, 16, -36,
    16, -34, 168, 8, 102, 26, 174, 10, 132, 2, 132, 2,
    174, 10, 102, 26, 168, 8, -66, -42, -38, -152, -56, -134,
    -36, -192, -36, -192, -56, -134, -38, -152, -66, -42, -16, -112,
    -20, -164, -22, -168, -26, -140, -26, -140, -22, -168, -20, -164,
    -16, -112, 14, -158, -6, -106, -44, -40, -52, -62, -52, -62,
    -44, -40, -6, -106, 14, -158, -14, -44, 10, -76, -56, 10,
    -72, 84, -72, 84, -56, 10, 10, -76, -14, -44, 24, -14,
    -42, 22, -66, 72, -84, 180, -84, 180, -66, 72, -42, 22,
    24, -14, 52, -16, 64, -46, -24, 118, -20, 74, -20, 74,
    -24, 118, 64, -46, 52, -16, 22, -10, -48, 26, 26, -28,
    -58, 114, -58, 114, 26, -28, -48, 26, 22, -10, -68, -20,
    52, -6, 106, -84, 42, 28, 42, 28, 106, -84, 52, -6,
    -68, -20, 8, -152, 42, -100, 42, -86, 46, -104, 46, -104,
    22, -86, 42, -100, 8, -152, 12, -72, 38, -50, -10, -52,
    -28, -50, -28, -50, -10, -52, 38, -50, 12, -72, -190, -38,
    -42, -42, -98, -16, -106, -18, -106, -18, -98, -16, -42, -42,
    -190, -38, -298, -48, -32, -32, -130, 6, -188, 26, -188, 26,
    -130, 6, -32, -32, -298, -48, -216, -32, 20, -6, -92, 22,
    -184, 36, -184, 36, -92, 22, 20, -6, -216, -32, 34, -40,
    102, 2, 242, -28, 316, -40, 316, -40, 242, -28, 102, 2,
    34, -40, 108, -106, 240, 44, 204, -20, 40, -14, 40, -14,
    204, -20, 240, 44, 108, -106, 334, -290, 200, -46, 84, -24,
    -320, 24, -320, 24, 84, -24, 200, -46, 334, -290, -182, -242,
    -104, -162, -66, -82, -58, -30, -38, -2, -30, 32, -10, 24,
    6, 40, 20, 24, 30, 34, 34, 38, 38, 42, 40, 44,
    42, 48, 44, 50, 46, 52, 48, 54, 50, 56, 50, 58,
    52, 58, 54, 60, 54, 60, 56, 62, 56, 64, 58, 64,
    58, 64, 58, 66, 60, 66, -176, -332, -98, -300, -76, -182,
    -58, -96, -42, -70, -28, -42, -18, -14, -10, 4, -10, 10,
    2, 14, -6, 30, 30, 10, 78, 4, 176, -78, 34, 40,
    36, 42, 38, 46, 40, 48, 42, 50, 42, 50, 44, 52,
    46, 54, 46, 54, 48, 56, 48, 58, 50, 58, 50, 58,
    50, 60, -420, -494, -76, -312, -36, -228, -24, -140, -30, -68,
    -26, -48, -30, -18, -24, -4, -12, 8, -2, 24, 8, 30,
    8, 40, 28, 50, 50, 40, 120, 6, 20, 36, 22, 38,
    24, 40, 24, 42, 24, 44, 26, 46, 26, 46, 28, 48,
    28, 50, 28, 50, 28, 50, 30, 52, 30, 52, -14, -24,
    -50, -634, 12, -542, -52, -320, -36, -400, -34, -280, -26, -216,
    -30, -128, -26, -100, -26, -50, -22, -30, -18, 4, -16, 16,
    -12, 40, -6, 44, -8, 58, 0, 52, 2, 52, 16, 34,
    22, 40, 32, 64, 60, -22, 52, -10, 182, -178, 136, -70,
    74, -146, -196, 60, -404, -2, 0, 0, 0, -28, -56, 22,
    -26, 56, 12, 102, 26, 188, 108, 320, 0, 0, 44, -10,
    12, 120, 0, 48, -14, -28, -22, -30, -16, -12, 0, 0,
    2, -8, 18, 22, 20, 26, 30, 34, 24, 128, 688, 4,
    0, 0, -48, -44, 64, -14, 32, 2, 42, 30, 40, 42,
    96, -6, -20, -8, 6, 34, 46, 24, 0, 80, 70, 42,
    24, 4, -94, -24, 0, 46, 18, 12, -2, 12, 0, -8,
    0, 34, 20, 8, 2, 0, 0, -12, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 74, -30, 2, 16, 22, 18, -24,
    0, -74, -78, -34, -6, 8, 2, -4, 0, -44, -26, -12,
    -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 22, 14, 16, 12, 0, 106, 19, 82, 86, -37, 0,
    -35, 0, 4, 10, 80, 98, 68, 136, 80, 114, 120, 106,
    40, 22, 24, 104,
)



def load():
    """The tuned vector when scripts/tune.py has written one, otherwise the defaults."""
    if len(TUNED) == N_PARAMS:
        return np.array(TUNED, dtype=np.int32)
    return np.rint(build_default()).astype(np.int32)
