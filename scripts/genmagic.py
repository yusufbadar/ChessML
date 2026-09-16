"""Find our own magic multipliers for rook and bishop sliding attacks.

Run once, offline. The output is embedded as data in the engine so that start-up does no
search at all. Everything here is derived from the geometry of the board.
"""
import random

M64 = (1 << 64) - 1
random.seed(0xC01DCAFE)


def rays(sq, deltas, edge_trim):
    """Squares reachable from sq along deltas. edge_trim drops the final square of each ray."""
    out = 0
    r0, f0 = divmod(sq, 8)
    for dr, df in deltas:
        r, f = r0 + dr, f0 + df
        while 0 <= r < 8 and 0 <= f < 8:
            nr, nf = r + dr, f + df
            if edge_trim and not (0 <= nr < 8 and 0 <= nf < 8):
                break
            out |= 1 << (r * 8 + f)
            r, f = nr, nf
    return out


def attacks(sq, occ, deltas):
    """Sliding attacks from sq, stopping on (and including) the first blocker."""
    out = 0
    r0, f0 = divmod(sq, 8)
    for dr, df in deltas:
        r, f = r0 + dr, f0 + df
        while 0 <= r < 8 and 0 <= f < 8:
            s = r * 8 + f
            out |= 1 << s
            if occ >> s & 1:
                break
            r, f = r + dr, f + df
    return out


ROOK_D = ((1, 0), (-1, 0), (0, 1), (0, -1))
BISH_D = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def subsets(mask):
    """Every occupancy subset of mask, by the Carry-Rippler trick."""
    sub = 0
    while True:
        yield sub
        sub = (sub - mask) & mask
        if sub == 0:
            return


def sparse():
    return random.getrandbits(64) & random.getrandbits(64) & random.getrandbits(64)


def find(sq, deltas, edge_trim=True):
    mask = rays(sq, deltas, edge_trim)
    bits = bin(mask).count("1")
    size = 1 << bits
    occs = list(subsets(mask))
    atts = [attacks(sq, o, deltas) for o in occs]
    shift = 64 - bits
    for _ in range(100_000_000):
        magic = sparse()
        if bin((mask * magic) & 0xFF00000000000000).count("1") < 6:
            continue
        table = [0] * size
        used = [False] * size
        ok = True
        for o, a in zip(occs, atts):
            idx = ((o * magic) & M64) >> shift
            if used[idx]:
                if table[idx] != a:
                    ok = False
                    break
            else:
                used[idx] = True
                table[idx] = a
        if ok:
            return mask, magic, bits, table
    raise SystemExit(f"no magic for {sq}")


rook, bish = [], []
rtab, btab = [], []
roff, boff = [], []
for sq in range(64):
    m, magic, bits, table = find(sq, ROOK_D)
    roff.append(len(rtab))
    rtab.extend(table)
    rook.append((m, magic, bits))
    m, magic, bits, table = find(sq, BISH_D)
    boff.append(len(btab))
    btab.extend(table)
    bish.append((m, magic, bits))

print("rook table entries", len(rtab), "bishop table entries", len(btab))


def as_signed(v):
    return v - (1 << 64) if v >> 63 else v


def emit(name, values, per_line=4):
    lines = [f"{name} = ("]
    for i in range(0, len(values), per_line):
        chunk = ", ".join(f"{as_signed(v)}" for v in values[i:i + per_line])
        lines.append(f"    {chunk},")
    lines.append(")")
    return "\n".join(lines)


with open("magic_data.py", "w") as fh:
    fh.write('"""Magic multipliers and masks, found offline by scripts/genmagic.py."""\n\n')
    fh.write(emit("ROOK_MASK", [r[0] for r in rook]) + "\n\n")
    fh.write(emit("ROOK_MAGIC", [r[1] for r in rook]) + "\n\n")
    fh.write(emit("ROOK_BITS", [r[2] for r in rook], 16) + "\n\n")
    fh.write(emit("ROOK_OFFSET", roff, 16) + "\n\n")
    fh.write(emit("BISHOP_MASK", [b[0] for b in bish]) + "\n\n")
    fh.write(emit("BISHOP_MAGIC", [b[1] for b in bish]) + "\n\n")
    fh.write(emit("BISHOP_BITS", [b[2] for b in bish], 16) + "\n\n")
    fh.write(emit("BISHOP_OFFSET", boff, 16) + "\n")
print("wrote magic_data.py")
