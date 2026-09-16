"""Fit the evaluation weights to self-play results, and write them back into nx_params.

This is texel tuning. Every position carries the result of the game it came from, and we
choose the weight vector that makes the evaluation predict those results as well as
possible, under a logistic that turns centipawns into an expected score:

    expected = 1 / (1 + 10 ** (-K * eval / 400))
    error    = mean((result - expected) ** 2)

Optimisation is coordinate descent with a per-parameter step: try moving one weight, keep
the move if the error falls, otherwise try the other direction. It is slow and it is
completely robust, which for a thousand correlated integer weights is the right trade.

    python scripts/tune.py --data data/*.npz --sweeps 6
"""

import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import engine_api  # noqa: F401  - sets NUMBA_* and puts the repo root on sys.path
from numba import njit

import nx_params as params
from nx_eval import S_N, evaluate
from nx_core import ST_SIDE, WHITE


@njit(nogil=True, cache=False)
def chunk_error(bb, sq, st, result, P, K, sc, ptt, lo, hi):
    """Summed squared error over one slice. nogil, so threads actually run in parallel."""
    total = 0.0
    for i in range(lo, hi):
        raw = evaluate(bb[i], sq[i], st[i], P, sc, ptt)
        white = raw if st[i, ST_SIDE] == WHITE else -raw
        expected = 1.0 / (1.0 + 10.0 ** (-K * white / 400.0))
        diff = result[i] - expected
        total += diff * diff
    return total


class Error:
    """Mean squared error over a set, spread across threads.

    The search functions are compiled nogil so the platform's watchdog can interrupt
    them; the same property makes the tuner scale across cores here, where unlike in a
    rated game there is more than one. Each thread gets its own scratch, and the pawn
    caches are cleared on every call because the weights have changed underneath them.
    """

    def __init__(self, threads):
        self.threads = max(1, threads)
        self.pool = ThreadPoolExecutor(self.threads)
        self.sc = [np.zeros(S_N, dtype=np.int64) for _ in range(self.threads)]
        self.ptt = [np.zeros((1 << 12, 5), dtype=np.int64) for _ in range(self.threads)]

    def __call__(self, bb, sq, st, result, P, K):
        n = result.shape[0]
        for cache in self.ptt:
            cache[:] = 0
        jobs = []
        for t in range(self.threads):
            lo, hi = t * n // self.threads, (t + 1) * n // self.threads
            if lo < hi:
                jobs.append(self.pool.submit(chunk_error, bb, sq, st, result, P, K,
                                             self.sc[t], self.ptt[t], lo, hi))
        return sum(job.result() for job in jobs) / n


def load(patterns, max_positions):
    """Load a balanced sample without materialising every multi-gigabyte shard at once."""
    paths = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if path not in paths:
                paths.append(path)
    if not paths:
        raise SystemExit("no data; run scripts/selfplay.py first")

    bbs, sqs, sts, res = [], [], [], []
    rng = np.random.default_rng(0)
    remaining = max_positions
    for file_index, path in enumerate(paths):
        with np.load(path) as data:
            count = len(data["result"])
            if count == 0:
                continue
            files_left = len(paths) - file_index
            take = min(count, (remaining + files_left - 1) // files_left)
            if take < count:
                pick = np.sort(rng.choice(count, take, replace=False))
                bbs.append(data["bb"][pick])
                sqs.append(data["sq"][pick])
                sts.append(data["st"][pick])
                res.append(data["result"][pick])
            else:
                bbs.append(data["bb"])
                sqs.append(data["sq"])
                sts.append(data["st"])
                res.append(data["result"])
            remaining -= take
        print(f"  {path}: sampled {take:,} of {count:,} positions", flush=True)
        if remaining <= 0:
            break
    if not res:
        raise SystemExit("no data; run scripts/selfplay.py first")
    return (np.concatenate(bbs), np.concatenate(sqs),
            np.concatenate(sts), np.concatenate(res))


def fit_k(error, bb, sq, st, result, P):
    """The scale that turns our centipawns into the winning chances the labels show."""
    best_k, best_e = 1.0, 1e9
    for k in np.arange(0.20, 3.01, 0.05):
        e = error(bb, sq, st, result, P, k)
        if e < best_e:
            best_k, best_e = float(k), e
    return best_k, best_e


LINE_SEARCH = 12


def build_groups(mirror):
    """Which weights move together.

    Piece-square entries are tied to their mirror image across the d/e file unless asked
    otherwise. Chess is not quite file-symmetric, but with a few tens of thousands of
    positions the pair is far better determined than the two squares separately, and the
    symmetry is a cheap regulariser against fitting noise.

    Entries that can never fire are left out entirely: a pawn cannot stand on the first
    or last rank, so those table values would drift to nonsense.
    """
    dead = set()
    for sq in list(range(8)) + list(range(56, 64)):
        dead.add(params.I_PSQT + sq * 2)
        dead.add(params.I_PSQT + sq * 2 + 1)

    groups = []
    seen = set()
    if mirror:
        for pt in range(6):
            for sq in range(64):
                for phase in range(2):
                    i = params.I_PSQT + (pt * 64 + sq) * 2 + phase
                    j = params.I_PSQT + (pt * 64 + (sq ^ 7)) * 2 + phase
                    if i in seen or i in dead or j in dead:
                        continue
                    seen.add(i)
                    seen.add(j)
                    groups.append((i,) if i == j else (i, j))
    for i in range(params.N_PARAMS):
        if i not in seen and i not in dead:
            groups.append((i,))
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", default=["data/*.npz"])
    parser.add_argument("--sweeps", type=int, default=6)
    parser.add_argument("--step", type=int, default=6)
    parser.add_argument("--no-mirror", action="store_true",
                        help="tune each piece-square entry independently")
    parser.add_argument("--out", type=Path, default=Path("nx_params.py"))
    parser.add_argument("--max-positions", type=int, default=1_200_000)
    parser.add_argument("--threads", type=int, default=0, help="0 means every core")
    parser.add_argument("--k", type=float, default=0.0,
                        help="fix the logistic scale instead of fitting it. With "
                             "engine-eval labels this sets the scale our evaluation "
                             "lands on: fitting drives K*ours -> label, so ours -> "
                             "label/K. Use 0.5. That is twice the labels' centipawns, "
                             "which is the magnitude the search's fixed pruning margins "
                             "want, and it measured +241 elo over fitting K freely.")
    parser.add_argument("--from-default", action="store_true",
                        help="cold start from the built-in weights instead of TUNED")
    parser.add_argument("--npy", type=Path, default=Path("data/tuned.npy"),
                        help="also save the raw vector, for scripts/arena.py")
    parser.add_argument("--no-write", action="store_true",
                        help="only save the vector; do not touch nx_params.py")
    args = parser.parse_args()

    bb, sq, st, result = load(args.data, args.max_positions)
    # A held-out slice the optimiser never sees, so overfitting is visible. Shuffle
    # first: the files arrive in order, so an unshuffled tail is one seed's games.
    order = np.random.default_rng(12345).permutation(len(result))
    bb, sq, st, result = bb[order], sq[order], st[order], result[order]
    cut = int(len(result) * 0.9)
    hb, hs, ht, hr = bb[cut:], sq[cut:], st[cut:], result[cut:]
    bb, sq, st, result = bb[:cut], sq[:cut], st[:cut], result[:cut]
    print(f"tuning on {len(result)} positions, holding out {len(hr)}")

    P = (np.rint(params.build_default()).astype(np.int32) if args.from_default
         else params.load().astype(np.int32))
    measure = Error(args.threads or (os.cpu_count() or 1))
    print(f"{measure.threads} threads")

    t0 = time.time()
    if args.k > 0:
        K = args.k
        error = measure(bb, sq, st, result, P, K)
        print(f"K = {K:.2f} (fixed), starting error {error:.6f}, "
              f"{(time.time() - t0) * 1000:.0f}ms per pass")
    else:
        K, error = fit_k(measure, bb, sq, st, result, P)
        passes = len(np.arange(0.20, 3.01, 0.05))
        print(f"K = {K:.2f} (fitted), starting error {error:.6f} "
              f"({time.time() - t0:.0f}s for {passes} passes, "
              f"{(time.time() - t0) / passes * 1000:.0f}ms each)")

    groups = build_groups(not args.no_mirror)
    print(f"{len(groups)} free groups over {params.N_PARAMS} weights")
    best_held = measure(hb, hs, ht, hr, P, K)
    best_P = P.copy()
    best_sweep = 0
    print(f"held-out error before tuning {best_held:.6f}")
    step = np.full(len(groups), args.step, dtype=np.int32)
    for sweep in range(args.sweeps):
        improved = 0
        t0 = time.time()
        for g, group in enumerate(groups):
            for direction in (1, -1):
                delta = direction * int(step[g])
                if delta == 0:
                    continue
                before = [P[i] for i in group]
                for i in group:
                    P[i] = P[i] + delta
                candidate = measure(bb, sq, st, result, P, K)
                if candidate >= error - 1e-12:
                    for i, v in zip(group, before):
                        P[i] = v
                    continue
                error = candidate
                improved += 1
                # A weight that wants to move usually wants to move further than one
                # step, so keep walking the same way until it stops paying. Without
                # this a weight can only travel one step per sweep, and rescaling the
                # whole vector would take more sweeps than there is time for.
                for _ in range(LINE_SEARCH):
                    saved = [P[i] for i in group]
                    for i in group:
                        P[i] = P[i] + delta
                    further = measure(bb, sq, st, result, P, K)
                    if further >= error - 1e-12:
                        for i, v in zip(group, saved):
                            P[i] = v
                        break
                    error = further
                break
            else:
                step[g] = max(1, step[g] // 2)
        held = measure(hb, hs, ht, hr, P, K)
        flag = ""
        if held < best_held:
            best_held = held
            best_P = P.copy()
            best_sweep = sweep + 1
            if args.npy:
                args.npy.parent.mkdir(parents=True, exist_ok=True)
                np.save(args.npy, best_P)
        else:
            flag = "  (worse on held-out, keeping sweep %d)" % best_sweep
        print(f"  sweep {sweep + 1}: fit {error:.6f}  held-out {held:.6f}  "
              f"{improved} groups moved  {time.time() - t0:.0f}s{flag}", flush=True)
        if improved == 0:
            break
        if held > best_held * 1.004:
            print("  held-out error is climbing; stopping before it fits noise")
            break

    # Early stopping: the weights we keep are the ones that generalised best, not the
    # ones that drove the training error lowest.
    P = best_P

    if args.npy:
        args.npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.npy, P)
        print(f"saved vector to {args.npy}")
    if not args.no_write:
        write(args.out, P)
        print(f"wrote {params.N_PARAMS} tuned weights to {args.out}")


def write(path, P):
    """Replace the TUNED tuple in nx_params.py, leaving the rest of the file alone."""
    text = Path(path).read_text(encoding="utf-8")
    marker = "\nTUNED = ("
    head = text.index(marker)
    tail = text.index("\n\n\ndef load()", head)
    rows = []
    for i in range(0, len(P), 12):
        rows.append("    " + ", ".join(str(int(v)) for v in P[i:i + 12]) + ",")
    body = "\nTUNED = (\n" + "\n".join(rows) + "\n)\n"
    Path(path).write_text(text[:head] + body + text[tail:], encoding="utf-8")


if __name__ == "__main__":
    main()
