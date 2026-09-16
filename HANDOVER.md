# Handover — AI Chessathon agent

**As of 4 September 2026.** Submissions lock **11 September 12:00** (uploads close 11:00),
10 uploads per team per day, latest passing upload is the one that plays.

Repo: `C:\AIChess\engine`. Build the zip with `python -m harness.package --include scripts`.

---

## 1. What it is

A bitboard chess engine written to be compiled by numba, then to search on one core.
python-chess is used for exactly two things: parsing the FEN, and checking the move we are
about to return is legal.

| file | what it is |
|---|---|
| `agent.py` | entry point: clock, repetition history, staged warm-up, interim search |
| `nx_core.py` | bitboards, attack tables, legal movegen, make/unmake, SEE |
| `nx_eval.py` | tapered evaluation, split into small functions for compile time |
| `nx_params.py` | all 1156 evaluation weights as one flat vector (`TUNED`) |
| `nx_search.py` | alpha-beta, one depth per call |
| `nx_fallback.py` | pure-python search, last resort only |
| `nx_magic.py` | magic multipliers, generated offline |

Search: iterative deepening + aspiration, PVS, lockless TT, null move, LMR, LMP, futility
and reverse futility, razoring, SEE pruning, check extensions, IIR, killers, butterfly
history, counter moves, two plies of continuation history, quiescence with SEE + delta.

Move generation is **fully legal**, not pseudo-legal — nothing in the search ever makes a
move to discover it was illegal.

---

## 2. Where it stands

| measure | value |
|---|---|
| perft | all 6 reference positions exact, **85 Mnps** |
| start-up | **27.7 s** pinned to one core here; **28 s** on the judge machine |
| init budget | **90 s** on the platform |
| puzzles | **75.7%** of 600 Lichess puzzles at 20k nodes |
| clock | latest 66-move self-play left 5.9–7.8 s, worst move 9.4 s |
| zip | 17 files, all Python source, 170 KB unzipped |

Tuning history: round 1 fitted to own self-play, **+203 Elo**. Round 2 refitted to 4M
Stockfish-annotated positions from the Lichess eval database, **+241 Elo** more.

---

## 3. What was done, and what it was worth

| change | effect | how measured |
|---|---|---|
| Whole engine in numba, not just the eval | the entire premise | perft 85 Mnps |
| Own magic bitboards + Zobrist from a seeded PRNG | nothing borrowed ships | `scripts/genmagic.py` |
| Literal args routed through typed constants | compile −40% | 5 copies of movegen → 1 |
| `error_model="numpy"` | compile down, **perft 66 → 81 Mnps** | `scripts/perft.py` |
| Killed a duplicate make/unmake specialisation | **−12 s** compile | `tools/compileprofile.py` |
| Texel tune on self-play | **+203 Elo** | 200 games, fixed nodes |
| Texel tune on Lichess evals, scale pinned | **+241 Elo** | 120 games at 20k nodes |
| Time management | **+36%** time per move | `tools/clocktest.py` |
| Staged warm-up + interim search | robustness only | `tools/interimtest.py` |
| Search-margin profile | **+29 Elo** | 48 paired games, 10k nodes, held-out openings |

An incremental 64-wide king-bucket network was also prototyped and rejected. Integer
inference reached 5.88M evaluations/s and all 100 randomized accumulator updates matched
full refreshes, but the 300k-position model solved only 65/100 puzzles and pushed the
full-search compile beyond the 90-second platform budget. Its runtime code and weights
are deliberately not in the submission.

---

## 4. What is lacking

**The evaluation is the ceiling.** The held-out fitting error was *still falling* when the
last tuning run stopped, and the fit had already halved. That says the handcrafted
functional form — not the data — is the limit. There are 390M more labelled positions
where the 4M came from.

**It has no plan in quiet positions.** One game was drawn in 11 moves after the engine
played `Rb1 Ra1 Rb1 Ra1 Rb1 Ra1`. The evaluation is flat when nothing is
forcing, so every move looks the same and it shuffles. This is the same problem as above
wearing a different hat.

**Pondering is unavailable.** The current live rules say the process is suspended while
the opponent moves. Older notes said it was allowed; do not spend time implementing it.

**Only one search-parameter family has been tuned.** The accepted pruning profile gained
29 Elo, but LMR and null-move parameters still use their original values.

**No pruning at PV nodes.** Conservative; ordinary Elo is available there.

**One unexplained event.** A single clock test flagged at ply 44 and never reproduced in
four later runs. Probably a machine stall under load. `agent.py` now prints `OVERSHOOT` to
stderr if a move ever exceeds its hard limit — **if that string appears in a match log,
that is the smoking gun and it needs fixing immediately.**

---

## 5. What to do next, ranked

### 1. Train a stronger network candidate offline
The first 64-wide candidate was too weak and compiled too slowly after integration. A
future attempt needs more data, a stronger architecture, and an isolated compile gate
before it touches the production search. The data pipeline already produces the labels:
```bash
python scripts/fetch_evals.py --positions 20000000
```

### 2. Tune the remaining LMR and null-move parameters
The accepted futility/razoring/SEE profile gained 29 Elo. Change one family at a time and
keep only paired-match winners.

### 3. More tuning data and more sweeps
The error was still falling. Bigger set, more sweeps, and consider dropping the
mirrored-PSQT grouping now that data is plentiful:
```bash
python scripts/tune.py --data "data/ev*.npz" --k 0.5 --sweeps 16 --no-write
```

### 4. Not worth it
An opening book — rated games start from curated positions, so a book keyed on move one is
already out of book.

---

## 6. Traps that cost real time — do not re-learn these

**Evaluation scale is coupled to the search.** Fitting with the logistic scale free halved
the held-out error and *lost 16 Elo*. The search's pruning margins are fixed centipawn
constants (currently RFP 66/ply, razoring 180, delta 140, plus SEE values), so shrinking the
evaluation silently makes all of them more aggressive. Multiplying the tuned vector by 2 —
same shape, nothing refitted — turned −16 into **+241**. **Always tune with `--k 0.5`.**
If the evaluation is ever rescaled, the margins move with it.

**numba compiles a second copy for every literal.** Passing `0`, `False`, or a plain
Python `int` where an `int32` is expected creates a whole extra specialisation. This cost
40% of start-up once and another 12 seconds later. `tools/compileprofile.py` prints
duplicate specialisations — check it after touching any call signature.

**`no_cpython_wrapper=True` segfaults**, it does not raise, if the function is ever called
from Python. Not worth the saving.

**Read the match logs before theorising.** A long stretch went into single-core compile
time on the theory that games opened with the fallback. The logs showed `Ready in 28.1 s`
of a `90.0 s` budget. It was never the problem.

**Only the match decides.** A lower fitting error that loses the arena is not an
improvement. Contempt looked obviously right for the round-4 shuffle draw; measured at 16,
28 and 45 over 150 games each it gained nothing, so it ships at zero.

---

## 7. Commands

```bash
python scripts/perft.py                    # movegen vs reference counts — must be 0 failures
python tools/differential.py 100 160        # random legal/make/undo checks vs python-chess
python scripts/puzzles.py --count 1000     # tactics by rating band, fixed nodes, ~30 s
python scripts/sanity.py                   # evaluation sanity, root move agreement
python scripts/bench.py 10                 # compile time and node rate
python tools/clocktest.py                  # full game at 120s + 0.5s — watch for FLAGGED
python tools/initprobe.py                  # start-up pinned to one core, like the platform
python tools/compileprofile.py             # per-function compile cost, duplicate signatures
python tools/interimtest.py                # the path covering a compilation overrun
python tools/match.py --black baselines/minimax --games 4 --base-ms 20000
python -m harness.package --include scripts
```

**Before every upload:** `perft.py` (0 failures) and `clocktest.py` (no `FLAGGED`, no
`OVERSHOOT`). Illegal moves and flagging are the two failure modes that lose whole games,
and both are cheap to rule out.

---

## 8. Compliance

Nothing borrowed ships. No engine source, no engine inside the zip, no native binaries, no
network, no subprocesses. Magic numbers and Zobrist keys are generated by our own code.

The evaluation *weights* are fitted to Stockfish-annotated positions from the Lichess
database (CC0). The rules permit this in two places — "Training data is unrestricted,
including positions annotated by an existing engine" and "the ban covers what ships and
runs inside the zip". `scripts/fetch_evals.py` ships alongside the engine so the
provenance can be read rather than trusted.
