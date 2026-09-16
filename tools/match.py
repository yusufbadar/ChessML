"""Play a match between two agent directories through the real referee.

Uses harness/referee.py unmodified, so the clock, legality, draw claims and 300-ply
adjudication are the platform's. Colours alternate and games start from balanced opening
positions, because rated games do not start from the initial position.

    python tools/match.py --black baselines/minimax --games 6 --base-ms 20000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.referee import play_match  # noqa: E402
from tools.winsandbox import local  # noqa: E402

# Balanced positions a few moves into mainstream openings, so neither side is prepared.
OPENINGS = [
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 2 5",
    "rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R b KQkq - 1 3",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbqkbnr/pp2pppp/8/2pp4/3P4/2N2N2/PPP1PPPP/R1BQKB1R b KQkq - 1 3",
    "rnbqk2r/ppppppbp/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4",
    "rnbqkb1r/pp3ppp/4pn2/2pp4/2PP4/5NP1/PP2PP1P/RNBQKB1R w KQkq - 0 5",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--white", default=".")
    parser.add_argument("--black", default="baselines/minimax")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--base-ms", type=int, default=120_000)
    parser.add_argument("--increment-ms", type=int, default=500)
    parser.add_argument("--pgn", type=Path)
    args = parser.parse_args()

    score = 0.0
    wins = draws = losses = 0
    pgns = []
    for game in range(args.games):
        opening = OPENINGS[game % len(OPENINGS)]
        us_white = game % 2 == 0
        first = args.white if us_white else args.black
        second = args.black if us_white else args.white
        outcome = play_match(local(Path(first)), local(Path(second)),
                             args.base_ms, args.increment_ms, start_fen=opening)
        pgns.append(outcome.pgn)
        if outcome.result == "draw":
            point = 0.5
        elif (outcome.result == "white") == us_white:
            point = 1.0
        else:
            point = 0.0
        score += point
        wins += point == 1.0
        draws += point == 0.5
        losses += point == 0.0
        print(f"  game {game + 1}: {'white' if us_white else 'black':>5} "
              f"{point:>3} by {outcome.termination:<22} running {score}/{game + 1}")

    played = args.games
    print(f"\n{args.white} vs {args.black}: {score}/{played} = "
          f"{100 * score / played:.1f}%  (+{wins} ={draws} -{losses})")
    if args.pgn:
        args.pgn.write_text("\n\n".join(pgns))
        print(f"pgn written to {args.pgn}")


if __name__ == "__main__":
    main()
