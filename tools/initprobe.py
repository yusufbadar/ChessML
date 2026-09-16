"""Import the agent pinned to one core, the way the platform runs it.

The init budget is 90 seconds on 1 dedicated core. Development machines have eight and a
boost clock, so a comfortable start-up here can still overrun there. This pins the process
to a single core and, optionally, loads the other cores so the pinned one is contended and
clocked down -- a crude stand-in for a slower judge machine.

    python tools/initprobe.py            # one core, otherwise idle
    python tools/initprobe.py --load 6   # one core, six busy neighbours
"""

import argparse
import ctypes
import multiprocessing
import subprocess
import sys
import time
from pathlib import Path


def pin_to_one_core():
    if sys.platform != "win32":
        import os
        os.sched_setaffinity(0, {0})
        return
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    if not ctypes.windll.kernel32.SetProcessAffinityMask(handle, 1):
        print("warning: could not set affinity")


def burn():
    while True:
        pow(7, 10007, 1000003)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=int, default=0,
                        help="background busy processes on the other cores")
    args = parser.parse_args()

    workers = [multiprocessing.Process(target=burn, daemon=True) for _ in range(args.load)]
    for w in workers:
        w.start()
    if workers:
        time.sleep(2.0)   # let the clocks settle before we measure

    root = Path(__file__).resolve().parent.parent
    script = (
        "import ctypes, sys, time\n"
        "h = ctypes.windll.kernel32.GetCurrentProcess()\n"
        "ctypes.windll.kernel32.SetProcessAffinityMask(h, 1)\n"
        "t0 = time.monotonic()\n"
        f"sys.path.insert(0, r'{root}')\n"
        "import agent\n"
        "print('READY_AFTER %.1f' % (time.monotonic() - t0))\n"
        "import chess\n"
        "t = time.monotonic()\n"
        "m = agent.get_move(chess.STARTING_FEN, 120000)\n"
        "print('FIRST_MOVE %s in %.2fs' % (m, time.monotonic() - t))\n"
    )
    started = time.monotonic()
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    wall = time.monotonic() - started
    for w in workers:
        w.terminate()

    print(out.stdout.strip())
    if out.stderr.strip():
        print("stderr:", out.stderr.strip()[-800:])
    ready = [ln for ln in out.stdout.splitlines() if ln.startswith("READY_AFTER")]
    if ready:
        seconds = float(ready[0].split()[1])
        margin = 90.0 - seconds
        verdict = "ok" if margin > 15 else ("tight" if margin > 0 else "OVERRUN")
        print(f"\nimport took {seconds:.1f}s of the 90s budget, "
              f"{margin:.1f}s spare -- {verdict}")
    print(f"wall including interpreter start: {wall:.1f}s")


if __name__ == "__main__":
    main()
