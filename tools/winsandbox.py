"""A Windows-compatible stand-in for harness/sandbox.py.

The platform is Linux and the shipped harness reads the agent's pipes with select(),
which on Windows only works on sockets. This talks the identical protocol over the same
harness/runner.py, using a reader thread instead, and exposes the same three methods, so
harness/referee.py runs unmodified on top of it and the clock, legality, draw and
adjudication rules stay exactly the ones the platform uses.

Nothing in harness/ is touched. This file only replaces the I/O layer under it.
"""

import json
import queue
import subprocess
import sys
import threading
from pathlib import Path

from harness.rules import STDOUT_CAP
from harness.sandbox import AgentFailure

RUNNER = Path(__file__).resolve().parent.parent / "harness" / "runner.py"


def local(directory: Path) -> "Agent":
    return Agent([sys.executable, str(RUNNER), str(Path(directory).resolve())])


class Agent:
    def __init__(self, command):
        self.command = command
        self.stderr_tail = ""
        self._process = None
        self._lines = queue.Queue()
        self._err = []

    def start(self, init_budget_s):
        self._process = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0)
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        line = self._await_line(init_budget_s)
        if line is None:
            raise AgentFailure("init" if self._process.poll() is None else "crash")
        try:
            payload = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise AgentFailure("init") from None
        if not (isinstance(payload, dict) and payload.get("ready") is True):
            raise AgentFailure("init")

    def move(self, fen, time_left_ms):
        request = json.dumps({"fen": fen, "time_left_ms": time_left_ms}).encode()
        try:
            self._process.stdin.write(request + b"\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError):
            raise AgentFailure("crash") from None
        line = self._await_line((time_left_ms + 500) / 1000.0)
        if line is None:
            raise AgentFailure("flag")
        if len(line) >= STDOUT_CAP:
            raise AgentFailure("illegal")
        try:
            payload = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise AgentFailure("illegal") from None
        move = payload.get("move") if isinstance(payload, dict) else None
        if not isinstance(move, str):
            raise AgentFailure("illegal")
        return move

    def stop(self):
        if self._process is None:
            return
        self._process.kill()
        self._process.wait()
        self.stderr_tail = b"".join(self._err).decode("utf-8", "replace")[-8192:]
        self._process = None

    def _await_line(self, timeout_s):
        try:
            item = self._lines.get(timeout=max(0.0, timeout_s))
        except queue.Empty:
            return None
        if item is None:
            raise AgentFailure("crash")
        return item

    def _pump_stdout(self):
        for line in self._process.stdout:
            self._lines.put(line.rstrip(b"\r\n"))
        self._lines.put(None)

    def _pump_stderr(self):
        for line in self._process.stderr:
            self._err.append(line)
