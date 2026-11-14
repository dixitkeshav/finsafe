"""Timestamped notebook stages and streamed subprocess logs, without secrets."""
import contextlib
import datetime as dt
import json
import os
import re
import subprocess
import time
from pathlib import Path


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(tmp, path)


class Progress:
    def __init__(self, root, name="kaggle-stage-02"):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.started = time.monotonic()

    def log(self, message):
        message = re.sub(r"hf_[A-Za-z0-9]{12,}", "[REDACTED_HF_TOKEN]", str(message))
        stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        line = f"[{stamp}] [session {time.monotonic()-self.started:.1f}s] {message}"
        print(line, flush=True)
        with (self.root / f"{self.name}.log").open("a") as stream:
            stream.write(line + "\n")

    @contextlib.contextmanager
    def stage(self, name):
        start = time.monotonic()
        self.log(f"START {name}")
        state = {"stage": name, "status": "running", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        atomic_json(self.root / f"{name}.json", state)
        try:
            yield
        except BaseException as error:
            state.update(status="failed", elapsed_seconds=time.monotonic()-start, error_type=type(error).__name__)
            atomic_json(self.root / f"{name}.json", state)
            self.log(f"FAILED {name} after {state['elapsed_seconds']:.1f}s ({type(error).__name__})")
            raise
        else:
            state.update(status="complete", elapsed_seconds=time.monotonic()-start)
            atomic_json(self.root / f"{name}.json", state)
            self.log(f"DONE {name} in {state['elapsed_seconds']:.1f}s")

    def run(self, args, cwd, env=None):
        # Report command identity, never environment values (which may contain credentials).
        self.log("COMMAND " + " ".join(map(str, args)))
        child_env = (env or os.environ).copy(); child_env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(list(map(str, args)), cwd=cwd, env=child_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                self.log(line.rstrip())
            returncode = process.wait()
        except BaseException:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise
        if returncode:
            raise subprocess.CalledProcessError(returncode, args)
