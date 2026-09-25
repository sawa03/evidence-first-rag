"""Launch a detached localhost service and wait until it is ready."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
import webbrowser

ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8766"


def ready():
    try:
        with urlopen(URL + "/api/status", timeout=1) as response:
            status = json.load(response)
        return status.get("service") == "evidence-first-rag" and status.get("provider") == "openai"
    except (OSError, ValueError):
        return False


def launch(open_browser=True):
    if not ready():
        with socket.socket() as connection:
            connection.settimeout(1)
            if connection.connect_ex(("127.0.0.1", 8766)) == 0:
                raise RuntimeError("Port 8766 is occupied by another service; it was not stopped.")
        runtime = ROOT / ".runtime"
        runtime.mkdir(exist_ok=True)
        options = {"cwd": str(ROOT), "stdin": subprocess.DEVNULL, "close_fds": True}
        if os.name == "nt":
            options["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        with (runtime / "server.log").open("ab") as log:
            process = subprocess.Popen(
                [sys.executable, str(ROOT / "app.py"), "--provider", "openai", "--port", "8766"],
                stdout=log, stderr=log, **options)
        for _ in range(40):
            if ready():
                break
            if process.poll() is not None:
                raise RuntimeError("Server exited during startup. See .runtime/server.log.")
            time.sleep(.25)
        else:
            raise RuntimeError("Startup timed out. See .runtime/server.log.")
    print("Ready: " + URL)
    if open_browser:
        webbrowser.open(URL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    try:
        launch(not args.no_open)
    except (OSError, RuntimeError) as exc:
        print("Startup failed: " + str(exc), file=sys.stderr)
        sys.exit(1)
