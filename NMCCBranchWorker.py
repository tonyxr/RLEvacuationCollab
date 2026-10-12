#!/usr/bin/env python3
"""Isolated simulator-process entry point for exact NMCC branches."""

import os
import sys
import threading
import time


def _exit_if_parent_dies(parent_pid: int) -> None:
    """Prevent a branch worker from surviving an interrupted training run."""
    while True:
        if os.getppid() != int(parent_pid):
            os._exit(70)
        time.sleep(0.5)


if __name__ == "__main__":
    if sys.argv[1:] != ["--persistent"]:
        raise SystemExit("expected --persistent")
    # Reserve a duplicate of stdout for the framed protocol before importing
    # the simulator.  Legacy constructors may print; redirecting the ordinary
    # streams first prevents any text from corrupting the binary channel.
    protocol_output = os.fdopen(
        os.dup(sys.stdout.fileno()),
        "wb",
        buffering=0,
    )
    sys.stdout = open(os.devnull, "w")
    threading.Thread(
        target=_exit_if_parent_dies,
        args=(os.getppid(),),
        daemon=True,
        name="nmcc-parent-watchdog",
    ).start()
    import CounterfactualBranch as CB

    CB.run_persistent_branch_worker(protocol_output=protocol_output)
