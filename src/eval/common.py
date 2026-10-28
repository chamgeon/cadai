"""Shared by the step 8 evals: API cost and memory footprint."""
import re
import subprocess

import psutil

# $ per million tokens: input, output, cache read, cache write (5-minute TTL = 1.25 x input)
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
}


def cost(usage, model):
    inp, out, read, write = PRICES[model]
    return (usage.get("input_tokens", 0) * inp + usage.get("output_tokens", 0) * out
            + usage.get("cache_read_input_tokens", 0) * read
            + usage.get("cache_creation_input_tokens", 0) * write) / 1e6


def _footprint_mb(pid):
    """Activity Monitor's "Memory": resident + compressed. Plain RSS drops as macOS compresses pages on an
    8 GB machine, so it can't show growth."""
    line = subprocess.run(["top", "-l", "1", "-pid", str(pid), "-stats", "pid,mem"],
                          capture_output=True, text=True).stdout.strip().splitlines()[-1]
    n, unit = re.fullmatch(r"\d+\s+([\d.]+)([BKMG])\+?", line.strip()).groups()
    return float(n) * {"B": 2**-20, "K": 2**-10, "M": 1, "G": 2**10}[unit]


def footprint_mb(pid=None):
    """Memory footprint of a process and its children (the UI's CAD worker), MB."""
    p = psutil.Process(pid)
    return {"self": _footprint_mb(p.pid),
            "children": sum(_footprint_mb(c.pid) for c in p.children(recursive=True))}
