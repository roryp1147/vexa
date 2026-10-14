#!/usr/bin/env python3
"""Pre-render Vexa's procedural UI sounds into assets/sounds/*.wav."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from player.sfx_gen import generate_all  # noqa: E402

if __name__ == "__main__":
    out = os.path.join(ROOT, "assets", "sounds")
    for k, v in generate_all(out).items():
        print(f"{k:12s} {os.path.getsize(v):7d} B  {v}")
