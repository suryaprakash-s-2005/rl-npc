"""Headless smoke test for demo/play.py combat demo.

Calls the module's main() with SDL dummy video driver so no window is needed,
and injects QUIT after a few frames by monkeypatching pygame.event.get to
return a FINISHED-andime-quit sequence after N frames.

Usage: python -m demo.smoke_test [model_path]
"""

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame

import demo.play as play

MAX_FRAMES = 30


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else None
    args = [model] if model else []

    real_get = pygame.event.get
    frame = {"n": 0}

    def fake_get():
        events = real_get()
        frame["n"] += 1
        if frame["n"] >= MAX_FRAMES:
            events.append(pygame.event.Event(pygame.QUIT, {}))
        return events

    pygame.event.get = fake_get
    rc = play.main(args)
    print(f"smoke test done: rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())