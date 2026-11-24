"""Print task.motion_speed of a FlashCHORD experiment recipe (flash_chord/configs/experiment/<exp>.yaml, its one owner);
the CHORD scoring line passes it as the scorer's time scale. Stops when the key is missing.
usage: motion_speed.py <experiment yaml>"""

import re
import sys
from pathlib import Path


def motion_speed(txt: str) -> str:
    m = re.search(r"^task:\s*\n(?:[ \t]+.*\n)*?[ \t]+motion_speed:\s*([0-9.]+)", txt, re.M)
    if not m:
        sys.exit("[CHORD_SPEED] no task.motion_speed in the recipe")
    return m.group(1)


if __name__ == "__main__":
    print(motion_speed(Path(sys.argv[1]).read_text()))
