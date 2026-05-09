#!/usr/bin/env python3
import sys

from run_scene_daily_tle_batch import main


if __name__ == "__main__":
    raise SystemExit(main(["--scene", "xian_urban", *sys.argv[1:]]))
