#!/usr/bin/env python3
import sys

from run_scene_smoke import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:], default_scene="xian_urban"))
