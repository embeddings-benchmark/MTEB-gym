"""python -m mteb_gym.leaderboard [--results PATH_OR_URL]"""

import argparse

from .app import get_leaderboard_app

parser = argparse.ArgumentParser(prog="python -m mteb_gym.leaderboard")
parser.add_argument("--results", help="a results folder, or a results repository's URL (default: gym-results)")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=7860)
args = parser.parse_args()
get_leaderboard_app(args.results).launch(server_name=args.host, server_port=args.port)
