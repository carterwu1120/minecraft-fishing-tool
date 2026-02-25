from __future__ import annotations

import argparse

from .agent import FishingAgent
from .config import FishingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR screen monitor and auto-click tool for fishing workflow."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config json file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FishingConfig.from_file(args.config)
    agent = FishingAgent(config)
    agent.run()


if __name__ == "__main__":
    main()
