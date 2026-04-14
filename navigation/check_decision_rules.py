from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from navigation import decide_navigation_action


def main() -> None:
    cases = {
        "stop": torch.tensor([[0.2, 0.5], [0.1, 1.5], [0.0, 2.0]]),
        "turn_right": torch.tensor([[-2.0, 1.0], [-1.8, 2.0], [-1.5, 3.0]]),
        "turn_left": torch.tensor([[2.0, 1.0], [1.8, 2.0], [1.5, 3.0]]),
        "go": torch.tensor([[3.0, 5.0], [3.2, 6.0], [3.5, 7.0]]),
    }
    for expected, pred in cases.items():
        decision = decide_navigation_action(pred)
        assert decision.action == expected, (expected, decision)
        print({"expected": expected, "action": decision.action, "reason": decision.reason})


if __name__ == "__main__":
    main()
