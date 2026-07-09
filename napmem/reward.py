from __future__ import annotations


def napmem_reward(format_valid: bool, correct: bool, memory_used: bool, use_usage_bonus: bool = True) -> float:
    """NapMem terminal reward with optional ablation of the U term.

    Paper reward:
    - malformed: -1
    - correct + used memory: 1
    - correct + no memory: 0
    - wrong + used memory: -0.5
    - wrong + no memory: -1

    With use_usage_bonus=False, correctness is rewarded without incentivizing
    tool calls for their own sake. This is the first honest-finding lane.
    """

    if not format_valid:
        return -1.0
    if use_usage_bonus:
        if correct and memory_used:
            return 1.0
        if correct and not memory_used:
            return 0.0
        if not correct and memory_used:
            return -0.5
        return -1.0
    return 1.0 if correct else -1.0
