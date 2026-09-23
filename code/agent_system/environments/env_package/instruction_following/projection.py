from typing import List, Tuple


def instruction_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """Pass instruction-following model outputs through unchanged."""
    return actions, [1] * len(actions)
