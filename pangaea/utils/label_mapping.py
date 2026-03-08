import torch

def map_damage_5_to_4(pred: torch.Tensor) -> torch.Tensor:
    """
    Convert 5-class damage predictions to 4-class scheme.

    Model classes
        0 background
        1 intact
        2 minor
        3 major
        4 destroyed

    Target classes
        0 background
        1 intact
        2 damaged
        3 destroyed
    """

    out = torch.empty_like(pred, dtype=torch.int64)

    out[pred == 0] = 0
    out[pred == 1] = 1
    out[(pred == 2) | (pred == 3)] = 2
    out[pred == 4] = 3

    return out