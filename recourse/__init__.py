"""Forward-looking evacuation orders (recourse) for shelters about to be overrun.

Evaluation-layer extension of the frozen v65.1 model: no file of the dynamics
fingerprint is edited. ``controller`` decides, ``transaction`` executes through
the simulator's own closure transaction, ``rollout`` grades order timing on
public-belief futures, ``runtime`` wires the controller into live episodes.
"""

from .controller import RecourseParams, ShelterRecourseController
from .runtime import LIVE, install
from .transaction import execute_evacuation_orders

__all__ = ["LIVE", "RecourseParams", "ShelterRecourseController", "execute_evacuation_orders", "install"]
