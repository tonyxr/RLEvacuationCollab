"""Exogenous scenario hooks for evaluation panels.

A hook edits the exogenous event schedule of a freshly initialized episode,
before its first transition, through the simulator's existing event
mechanism (``HazardRegimeProcess.scheduled``, the same list the constructed
decoy scenario writes to). No file in ``POLICY_DYNAMICS_SOURCES`` changes, so
frozen checkpoints are evaluated on frozen physics; only the exogenous input
differs. ``scripts/panel_eval.py`` loads a hook named in a panel spec, puts
its parameters and the digest of this package into the episode key, applies it
identically for every strategy on a tape, and records what it did.
"""

from .initial_site_loss import InitialSiteLoss

__all__ = ["InitialSiteLoss"]
