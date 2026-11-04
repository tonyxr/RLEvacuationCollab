"""Comparator plug-ins for ``deployment_strategy='external'`` (v65.1).

Modules here are *not* part of ``POLICY_DYNAMICS_SOURCES``: adding or
editing a comparator never changes the evaluation contract of a trained
policy.  Each class is constructed as ``Class(core=core, params=dict)`` and
must provide ``select(observation, deterministic=True) -> PolicyDecision``
plus an ``INFORMATION_CONTRACT`` dict (role, label, information, timing).

Information rule: read only the public regional observation passed to
``select`` and static public map data reachable from ``core`` at
construction.  Never read ``core.hazard_regime`` (the latent tape) or any
pedestrian state that the observation does not expose.
"""
