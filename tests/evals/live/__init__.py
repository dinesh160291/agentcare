"""The live sweep: the same scenario grammar, replayed against a real provider.

Nothing in this package is collected by pytest — there is no ``test_*.py`` here
on purpose. The sweep is billed and non-deterministic, and a billed check that
runs on every push is a check somebody eventually disables. It is driven by
``scripts/live_sweep.py`` instead, on demand, before and after a change.
"""
