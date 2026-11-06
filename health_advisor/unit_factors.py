"""Unit conversion factors shared by the normalize and vault tables (#93).

This is a leaf module: it imports nothing from the package, so both
``normalize`` and ``vault`` can build their conversion tables from it without
an import cycle. ``normalize`` cannot import ``vault`` (vault reaches normalize
through metrics), which is why the shared constants live here instead.

Each factor is defined exactly once. Every table that needs one references the
name below, so changing a factor changes every consumer together and no
consumer can drift to a private copy of the literal.
"""

# Exact international definitions (not rounded approximations).
FT_TO_M: float = 0.3048           # international foot, in metres
LB_TO_KG: float = 0.45359237      # avoirdupois pound, in kilograms
MPH_TO_M_PER_S: float = 0.44704   # statute mile per hour, in metres per second
