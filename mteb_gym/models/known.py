"""The decision models the library knows by name, each with its family and the version of it that was tested."""

from __future__ import annotations

KNOWN = {
    "autotrust/JEV-9B": ("jev", "7ac669b8fb476ba926491e3d1c39f9330d68dbed"),
    "autotrust/JEV-27B": ("jev", "14b425269ed24452e0a5ad5a8f3e51d139647a88"),
    "autotrust/JEV-35B": ("jev", "a5d0d580bb9324f0f4c07dcba5c5ca91e813e7c9"),
    "autotrust/GEV-26B-Decide": ("jev", "eea849038e792cd09aa731d59d586f0eaf2c18f6"),
    "Mapika/decider-4b": ("decider", "585473dec4fa93f0325c2bf02dda7273b61f44c1"),
    "autotrust/AutoTrust-JEV-27B-v1.1": ("v11", "ae156900e9f338e423042b89e25d0fd8093064d4"),
}


def pinned(name: str) -> str | None:
    """The tested version of a known model. Any other model is at its latest."""
    return KNOWN[name][1] if name in KNOWN else None
