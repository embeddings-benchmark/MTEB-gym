"""Progress bars for the stages of a run, as mteb shows its own."""

from __future__ import annotations

from tqdm.auto import tqdm


def progress_bar(desc: str, total: int, show: bool, unit: str = "it") -> tqdm:
    """A bar that prints nothing when `show` is off."""
    return tqdm(total=total, desc=desc, unit=unit, disable=not show)
