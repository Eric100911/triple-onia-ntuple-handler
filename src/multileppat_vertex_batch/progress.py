from __future__ import annotations

from collections.abc import Iterable
from typing import Any


VALID_PROGRESS_BACKENDS = {"auto", "notebook", "terminal", "none"}


def get_tqdm(progress_backend: str):
    backend = (progress_backend or "auto").lower()
    if backend not in VALID_PROGRESS_BACKENDS:
        raise ValueError(
            f"Unsupported progress backend '{progress_backend}'. "
            f"Expected one of {sorted(VALID_PROGRESS_BACKENDS)}."
        )

    if backend == "none":
        return None
    if backend == "terminal":
        from tqdm.std import tqdm as std_tqdm

        return std_tqdm
    if backend == "notebook":
        try:
            from tqdm.notebook import tqdm as notebook_tqdm

            return notebook_tqdm
        except Exception:
            from tqdm.std import tqdm as std_tqdm

            return std_tqdm

    from tqdm.auto import tqdm as auto_tqdm

    return auto_tqdm


def wrap_iterable(
    iterable: Iterable[Any],
    *,
    enabled: bool,
    progress_backend: str,
    **kwargs: Any,
):
    if not enabled:
        return iterable

    tqdm_factory = get_tqdm(progress_backend)
    if tqdm_factory is None:
        return iterable
    return tqdm_factory(iterable, **kwargs)
