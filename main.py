from __future__ import annotations

import warnings

from preprocess import main as preprocess_main


def main() -> None:
    warnings.warn(
        "main.py is deprecated. Use preprocess.py as the canonical preprocessing entrypoint.",
        DeprecationWarning,
        stacklevel=2,
    )
    preprocess_main()


if __name__ == "__main__":
    main()