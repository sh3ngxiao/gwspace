from __future__ import annotations

import os

from tianqin_dc.numerical_tq_orbit import DEFAULT_ORBIT_PATH, ORBIT_PATH_ENV, register_numerical_tianqin_orbit


register_numerical_tianqin_orbit()


def main() -> int:
    from tianqin_dc.minimal_catalog_aet import main as minimal_catalog_main

    orbit_path = os.environ.get(ORBIT_PATH_ENV, str(DEFAULT_ORBIT_PATH))
    print(f"Using numerical TianQin orbit from {orbit_path}.", flush=True)
    return minimal_catalog_main()


if __name__ == "__main__":
    raise SystemExit(main())
