from __future__ import annotations

import argparse

from tianqin_dc.dwd_export import (
    SimpleDWDBuilder,
    SimpleDWDConfig,
    load_simple_dwd_config,
    save_simple_dwd_hdf5,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a simple TianQin DWD XYZ HDF5 file from a source table.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    parser.add_argument(
        "--output",
        help="Optional output path override. If given, it replaces output.path in the config file.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config, raw_config = load_simple_dwd_config(args.config)
    if args.output:
        raw_config = dict(raw_config)
        raw_output = dict(raw_config.get("output", {}))
        raw_output["path"] = args.output
        raw_config["output"] = raw_output
        config = SimpleDWDConfig.from_config(raw_config)

    builder = SimpleDWDBuilder(config, raw_config)
    bundle = builder.build()
    output_path = save_simple_dwd_hdf5(bundle, config)

    print(f"Wrote simple DWD dataset to {output_path}")
    print(f"Selected sources: {len(bundle.selected_sources)}")
    print("Channels: X, Y, Z")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
