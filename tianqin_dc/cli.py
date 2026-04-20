from __future__ import annotations

import argparse

from tianqin_dc.builder import TianQinDatasetBuilder
from tianqin_dc.config import RunConfig, load_run_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate TianQin mock data on top of GWspace.")
    parser.add_argument("--config", required=True, help="Path to a JSON run configuration file.")
    parser.add_argument(
        "--output",
        help="Optional output path override. If given, it replaces output.path in the config file.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config, raw_config = load_run_config(args.config)
    if args.output:
        raw_config = dict(raw_config)
        raw_output = dict(raw_config.get("output", {}))
        raw_output["path"] = args.output
        raw_config["output"] = raw_output
        config = RunConfig.from_config(raw_config)

    builder = TianQinDatasetBuilder(config, raw_config)
    output_path = builder.build_and_save()

    print(f"Wrote dataset to {output_path}")
    print(f"Channels: {', '.join(config.observation.channels)}")
    total_sources = sum(
        (len(pop.parameters) if pop.parameters is not None else pop.count)
        for pop in config.sources
        if pop.enabled
    )
    print(f"Sampled sources: {total_sources}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
