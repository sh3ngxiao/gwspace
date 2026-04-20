# TianQin Data Challenge Prototype

## Design Flow

This prototype follows the pipeline:

1. Sample source parameters from JSON configuration.
2. Resolve each configured source type to a registered source factory.
3. Generate per-source TianQin A/E/T channels through either:
   - GWspace time-domain waveform + TD response, or
   - GWspace frequency-domain response + local FD-to-TD adapter.
4. Sum all injected sources, regardless of source class.
5. Generate TianQin noise from GWspace PSD models.
6. Combine signal and noise.
7. Save channels, injections, labels, configuration, and provenance into one HDF5 file.

## Source Types

- `dwd`
- `gcb`
- `emri`
- `sbbh`
- `smbhb`
- `burst` (kept for backward compatibility with the first prototype)

## Source-Type Relationships

- `dwd` and `gcb` currently share the same GWspace `GCBWaveform` engine.
  - `dwd` is the challenge-level detached-white-dwarf-binary label.
  - `gcb` is kept as the broader GWspace/foreground label.
- `sbbh` and `smbhb` currently share the same compact-binary engine family.
  - By default the resolver uses `bhb_EccFD`.
  - If `PyIMRPhenomD` becomes available later, `engine="bhb_PhenomD"` can be requested in config.
- `emri` uses `GWspace.EMRIWaveform`.

## Important Interface Notes

- `burst` uses `GWspace.BurstWaveform` for `h_plus/h_cross`, then adds a local sky-location adapter because the native GWspace burst class does not expose the geometry interface needed by the response code.
- Compact-binary FD sources are converted to TD A/E/T on the observation rFFT grid in `tianqin_dc/response.py`.
- Noise is generated from `GWspace.Noise.TianQinNoise.noise_AET()`. If future TianQin challenge conventions adopt a different A/E/T normalization, only `tianqin_dc/noise.py` and the FD adapter in `tianqin_dc/response.py` need to change.

## Example Commands

```bash
python -m tianqin_dc.cli --config configs/tianqin_dc/only_dwd.json
python -m tianqin_dc.cli --config configs/tianqin_dc/dwd_smbhb.json
python -m tianqin_dc.cli --config configs/tianqin_dc/dwd_emri.json
python -m tianqin_dc.cli --config configs/tianqin_dc/dwd_sbbh.json
python -m tianqin_dc.cli --config configs/tianqin_dc/all_sources.json
python -m tianqin_dc.cli --config configs/tianqin_dc/all_sources_with_gcb_foreground.json
python -m tianqin_dc.dwd_cli --config configs/tianqin_dc/dwd_catalog_simple.json
python -m tianqin_dc.dwd_cli --config configs/tianqin_dc/dwd_catalog_batch.json
```

## Catalog-Driven DWD Export

For large 8-column DWD source tables such as `tianqin_dc/sim_lisa_part01.txt`, use
`python -m tianqin_dc.dwd_cli`. The exporter maps the columns

- `f0`
- `dfdt_0`
- `b_ecl`
- `l_ecl`
- `Amp`
- `iota`
- `psi`
- `phi_0`

directly onto a local amplitude-driven DWD waveform adapter, without trying to infer
`mass1/mass2/DL`. To avoid accidentally ingesting millions of sources at once, the DWD
catalog config requires either `catalog.row_numbers` or `catalog.rows_per_file`.
