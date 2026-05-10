#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DWD_JOB="$(sbatch --parsable "$SCRIPT_DIR/run_minimal_dwd_signal_all_aet.slurm.sh")"
SMBHB_JOB="$(sbatch --parsable "$SCRIPT_DIR/run_minimal_smbhb_signal_all_aet.slurm.sh")"
SBBH_JOB="$(sbatch --parsable "$SCRIPT_DIR/run_minimal_sbbh_signal_all_aet.slurm.sh")"
EMRI_JOB="$(sbatch --parsable "$SCRIPT_DIR/run_minimal_emri_signal_all_aet.slurm.sh")"
SGWB_JOB="$(sbatch --parsable "$SCRIPT_DIR/run_minimal_sgwb_signal_all_aet.slurm.sh")"

MERGE_DEPENDENCY="afterok:${DWD_JOB}:${SMBHB_JOB}:${SBBH_JOB}:${EMRI_JOB}:${SGWB_JOB}"
MERGE_CONFIG="${MERGE_CONFIG:-configs/tianqin_dc/tianqin_tdi2_dt1s/minimal_all_sources_plus_sgwb_merged_aet.json}"
MERGE_JOB="$(CONFIG="$MERGE_CONFIG" sbatch --parsable --dependency="$MERGE_DEPENDENCY" "$SCRIPT_DIR/run_minimal_all_sources_merged_aet.slurm.sh")"

echo "Submitted minimal A/E/T signal jobs:"
echo "  DWD:   $DWD_JOB"
echo "  SMBHB: $SMBHB_JOB"
echo "  SBBH:  $SBBH_JOB"
echo "  EMRI:  $EMRI_JOB"
echo "  SGWB:  $SGWB_JOB"
echo "Submitted merge job:"
echo "  MERGE: $MERGE_JOB"
