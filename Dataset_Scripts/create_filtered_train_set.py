import pandas as pd
import os
import argparse
from pathlib import Path
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_filtered_training_set(
    data_root_dir: str,
    source_sequences_file: str,
    source_labels_file: str,
    output_train_sequences_file: str,
    output_train_labels_file: str,
    cutoff_date: str,
    max_sequence_length: int = 1000,
    target_id_col_seq: str = "target_id",
    sequence_col: str = "sequence",
    temporal_cutoff_col: str = "temporal_cutoff",
    label_id_col: str = "ID"
):
    """
    Creates a new, high-quality training set by filtering a source dataset
    for samples BEFORE a specific cutoff date and applying quality checks.
    """
    root_path = Path(data_root_dir)
    source_seq_path = root_path / source_sequences_file
    source_labels_path = root_path / source_labels_file
    out_train_seq_path = root_path / output_train_sequences_file
    out_train_labels_path = root_path / output_train_labels_file

    try:
        logger.info(f"Loading source sequences from: {source_seq_path}")
        df_source_seq_all = pd.read_csv(source_seq_path)
        
        logger.info(f"Loading source labels from: {source_labels_path}")
        df_source_labels_all_raw = pd.read_csv(source_labels_path).fillna(-1e18)
    except FileNotFoundError as e:
        logger.error(f"Error: Source file not found: {e}")
        return

    # --- Pre-process labels for quality checks (identical to your script) ---
    df_source_labels_all_raw["base_target_id"] = df_source_labels_all_raw[label_id_col].apply(lambda x: "_".join(str(x).split("_")[:-1]))
    df_source_labels_all_raw = df_source_labels_all_raw.sort_values(["base_target_id", "resid"])
    name_to_xyz, name_to_seq_from_labels = {}, {}
    for tid, grp in df_source_labels_all_raw.groupby("base_target_id"):
        seq_chars, xyz_per_res_list = [], []
        n_gt_max = max((int(c.split("_")[1]) for c in df_source_labels_all_raw.columns if c.startswith("x_")), default=1)
        for _, row in grp.iterrows():
            seq_chars.append(row["resname"])
            coords_for_this_res = [[row.get(f"x_{i}", -1e18), row.get(f"y_{i}", -1e18), row.get(f"z_{i}", -1e18)] for i in range(1, n_gt_max + 1)]
            xyz_per_res_list.append(coords_for_this_res)
        if xyz_per_res_list:
            name_to_xyz[tid] = np.asarray(xyz_per_res_list, dtype=np.float32).transpose(1,0,2)
            name_to_seq_from_labels[tid] = "".join(seq_chars)

    # --- Apply Quality Filters (identical to your script) ---
    logger.info(f"Applying quality filters (max_length={max_sequence_length})...")
    valid_ids = []
    for _, row in df_source_seq_all.iterrows():
        tid, seq = row[target_id_col_seq], str(row[sequence_col])
        xyz_data, seq_from_labels = name_to_xyz.get(tid), name_to_seq_from_labels.get(tid)
        if len(seq) > max_sequence_length: continue
        if (xyz_data is None or seq_from_labels is None or
            len(seq) != xyz_data.shape[1] or len(seq) != len(seq_from_labels) or
            "-" in seq or not np.any(xyz_data[0, :, 0] > -1e17)):
            continue
        valid_ids.append(tid)

    df_quality_filtered_seq = df_source_seq_all[df_source_seq_all[target_id_col_seq].isin(valid_ids)].copy()
    logger.info(f"Kept {len(df_quality_filtered_seq)} of {len(df_source_seq_all)} samples after quality filtering.")

    # --- Apply Temporal Filter (THE KEY CHANGE) ---
    logger.info(f"Filtering high-quality samples BEFORE {cutoff_date}")
    df_quality_filtered_seq['dt_col'] = pd.to_datetime(df_quality_filtered_seq[temporal_cutoff_col], errors='coerce')
    cutoff_dt = pd.to_datetime(cutoff_date)
    
    # We now select all samples where the date is LESS THAN the cutoff.
    df_new_train_seq = df_quality_filtered_seq[
        (df_quality_filtered_seq['dt_col'] < cutoff_dt) &
        (df_quality_filtered_seq['dt_col'].notna())
    ].copy().drop(columns=['dt_col'])

    if df_new_train_seq.empty:
        logger.warning(f"No sequences found before {cutoff_date}. No training set created.")
        return

    logger.info(f"Found {len(df_new_train_seq)} sequences for the new filtered training set.")
    
    # --- Get Corresponding Labels ---
    final_target_ids = set(df_new_train_seq[target_id_col_seq])
    df_new_train_labels = df_source_labels_all_raw[df_source_labels_all_raw['base_target_id'].isin(final_target_ids)].copy()
    df_new_train_labels.drop(columns=['base_target_id'], inplace=True, errors='ignore')

    # --- Save New Files ---
    df_new_train_seq.to_csv(out_train_seq_path, index=False)
    logger.info(f"Saved filtered training sequences ({len(df_new_train_seq)} total) to: {out_train_seq_path}")
    
    df_new_train_labels.to_csv(out_train_labels_path, index=False)
    logger.info(f"Saved filtered training labels ({len(df_new_train_labels)} total rows) to: {out_train_labels_path}")

def main():
    parser = argparse.ArgumentParser(description="Create a clean, filtered training set from a source dataset based on a time cutoff.")
    parser.add_argument("--data_root_dir", type=str, required=True, help="Root directory for the dataset files.")
    parser.add_argument("--source_sequences_csv", type=str, default="train_sequences.csv", help="Source sequences file to filter from (v1).")
    parser.add_argument("--source_labels_csv", type=str, default="train_labels.csv", help="Source labels file (v1).")
    parser.add_argument("--output_train_sequences_csv", type=str, default="train_sequences_filtered.csv", help="Filename for the new training sequences.")
    parser.add_argument("--output_train_labels_csv", type=str, default="train_labels_filtered.csv", help="Filename for the new training labels.")
    parser.add_argument("--cutoff_date", type=str, required=True, help="Temporal cutoff date (YYYY-MM-DD). All data BEFORE this date will be included.")
    parser.add_argument("--max_sequence_length", type=int, default=500, help="Maximum sequence length to include.")
    
    args = parser.parse_args()

    create_filtered_training_set(
        data_root_dir=args.data_root_dir,
        source_sequences_file=args.source_sequences_csv,
        source_labels_file=args.source_labels_csv,
        output_train_sequences_file=args.output_train_sequences_csv,
        output_train_labels_file=args.output_train_labels_csv,
        cutoff_date=args.cutoff_date,
        max_sequence_length=args.max_sequence_length
    )

if __name__ == "__main__":
    main()