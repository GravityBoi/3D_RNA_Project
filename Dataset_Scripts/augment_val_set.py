import pandas as pd
import os
import argparse
from pathlib import Path
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def augment_validation_set(
    data_root_dir: str,
    train_sequences_file: str,
    train_labels_file: str,
    val_sequences_file: str,
    val_labels_file: str,
    output_val_sequences_file: str,
    output_val_labels_file: str,
    cutoff_date_str: str,
    max_sequence_length: int = 1000,
    target_id_col_seq: str = "target_id",
    sequence_col: str = "sequence",
    temporal_cutoff_col: str = "temporal_cutoff",
    label_id_col: str = "ID"
):
    """
    Augments an existing validation set with samples from a training set
    that fall on or after a specified temporal cutoff date, after applying
    quality filters to match the dataloader logic.
    """

    root_path = Path(data_root_dir)
    train_seq_path = root_path / train_sequences_file
    train_labels_path = root_path / train_labels_file
    val_seq_path = root_path / val_sequences_file
    val_labels_path = root_path / val_labels_file
    out_val_seq_path = root_path / output_val_sequences_file
    out_val_labels_path = root_path / output_val_labels_file

    try:
        logger.info(f"Loading existing validation sequences from: {val_seq_path}")
        df_val_seq_orig = pd.read_csv(val_seq_path)
        logger.info(f"Loaded {len(df_val_seq_orig)} original validation sequences.")

        logger.info(f"Loading existing validation labels from: {val_labels_path}")
        df_val_labels_orig = pd.read_csv(val_labels_path)
        logger.info(f"Loaded {len(df_val_labels_orig)} original validation label rows.")
    except FileNotFoundError as e:
        logger.error(f"Error: Input validation file not found: {e}")
        return
    
    try:
        logger.info(f"Loading training sequences from: {train_seq_path}")
        df_train_seq_all = pd.read_csv(train_seq_path)
        if temporal_cutoff_col not in df_train_seq_all.columns:
            logger.error(f"Column '{temporal_cutoff_col}' not found in {train_seq_path}. Cannot apply temporal filter.")
            return
        
        logger.info(f"Loading training labels from: {train_labels_path}")
        df_train_labels_all_raw = pd.read_csv(train_labels_path).fillna(-1e18)
        if label_id_col not in df_train_labels_all_raw.columns:
            logger.error(f"Column '{label_id_col}' not found in {train_labels_path}.")
            return
    except FileNotFoundError as e:
        logger.error(f"Error: Input training file not found: {e}")
        return

    # --- Pre-process training labels using the EXACT dataloader logic ---
    logger.info("Pre-processing training labels for quality checks (using dataloader logic)...")
    df_train_labels_all_raw["base_target_id"] = df_train_labels_all_raw[label_id_col].apply(lambda x: "_".join(str(x).split("_")[:-1]))
    df_train_labels_all_raw = df_train_labels_all_raw.sort_values(["base_target_id", "resid"])
    
    name_to_xyz: dict[str, np.ndarray] = {}
    name_to_seq_from_labels: dict[str, str] = {}
    only_top1_gt = True # This is the default in the dataloader

    for tid, grp in df_train_labels_all_raw.groupby("base_target_id"):
        seq_chars, xyz_per_res_list = [], []
        n_gt_max = (
            1 if only_top1_gt
            else max((int(c.split("_")[1]) for c in df_train_labels_all_raw.columns if c.startswith("x_")), default=1)
        )
        for _, row in grp.iterrows():
            seq_chars.append(row["resname"])
            coords_for_this_res = []
            for i in range(1, n_gt_max + 1):
                x, y, z = row.get(f"x_{i}", -1e18), row.get(f"y_{i}", -1e18), row.get(f"z_{i}", -1e18)
                # This logic ensures we capture GTs even if the first one is missing
                if i == 1 and (x == -1e18 and y == -1e18 and z == -1e18):
                    coords_for_this_res = [[-1e18, -1e18, -1e18]] * n_gt_max
                    break
                if x == -1e18 and y == -1e18 and z == -1e18:
                    break 
                coords_for_this_res.append([x,y,z])
            while len(coords_for_this_res) < n_gt_max:
                coords_for_this_res.append([-1e18, -1e18, -1e18])
            xyz_per_res_list.append(coords_for_this_res)
        
        if xyz_per_res_list:
            # Transpose is critical to match the dataloader's array shape
            name_to_xyz[tid] = np.asarray(xyz_per_res_list, dtype=np.float32).transpose(1,0,2)
            name_to_seq_from_labels[tid] = "".join(seq_chars)
    
    # --- Apply Quality Filters (from _SimpleRNADataset __init__) ---
    logger.info(f"Applying quality filters to {len(df_train_seq_all)} training samples (max_length={max_sequence_length})...")
    
    valid_ids = []
    for _, row in df_train_seq_all.iterrows():
        tid = row[target_id_col_seq]
        canonical_sequence = str(row[sequence_col])
        xyz_data_for_tid = name_to_xyz.get(tid)
        seq_from_labels = name_to_seq_from_labels.get(tid)
        
        # 1. Length check (user-requested pre-filter)
        if len(canonical_sequence) > max_sequence_length:
            continue
            
        # 2. Data consistency checks (IDENTICAL to dataloader)
        if (xyz_data_for_tid is None or seq_from_labels is None or
            len(canonical_sequence) != xyz_data_for_tid.shape[1] or
            len(canonical_sequence) != len(seq_from_labels) or
            "-" in canonical_sequence or
            not np.any(xyz_data_for_tid[0, :, 0] > -1e17)):
            continue
            
        valid_ids.append(tid)

    num_before_filter = len(df_train_seq_all)
    df_train_seq_valid = df_train_seq_all[df_train_seq_all[target_id_col_seq].isin(valid_ids)].copy()
    num_after_filter = len(df_train_seq_valid)
    logger.info(f"Quality filtering complete. Kept {num_after_filter} of {num_before_filter} training samples.")

    # --- Apply Temporal Filter on the quality-checked data ---
    logger.info(f"Filtering valid training samples with {temporal_cutoff_col} >= {cutoff_date_str}")
    
    temp_dt_col = temporal_cutoff_col + "_dt_temp" 
    df_train_seq_valid[temp_dt_col] = pd.to_datetime(df_train_seq_valid[temporal_cutoff_col], errors='coerce')
    cutoff_date_dt = pd.to_datetime(cutoff_date_str)

    df_train_to_add_to_val_seq = df_train_seq_valid[
        (df_train_seq_valid[temp_dt_col] >= cutoff_date_dt) & \
        (df_train_seq_valid[temp_dt_col].notna())
    ].copy()
    
    if temp_dt_col in df_train_to_add_to_val_seq.columns:
        df_train_to_add_to_val_seq.drop(columns=[temp_dt_col], inplace=True)
    
    if df_train_to_add_to_val_seq.empty:
        logger.warning("No high-quality training samples found on or after the cutoff date to add to validation.")
        logger.info("Saving original validation files without augmentation.")
        df_val_seq_orig.to_csv(out_val_seq_path, index=False)
        df_val_labels_orig.to_csv(out_val_labels_path, index=False)
        return

    logger.info(f"Found {len(df_train_to_add_to_val_seq)} high-quality training sequences to add to validation set.")
    target_ids_to_add = set(df_train_to_add_to_val_seq[target_id_col_seq].unique())

    df_train_to_add_to_val_labels = df_train_labels_all_raw[df_train_labels_all_raw['base_target_id'].isin(target_ids_to_add)].copy()
    df_train_to_add_to_val_labels.drop(columns=['base_target_id'], inplace=True, errors='ignore')
    logger.info(f"Found {len(df_train_to_add_to_val_labels)} training label rows for the selected sequences.")

    # --- Align columns for sequence files ---
    df_train_to_add_to_val_seq_aligned = pd.DataFrame(columns=df_val_seq_orig.columns)
    for col in df_val_seq_orig.columns:
        if col in df_train_to_add_to_val_seq.columns:
            df_train_to_add_to_val_seq_aligned[col] = df_train_to_add_to_val_seq[col]
        else:
            logger.warning(f"Sequence column '{col}' missing in training subset. Filling with pd.NA.")
            df_train_to_add_to_val_seq_aligned[col] = pd.NA
    df_train_to_add_to_val_seq_aligned = df_train_to_add_to_val_seq_aligned.astype(df_val_seq_orig.dtypes, errors='ignore')

    # --- Align columns for label files ---
    df_train_to_add_to_val_labels_aligned = pd.DataFrame(columns=df_val_labels_orig.columns)
    common_cols_in_labels = list(set(df_val_labels_orig.columns) & set(df_train_to_add_to_val_labels.columns))
    df_train_to_add_to_val_labels_aligned[common_cols_in_labels] = df_train_to_add_to_val_labels[common_cols_in_labels]

    for col in df_val_labels_orig.columns:
        if col not in df_train_to_add_to_val_labels_aligned.columns:
            if col.startswith(('x_', 'y_', 'z_')):
                df_train_to_add_to_val_labels_aligned[col] = -1e18
            else:
                df_train_to_add_to_val_labels_aligned[col] = pd.NA
    
    df_train_to_add_to_val_labels_aligned = df_train_to_add_to_val_labels_aligned[df_val_labels_orig.columns]
    df_train_to_add_to_val_labels_aligned = df_train_to_add_to_val_labels_aligned.astype(df_val_labels_orig.dtypes, errors='ignore')

    # --- Combine and Save ---
    df_val_seq_new = pd.concat([df_val_seq_orig, df_train_to_add_to_val_seq_aligned], ignore_index=True)
    df_val_seq_new.drop_duplicates(subset=[target_id_col_seq], keep='first', inplace=True)
    
    df_val_labels_new = pd.concat([df_val_labels_orig, df_train_to_add_to_val_labels_aligned], ignore_index=True)
    df_val_labels_new.drop_duplicates(subset=[label_id_col], keep='first', inplace=True)

    df_val_seq_new.to_csv(out_val_seq_path, index=False)
    logger.info(f"Saved augmented validation sequences ({len(df_val_seq_new)} total) to: {out_val_seq_path}")

    df_val_labels_new.to_csv(out_val_labels_path, index=False)
    logger.info(f"Saved augmented validation labels ({len(df_val_labels_new)} total rows) to: {out_val_labels_path}")

    logger.info("Augmentation complete.")


def main():
    parser = argparse.ArgumentParser(description="Augment validation set with late training samples, applying dataloader quality filters.")
    parser.add_argument("--data_root_dir", type=str, required=True, help="Root directory for the dataset files.")
    parser.add_argument("--train_sequences_csv", type=str, default="train_sequences.csv", help="Filename of training sequences to filter from.")
    parser.add_argument("--train_labels_csv", type=str, default="train_labels.csv", help="Filename of training labels corresponding to the training sequences.")
    parser.add_argument("--val_sequences_csv", type=str, default="validation_sequences.csv", help="Filename of the original validation sequences to augment.")
    parser.add_argument("--val_labels_csv", type=str, default="validation_labels.csv", help="Filename of the original validation labels to augment.")
    parser.add_argument("--output_val_sequences_csv", type=str, default="validation_sequences_augmented.csv", help="Filename for the new augmented validation sequences.")
    parser.add_argument("--output_val_labels_csv", type=str, default="validation_labels_augmented.csv", help="Filename for the new augmented validation labels.")
    parser.add_argument("--cutoff_date", type=str, default="2022-05-27", help="Temporal cutoff date (YYYY-MM-DD). Samples from training with date on or after this will be added to validation.")
    parser.add_argument("--max_sequence_length", type=int, default=500, help="Maximum sequence length to include in the validation set.")
    
    parser.add_argument("--target_id_col_seq", type=str, default="target_id")
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--temporal_cutoff_col", type=str, default="temporal_cutoff")
    parser.add_argument("--label_id_col", type=str, default="ID")

    args = parser.parse_args()

    augment_validation_set(
        data_root_dir=args.data_root_dir,
        train_sequences_file=args.train_sequences_csv,
        train_labels_file=args.train_labels_csv,
        val_sequences_file=args.val_sequences_csv,
        val_labels_file=args.val_labels_csv,
        output_val_sequences_file=args.output_val_sequences_csv,
        output_val_labels_file=args.output_val_labels_csv,
        cutoff_date_str=args.cutoff_date,
        max_sequence_length=args.max_sequence_length,
        target_id_col_seq=args.target_id_col_seq,
        sequence_col=args.sequence_col,
        temporal_cutoff_col=args.temporal_cutoff_col,
        label_id_col=args.label_id_col
    )

if __name__ == "__main__":
    main()