import argparse
import json
import pathlib
import pandas as pd
from Bio import AlignIO
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_msas_to_stockholm(
    msa_fasta_dir: pathlib.Path,
    output_stockholm_dir: pathlib.Path,
    expected_msa_filename_pattern: str = "*.MSA.fasta"
):
    """
    Converts FASTA MSA files to Stockholm format and organizes them.
    Each FASTA MSA is converted into output_stockholm_dir/{target_id}/rna_align.sto.

    Args:
        msa_fasta_dir: Path to the directory containing .MSA.fasta files.
        output_stockholm_dir: Path to the directory where Stockholm files will be saved.
        expected_msa_filename_pattern: Glob pattern to find MSA files.
    """
    logger.info(f"Converting FASTA alignments from '{msa_fasta_dir}' to Stockholm format in '{output_stockholm_dir}'...")

    fasta_files = list(msa_fasta_dir.glob(expected_msa_filename_pattern))
    if not fasta_files:
        logger.warning(f"No '{expected_msa_filename_pattern}' files found in {msa_fasta_dir}. Skipping MSA conversion.")
        return

    output_stockholm_dir.mkdir(parents=True, exist_ok=True)

    for fasta_file_path in tqdm(fasta_files, desc="MSA FASTA → Stockholm", unit="file"):
        # Assumes filename like: {target_id}.MSA.fasta
        target_id = fasta_file_path.stem.split(".")[0]
        if not target_id:
            logger.warning(f"Could not extract target_id from {fasta_file_path.name}. Skipping.")
            continue

        msa_target_out_dir = output_stockholm_dir / target_id
        msa_target_out_dir.mkdir(parents=True, exist_ok=True)

        # Standardized name. Protenix's RNAMSAFeaturizer will be slightly modified
        # to look for this specific file instead of iterating db_names.
        sto_path = msa_target_out_dir / "rna_align.sto"

        try:
            alignment = AlignIO.read(fasta_file_path, "fasta")
            AlignIO.write(alignment, sto_path, "stockholm")
        except ValueError as ve:
            logger.error(f"Skipping {fasta_file_path.name} due to parsing error (possibly empty, malformed, or not a valid alignment): {ve}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while processing {fasta_file_path.name}: {e}")

    logger.info(f"Stockholm conversion complete. Files saved in {output_stockholm_dir}")


def create_sequence_to_target_map(
    sequences_csv_path: pathlib.Path,
    output_json_path: pathlib.Path,
    sequence_col: str = "sequence",
    target_id_col: str = "target_id"
):
    """
    Creates a JSON file mapping RNA sequences to a list of their target_ids.
    This is analogous to Protenix's seq_to_pdb_index.json but adapted for RNA target_ids.

    Args:
        sequences_csv_path: Path to the CSV file containing sequence and target_id columns.
        output_json_path: Path where the output JSON mapping will be saved.
        sequence_col: Name of the column containing RNA sequences in the CSV.
        target_id_col: Name of the column containing target IDs in the CSV.
    """
    logger.info(f"Creating sequence-to-target_id map from '{sequences_csv_path}'...")
    try:
        df = pd.read_csv(sequences_csv_path)
    except FileNotFoundError:
        logger.error(f"Error: Sequences CSV file not found at {sequences_csv_path}")
        return
    except Exception as e:
        logger.error(f"Error reading {sequences_csv_path}: {e}")
        return

    if not {sequence_col, target_id_col}.issubset(df.columns):
        logger.error(f"Error: {sequences_csv_path} must contain '{sequence_col}' and '{target_id_col}' columns.")
        return

    seq_to_target_ids = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building seq→target_id map"):
        sequence = row[sequence_col]
        target_id = row[target_id_col]
        if pd.isna(sequence) or pd.isna(target_id):
            logger.warning(f"Skipping row with NaN sequence or target_id: {row}")
            continue
        seq_to_target_ids.setdefault(str(sequence), []).append(str(target_id))

    # Ensure parent directory of the output JSON file exists
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w") as fh:
        json.dump(seq_to_target_ids, fh, indent=2)

    logger.info(f"Sequence-to-target_id map saved to '{output_json_path}'")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare RNA MSA data for Protenix finetuning. "
                    "Converts FASTA MSAs to Stockholm format and creates a sequence-to-target_id mapping."
    )
    parser.add_argument(
        "--msa_input_dir",
        type=str,
        required=True,
        help="Directory containing the input FASTA MSA files (e.g., your 'MSA' or 'MSA_v2' folder from Kaggle)."
    )
    parser.add_argument(
        "--sequences_csv",
        type=str,
        required=True,
        help="Path to the train_sequences.csv (or similar) file containing RNA sequences and their target_ids."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ProtenixRNA_MSA_data",
        help="Main output directory. Stockholm MSAs will be in 'output_dir/RNA_MSA_Stockholm' "
             "and the JSON map in 'output_dir/seq_to_target_map.json'."
    )
    parser.add_argument(
        "--msa_filename_pattern",
        type=str,
        default="*.MSA.fasta",
        help="Glob pattern for matching MSA files in the msa_input_dir (e.g., '*.MSA.fasta' or '*.fasta')."
    )
    parser.add_argument(
        "--sequence_column_name",
        type=str,
        default="sequence",
        help="Name of the column containing sequences in the sequences_csv."
    )
    parser.add_argument(
        "--target_id_column_name",
        type=str,
        default="target_id",
        help="Name of the column containing target IDs in the sequences_csv."
    )

    args = parser.parse_args()

    msa_fasta_dir = pathlib.Path(args.msa_input_dir)
    sequences_csv_path = pathlib.Path(args.sequences_csv)
    main_output_dir = pathlib.Path(args.output_dir)

    output_stockholm_dir = main_output_dir / "RNA_MSA_Stockholm"
    output_json_path = main_output_dir / "seq_to_target_map.json" # Protenix expects `seq_to_pdb_idx_path` but this name is fine for the file itself

    # Step 1: Convert FASTA MSAs to Stockholm format
    convert_msas_to_stockholm(msa_fasta_dir, output_stockholm_dir, args.msa_filename_pattern)

    # Step 2: Create sequence to target_id mapping JSON
    create_sequence_to_target_map(sequences_csv_path, output_json_path, args.sequence_column_name, args.target_id_column_name)

    logger.info("\nProcessing finished.")
    logger.info(f"  Stockholm MSAs are in: {output_stockholm_dir.resolve()}")
    logger.info(f"  Sequence to target_id map is at: {output_json_path.resolve()}")
    
if __name__ == "__main__":
    main()