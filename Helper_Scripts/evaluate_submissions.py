import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import subprocess
from multiprocessing import Pool, cpu_count

# =============================================================================
# Configuration
# =============================================================================

SUBMISSION_FILES = [
    "/home/max/Documents/ProtenixFinetuningFinalResults/submission_5999_EMA_MSA.csv",
    "/home/max/Documents/ProtenixFinetuningFinalResults/submission_5999_EMA_NoMSA.csv",
    "/home/max/Documents/ProtenixFinetuningFinalResults/submission_5999_MSA.csv",
    "/home/max/Documents/ProtenixFinetuningFinalResults/submission_5999_NoMSA.csv",
    "/home/max/Documents/ProtenixFinetuningFinalResults/submission_9999_EMA_NoMSA.csv",
    "/home/max/Documents/ProtenixFinetuningFinalResults/submission_9999_NoMSA.csv"
]
SOLUTION_FILE = "/home/max/Documents/Protenix-KaggleRNA3D/data/stanford-rna-3d-folding/validation_labels_clean.csv"
USALIGN_EXECUTABLE = "/home/max/Documents/Protenix-KaggleRNA3D/af3-dev/USalign/USalign"
TEMP_DIR = "./scoring_temp/"

# =============================================================================
# Helper Functions (Using your provided robust versions)
# =============================================================================

def parse_tmscore_output(output: str) -> float:
    tm_score_match = re.findall(r'TM-score=\s+([\d.]+)', output)
    if len(tm_score_match) > 1:
        return float(tm_score_match[1])
    return 0.0 # Return 0.0 if parsing fails

def write_target_line(atom_name, atom_serial, residue_name, chain_id, residue_num,
                      x_coord, y_coord, z_coord, occupancy=1.0, b_factor=0.0, atom_type='C'):
    atom_name_padded = f" {atom_name.ljust(3)}" if len(atom_name) < 4 else atom_name
    return (
        f"ATOM  {atom_serial:5d} {atom_name_padded:<4s} {residue_name:<3s} {chain_id}"
        f"{residue_num:4d}    {x_coord:8.3f}{y_coord:8.3f}{z_coord:8.3f}"
        f"{occupancy:6.2f}{b_factor:6.2f}          {atom_type:>2s}  \n"
    )

def write2pdb(df: pd.DataFrame, xyz_id: int, target_path: str) -> int:
    resolved_cnt = 0
    written_resids = set()
    with open(target_path, 'w') as f:
        for _, row in df.iterrows():
            resid = int(row['resid'])
            if resid in written_resids: continue
            x, y, z = row.get(f'x_{xyz_id}'), row.get(f'y_{xyz_id}'), row.get(f'z_{xyz_id}')
            if pd.notna(x) and x > -1e17:
                resolved_cnt += 1
                f.write(write_target_line("C1'", resid, row['resname'], 'A', resid, x, y, z))
    return resolved_cnt

def get_base_target_id(long_id):
    return "_".join(str(long_id).split("_")[:-1])

# --- New function for parallel execution ---
def run_usalign_command(command):
    """Executes a single USalign command and returns the parsed TM-score."""
    try:
        # Use subprocess.run for better error handling and capturing output
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return parse_tmscore_output(result.stdout)
    except subprocess.CalledProcessError as e:
        # This handles cases where USalign itself fails
        print(f"Warning: USalign command failed for command '{command}': {e.stderr}", file=sys.stderr)
        return 0.0
    except Exception as e:
        print(f"An unexpected error occurred running command '{command}': {e}", file=sys.stderr)
        return 0.0

# =============================================================================
# Core Scoring Function (Rewritten for Speed and Parallelism)
# =============================================================================

def score_submission_parallel(solution_df: pd.DataFrame, submission_df: pd.DataFrame) -> dict:
    solution_df['target_id'] = solution_df['ID'].apply(get_base_target_id)
    submission_df['target_id'] = submission_df['ID'].apply(get_base_target_id)

    native_idxs = sorted(int(c.split('_')[1]) for c in solution_df.columns if c.startswith('x_'))
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    all_target_scores = []
    common_targets = sorted(list(set(solution_df['target_id'].unique()) & set(submission_df['target_id'].unique())))
    
    print(f"\nScoring {len(common_targets)} common targets...")
    for tid in tqdm(common_targets, desc="Scoring Targets"):
        grp_nat = solution_df[solution_df['target_id'] == tid]
        grp_pred = submission_df[submission_df['target_id'] == tid]
        
        # --- Stage 1: Write all PDB files with unique names FIRST ---
        native_paths = {}
        for nat_cnt in native_idxs:
            path = os.path.join(TEMP_DIR, f"native_{tid}_{nat_cnt}.pdb")
            if write2pdb(grp_nat, nat_cnt, path) > 0:
                native_paths[nat_cnt] = path
        
        predicted_paths = {}
        for pred_cnt in range(1, 6):
            path = os.path.join(TEMP_DIR, f"predicted_{tid}_{pred_cnt}.pdb")
            if write2pdb(grp_pred, pred_cnt, path) > 0:
                predicted_paths[pred_cnt] = path

        # --- Stage 2: Generate all USalign commands ---
        commands_to_run = []
        # This will map a command back to its prediction index (1-5)
        command_to_pred_idx = {} 

        for pred_cnt, pred_path in predicted_paths.items():
            for nat_cnt, nat_path in native_paths.items():
                command = f'{USALIGN_EXECUTABLE} {pred_path} {nat_path} -atom " C1\'"'
                commands_to_run.append(command)
                command_to_pred_idx[command] = pred_cnt

        if not commands_to_run:
            all_target_scores.append([0.0] * 5)
            continue
            
        # --- Stage 3: Execute all commands in parallel ---
        num_cores = cpu_count()
        with Pool(processes=num_cores) as pool:
            tm_scores = list(pool.imap(run_usalign_command, commands_to_run))

        # --- Stage 4: Aggregate results ---
        scores_by_pred_idx = {i: [] for i in range(1, 6)}
        for command, score in zip(commands_to_run, tm_scores):
            pred_idx = command_to_pred_idx[command]
            scores_by_pred_idx[pred_idx].append(score)

        scores_for_this_target = []
        for pred_cnt in range(1, 6):
            # If a prediction had scores, take the max. Otherwise, it's 0.
            best_for_this_pred = max(scores_by_pred_idx[pred_cnt]) if scores_by_pred_idx[pred_cnt] else 0.0
            scores_for_this_target.append(best_for_this_pred)
            
        all_target_scores.append(scores_for_this_target)

    # --- Aggregate Statistics ---
    if not all_target_scores:
        return {"error": "No common targets found or scores calculated."}

    all_individual_scores = [score for target_list in all_target_scores for score in target_list]
    best_of_5_scores = [max(target_list) if target_list else 0.0 for target_list in all_target_scores]
    competition_score = np.mean(best_of_5_scores)

    return {
        'competition_score': competition_score,
        'mean_all_preds': np.mean(all_individual_scores),
        'median_all_preds': np.median(all_individual_scores),
        'std_dev_all_preds': np.std(all_individual_scores),
        'mean_of_bests': np.mean(best_of_5_scores),
        'std_dev_of_bests': np.std(best_of_5_scores)
    }

# =============================================================================
# Main Execution Logic
# =============================================================================

def main():
    print("--- Starting Batch Evaluation of Submission Files (Parallel Version) ---")
    
    try:
        solution_df = pd.read_csv(SOLUTION_FILE)
    except FileNotFoundError:
        print(f"FATAL ERROR: Solution file not found at '{SOLUTION_FILE}'", file=sys.stderr)
        return

    all_results = []
    for sub_file in SUBMISSION_FILES:
        print(f"\n{'='*20} Evaluating: {os.path.basename(sub_file)} {'='*20}")
        try:
            submission_df = pd.read_csv(sub_file)
        except FileNotFoundError:
            print(f"  -> SKIPPING: Submission file not found at '{sub_file}'", file=sys.stderr)
            continue
        
        result_stats = score_submission_parallel(solution_df.copy(), submission_df.copy())
        
        if "error" in result_stats:
            print(f"  -> ERROR: {result_stats['error']}", file=sys.stderr)
            continue

        result_stats['submission_file'] = os.path.basename(sub_file)
        all_results.append(result_stats)
        print(f"  -> Competition Score: {result_stats.get('competition_score', 'N/A'):.4f}")
        print(f"  -> Mean Score (all preds): {result_stats.get('mean_all_preds', 'N/A'):.4f}")

    if not all_results:
        print("\nNo submission files were successfully evaluated.")
        return
        
    results_df = pd.DataFrame(all_results)
    column_order = ['submission_file', 'competition_score', 'mean_all_preds', 
                    'median_all_preds', 'std_dev_all_preds', 'std_dev_of_bests']
    results_df = results_df[[col for col in column_order if col in results_df.columns]]
    
    print("\n\n" + "="*60)
    print("           >>> FINAL BATCH EVALUATION REPORT <<<")
    print("="*60)
    print(results_df.to_string(index=False))
    print("\n'competition_score' is the official mean best-of-5 metric.")
    print("'mean_all_preds' is the average over all 5 predictions per target.")


if __name__ == "__main__":
    main()