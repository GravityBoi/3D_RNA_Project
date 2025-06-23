import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

def parse_log_file(filepath):
    """
    Parses a log file to extract training, validation, and EMA validation loss data.
    
    Args:
        filepath (str): The path to the log file.
        
    Returns:
        tuple: A tuple containing three lists: (train_data, val_data, ema_val_data)
               Each list contains tuples of (step, loss).
    """
    # Regex to find the different loss lines
    train_pattern = re.compile(r"INFO root: Step (\d+) train: .*?'train/loss\.avg': ([\d.]+)")
    val_pattern = re.compile(r"INFO root: Step (\d+), eval kaggle_val: .*?'kaggle_val/mse_loss\.avg': ([\d.]+)")
    ema_val_pattern = re.compile(r"INFO root: Step (\d+), eval kaggle_val: .*?'kaggle_val/ema.*?_mse_loss\.avg': ([\d.]+)")

    train_data = []
    val_data = []
    ema_val_data = []
    
    seen_train_steps = set()
    seen_val_steps = set()
    seen_ema_val_steps = set()

    try:
        with open(filepath, 'r') as f:
            for line in f:
                # Search for training loss
                train_match = train_pattern.search(line)
                if train_match:
                    step = int(train_match.group(1))
                    loss = float(train_match.group(2))
                    if step not in seen_train_steps:
                        train_data.append((step, loss))
                        seen_train_steps.add(step)
                
                # Search for standard validation loss
                val_match = val_pattern.search(line)
                if val_match:
                    step = int(val_match.group(1))
                    loss = float(val_match.group(2))
                    # This check ensures we don't accidentally match the EMA line
                    if "'kaggle_val/loss.avg'" in line and step not in seen_val_steps:
                        val_data.append((step, loss))
                        seen_val_steps.add(step)
                
                # Search for EMA validation loss
                ema_val_match = ema_val_pattern.search(line)
                if ema_val_match:
                    step = int(ema_val_match.group(1))
                    loss = float(ema_val_match.group(2))
                    if step not in seen_ema_val_steps:
                        ema_val_data.append((step, loss))
                        seen_ema_val_steps.add(step)

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return [], [], []
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return [], [], []

    # Sort data by step number
    train_data.sort()
    val_data.sort()
    ema_val_data.sort()
    
    return train_data, val_data, ema_val_data

def plot_training_loss(train_data, save_path="training_loss.png"):
    """
    Generates and saves a plot for training loss.
    
    Args:
        train_data (list): A list of (step, loss) tuples for training.
        save_path (str): The path to save the output plot image.
    """
    if not train_data:
        print("No training data to plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    train_steps, train_losses = zip(*train_data)
    
    ax.plot(train_steps, train_losses, label='Training Loss', color='dodgerblue', linewidth=2)
    ax.set_yscale('log')
    ax.set_title('Training Loss Curve', fontsize=20, weight='bold')
    ax.set_xlabel('Training Steps', fontsize=18)
    ax.set_ylabel('Loss (log scale)', fontsize=18)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", ls="--", c='0.7')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close(fig) # Close the figure to free up memory
    print(f"Training loss plot saved to '{save_path}'")

def plot_validation_loss(val_data, ema_val_data, save_path="validation_loss.png"):
    """
    Generates and saves a plot comparing validation and EMA validation loss.
    
    Args:
        val_data (list): A list of (step, loss) tuples for validation.
        ema_val_data (list): A list of (step, loss) tuples for EMA validation.
        save_path (str): The path to save the output plot image.
    """
    if not val_data and not ema_val_data:
        print("No validation data to plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    if val_data:
        val_steps, val_losses = zip(*val_data)
        ax.plot(val_steps, val_losses, label='Validation Loss', color='orangered', marker='o', linestyle='-', linewidth=2)

    if ema_val_data:
        ema_steps, ema_losses = zip(*ema_val_data)
        ax.plot(ema_steps, ema_losses, label='EMA Validation Loss', color='forestgreen', marker='s', linestyle='--', linewidth=2)

    ax.set_title('Validation Loss Comparison', fontsize=10, weight='bold')
    ax.set_xlabel('Training Steps', fontsize=18)
    ax.set_ylabel('Loss', fontsize=18)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", ls="--", c='0.7')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, p: format(int(y), ',')))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close(fig) # Close the figure to free up memory
    print(f"Validation loss plot saved to '{save_path}'")

# --- Main Execution ---
if __name__ == "__main__":
    # Define the path to your log file
    log_filepath = "/home/max/Documents/ProtenixFinetuningFinalResults/NoMSARUN_proto_ft_34945243.err"

    if os.path.exists(log_filepath):
        # Parse the log file to get the data
        training_data, validation_data, ema_validation_data = parse_log_file(log_filepath)
        
        # Generate and save the plots
        plot_training_loss(training_data)
        plot_validation_loss(validation_data, ema_validation_data)
    else:
        print(f"Error: File not found at the specified path: {log_filepath}")
        print("Please ensure the file path is correct and you have read permissions.")