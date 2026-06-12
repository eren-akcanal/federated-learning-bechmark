import re
import matplotlib.pyplot as plt
import os

def parse_and_plot_rounds(log_file_path, output_image_path='fedmuon_rounds.png'):
    rounds = []
    mean_local_test_accs = []
    global_test_accs = []
    
    current_round_locals = []
    current_round = None

    # --- Step 1: Read and Parse the Log File ---
    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                # 1. Detect the start of a new communication round
                round_match = re.search(r"in comm round:(\d+)", line)
                if round_match:
                    # Before switching rounds, process the previous round's collected client data
                    if current_round is not None and current_round_locals:
                        rounds.append(current_round)
                        mean_local_test_accs.append(sum(current_round_locals) / len(current_round_locals))
                        current_round_locals = []
                    
                    current_round = int(round_match.group(1))
                    continue

                # 2. Capture final test accuracy for each individual client local network
                net_match = re.search(r"net \d+ final test acc (\d+\.\d+)", line)
                if net_match:
                    current_round_locals.append(float(net_match.group(1)))
                    continue

                # 3. Capture the final aggregated Global Server Model Test accuracy
                global_match = re.search(r">> Global Model Test accuracy:\s+(\d+\.\d+)", line)
                if global_match:
                    global_test_accs.append(float(global_match.group(1)))
                    
                    # If the log entry didn't explicitly trigger a new round string yet,
                    # force save the current round data array pairings right here
                    if current_round is not None and current_round not in rounds:
                        rounds.append(current_round)
                        if current_round_locals:
                            mean_local_test_accs.append(sum(current_round_locals) / len(current_round_locals))
                        else:
                            mean_local_test_accs.append(float(global_match.group(1))) # Fallback
                        current_round_locals = []

    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found.")
        return

    # Align arrays to the shortest matched layout if data parsing cuts off mid-execution
    min_len = min(len(rounds), len(mean_local_test_accs), len(global_test_accs))
    rounds = rounds[:min_len]
    mean_local_test_accs = mean_local_test_accs[:min_len]
    global_test_accs = global_test_accs[:min_len]

    if min_len == 0:
        print("Error: Could not parse complete round-by-round server or client metric sets from this log file.")
        return

    # --- Step 2: Plotting and Layout Customization ---
    plt.figure(figsize=(8, 5), dpi=150)

    plt.plot(rounds, mean_local_test_accs, label='Mean Local Clients Acc', color='#9467bd', marker='o', linestyle='-', linewidth=2)
    plt.plot(rounds, global_test_accs, label='Global Server Model Acc', color='#e377c2', marker='^', linestyle='--', linewidth=2)

    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    
    # Ensure standard ticks match integer round identifiers
    plt.xticks(rounds)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10, loc='best')
    plt.title('Federated Convergence History (Global vs. Local Accuracies)', fontsize=11, pad=12)

    # --- Step 3: Save Output File ---
    plt.tight_layout()
    output_dir = os.path.dirname(output_image_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    print(f"Success! Performance tracking graph saved to: {output_image_path}")

if __name__ == "__main__":
    parse_and_plot_rounds('./fedmuon/experiment_log-2026-06-11-22:33-19.log', 'fedmuon/rcv1_noniidlabeldir2.png')