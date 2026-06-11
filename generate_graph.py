import re
import matplotlib.pyplot as plt

def parse_and_plot_log(log_file_path, output_image_path='fedsap_performance.png'):
    rounds = []
    mean_local_accs = []
    global_accs = []
    current_round_locals = []

    # --- Step 1: Read and Parse the Log File ---
    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                # Detect the start of a new communication round
                round_match = re.search(r"in comm round:(\d+)", line)
                if round_match:
                    if current_round_locals:
                        mean_local_accs.append(sum(current_round_locals) / len(current_round_locals))
                        current_round_locals = []
                    rounds.append(int(round_match.group(1)))
                    
                # Capture individual client network test accuracies
                net_match = re.search(r"net \d+ final test acc (\d+\.\d+)", line)
                if net_match:
                    current_round_locals.append(float(net_match.group(1)))
                    
                # Capture global server model test accuracy
                global_match = re.search(r">> Global Model Test accuracy: (\d+\.\d+)", line)
                if global_match:
                    global_accs.append(float(global_match.group(1)))

        # Append final round local calculations if log cuts off
        if current_round_locals:
            mean_local_accs.append(sum(current_round_locals) / len(current_round_locals))
            
    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found. Please check the path.")
        return

    # Handle case where arrays might be mismatched due to incomplete log chunks
    min_length = min(len(rounds), len(mean_local_accs), len(global_accs))
    rounds = rounds[:min_length]
    mean_local_accs = mean_local_accs[:min_length]
    global_accs = global_accs[:min_length]

    if min_length == 0:
        print("Error: No valid Federated Learning metrics could be parsed from the log file.")
        return

    # --- Step 2: Plotting and Formatting ---
    plt.figure(figsize=(7, 5), dpi=150)

    # Replicating style guidelines from target publication graphs
    plt.plot(rounds, mean_local_accs, label='FedSAP (Mean Local Clients)', color='#9467bd', linestyle='-', linewidth=2)
    plt.plot(rounds, global_accs, label='FedSAP (Global Model Server)', color='#e377c2', linestyle='--', linewidth=2)

    plt.xlabel('Communication round', fontsize=14)
    plt.ylabel('Test acc', fontsize=14)
    
    # Adaptive plot limits based on parsed content length
    plt.xlim(-0.5, max(rounds) + 0.5)
    plt.ylim(0.05, 0.25)  
    
    plt.xticks(rounds)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11, loc='upper right')
    plt.title('Parsed FedSAP Execution Tracking', fontsize=11, pad=10)

    # --- Step 3: Save and Close ---
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    print(f"Success! Graph generated and saved to: {output_image_path}")

# --- Execution Entry Point ---
if __name__ == "__main__":
    # Change 'federated_learning.log' to match your actual filename
    parse_and_plot_log('./logs/experiment_log-2026-06-09-06:55-44.log', 'fedsoap_plot_2.png')