import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import *

RESULTS_DIR = "results"

# --------------------------------------
# PART 2 : ROBUSTNESS
# --------------------------------------

def experiment_robustness(size, num_patterns, weight_rule, perturbations, num_trials=10, max_iter=100):
    """
    Conducts an experiment to test the robustness of a Hopfield network by evaluating its performance
    with various levels of perturbation on the patterns stored in the network.

    Args:
        size (int): Size of the network (number of neurons).
        num_patterns (int): Number of patterns stored in the network.
        weight_rule (str): The learning rule to be used ("hebbian" or "storkey").
        perturbations (list): List of perturbation percentages to test (e.g., [0.2, 0.3, 0.4]).
        num_trials (int): The number of trials to run for each perturbation level.
        max_iter (int): The maximum number of iterations allowed during the network dynamics.

    Returns:
        list: A list of dictionaries, each containing the experiment results for a given configuration.
    """
    results = []

    # Generate patterns and compute weights
    patterns = generate_patterns(num_patterns, size)
    weights = hebbian_weights(patterns) if weight_rule == "hebbian" else storkey_weights(patterns)

    for perturb_percent in perturbations:
        matches = 0
        num_perturb = int(perturb_percent * size)  # Convert to number of perturbed bits

        for _ in range(num_trials):
            base_pattern = patterns[np.random.randint(num_patterns)]
            perturbed_pattern = perturb_pattern(base_pattern, num_perturb)

            # Apply dynamics
            history = dynamics(perturbed_pattern, weights, max_iter)
            final_state = history[-1]
            if np.array_equal(final_state, base_pattern):
                matches += 1

        # Append results
        results.append({
            "network_size": size,
            "weight_rule": weight_rule,
            "num_patterns": num_patterns,
            "num_perturb": num_perturb,
            "match_frac": matches / num_trials
        })

    return results


def run_robustness_experiments():
    """
    Runs robustness experiments for various network sizes and learning rules (Hebbian and Storkey).
    The function tests the robustness of the network by progressively increasing the perturbation levels
    and storing the results in a DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the results of the robustness experiments.
    """
    sizes = [10, 18, 34, 63, 116, 215, 397, 733, 1354, 2500]
    weight_rules = ["hebbian", "storkey"]
    perturbations = np.arange(0.2, 1.01, 0.05)
    results = []

    for size in sizes:
        for weight_rule in weight_rules:
            # Set t = 2
            num_patterns = 2
            # Run robustness experiments
            results.extend(experiment_robustness(size, num_patterns, weight_rule, perturbations))

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)   
    results_hdf5_path = os.path.join(RESULTS_DIR, "hopfield_robustness.h5")
    results_df.to_hdf(results_hdf5_path, key="df", mode="w")
    print(f"Results saved to {results_hdf5_path}")

    # Convert in Markdown format
    markdown_table = results_df.to_markdown()

    # Define the path to save the file in the 'results' directory
    results_file_path = os.path.join(RESULTS_DIR, 'output_table.md')

    # Save the Markdown table to a file in the 'results' directory
    with open(results_file_path, 'w') as f:
        f.write(markdown_table)

    print(f"Table saved to '{results_file_path}'.")
    
    return results_df



def plot_robustness_curves_by_size(results_df):
    """
    Generates a plot with multiple subplots, each showing the robustness curve (fraction of retrieved patterns)
    for a specific network size, comparing the performance of Hebbian and Storkey learning rules under different
    perturbation levels.

    Args:
        results_df (pd.DataFrame): A DataFrame containing the results of robustness experiments, including
                                    network size, weight rule, perturbation levels, and fraction of retrieved patterns.
    """
    # Get unique network sizes and calculate the grid size for subplots
    sizes = results_df["network_size"].unique()
    num_sizes = len(sizes)
    num_cols = 4  # Number of columns in the grid
    num_rows = (num_sizes + num_cols - 1) // num_cols  # Calculate rows dynamically

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 10))
    axes = axes.flatten()  # Flatten axes array for easier indexing

    # Iterate over each network size and plot
    for i, size in enumerate(sizes):
        ax = axes[i]
        subset = results_df[results_df["network_size"] == size]

        # Plot Hebbian and Storkey curves
        for weight_rule in ["hebbian", "storkey"]:
            rule_data = subset[subset["weight_rule"] == weight_rule]
            ax.plot(
                rule_data["num_perturb"],
                rule_data["match_frac"],
                label=f"{weight_rule.capitalize()}",
                linewidth=2,
            )

        # Customize the subplot
        ax.set_title(f"Size={size}", fontsize=12)
        ax.set_xlabel("Number of perturbed bits", fontsize=10)
        ax.set_ylabel("Fraction retrieved", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(fontsize=8)

    # Hide unused subplots if there are empty slots in the grid
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()

    # Save the combined plot to the results directory
    combined_plot_path = os.path.join(RESULTS_DIR, "robustness_all_sizes.png")
    plt.savefig(combined_plot_path, dpi=300)
    print(f"Combined robustness plot saved at {combined_plot_path}")
    plt.close()




if __name__ == "__main__":
    # Run robustness experiments
    robustness_df = run_robustness_experiments()

    # Generate and save separate robustness plots by size
    plot_robustness_curves_by_size(robustness_df)
