import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import generate_patterns, perturb_pattern, hebbian_weights, storkey_weights, dynamics

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def experiment(size, num_patterns, weight_rule, num_perturb, num_trials=10, max_iter=100):
    """
    Conducts an experiment for a Hopfield network with specified size and perturbation parameters.

    Args:
        size (int): Number of neurons in the network.
        num_patterns (int): Number of patterns to store in the network.
        weight_rule (str): Learning rule to use for weight matrix ("hebbian" or "storkey").
        num_perturb (int): Number of bits to flip in the input pattern to create a corrupted pattern.
        num_trials (int, optional): Number of trials to perform for the experiment. Default is 10.
        max_iter (int, optional): Maximum number of iterations for the network dynamics. Default is 100.

    Returns:
        dict: A dictionary containing:
            - "network_size" (int): Size of the network.
            - "weight_rule" (str): Learning rule used.
            - "num_patterns" (int): Number of patterns stored.
            - "num_perturb" (int): Number of bits perturbed in each trial.
            - "match_frac" (float): Fraction of trials where the network successfully retrieved the original pattern.
    """
    results = {
        "network_size": size,
        "weight_rule": weight_rule,
        "num_patterns": num_patterns,
        "num_perturb": num_perturb,
        "match_frac": 0.0
    }
    matches = 0

    patterns = generate_patterns(num_patterns, size)
    weights = hebbian_weights(patterns) if weight_rule == "hebbian" else storkey_weights(patterns)

    for _ in range(num_trials):
        base_pattern = patterns[np.random.randint(num_patterns)]
        perturbed_pattern = perturb_pattern(base_pattern, num_perturb)

        history = dynamics(perturbed_pattern, weights, max_iter)
        final_state = history[-1]
        if np.array_equal(final_state, base_pattern):
            matches += 1

    results["match_frac"] = matches / num_trials
    return results

def run_capacity_experiments():
    """
    Runs experiments to test the capacity of Hopfield networks.

    This function tests the capacity of Hopfield networks by evaluating the number of patterns 
    that can be reliably stored and recalled as a function of the network size and weight update 
    rules (Hebbian and Storkey). For each combination of network size and weight rule, the function 
    computes the critical capacity and runs multiple experiments for different numbers of patterns 
    based on that capacity.

    The results of the experiments are stored in a Pandas DataFrame, which is then saved as an HDF5 
    file to the specified directory.

    Returns:
        pandas.DataFrame: A DataFrame containing the results of all experiments with columns 
                          related to the network size, number of patterns, weight rule, 
                          and other relevant metrics.

    Saves:
        - The results are saved as an HDF5 file at `RESULTS_DIR` with the filename 
          "hopfield_capacity.h5".
    """
    sizes = [10, 18, 34, 63, 116, 215, 397, 733, 1354, 2500]
    weight_rules = ["hebbian", "storkey"]
    results = []

    for size in sizes:
        for weight_rule in weight_rules:
            C = size / (2 * np.log(size)) if weight_rule == "hebbian" else size / np.sqrt(2 * np.log(size))
            t_values = np.linspace(0.5 * C, 2 * C, 10).astype(int)

            for num_patterns in t_values:
                result = experiment(size, num_patterns, weight_rule, num_perturb=int(0.2 * size))
                results.append(result)

    results_df = pd.DataFrame(results)
    results_hdf5_path = os.path.join(RESULTS_DIR, "hopfield_capacity.h5")
    results_df.to_hdf(results_hdf5_path, key="df", mode="w")
    print(f"Résultats sauvegardés dans {results_hdf5_path}")
    return results_df

def plot_capacity_curves(results_df):
    """
    Generates and saves capacity curves organized into subplots by network size.

    This function creates capacity curves that show the fraction of retrieved patterns as a 
    function of the number of patterns for different network sizes. The results are grouped by 
    weight update rules (Hebbian and Storkey), and each subplot represents a different network 
    size. The curves are plotted for each weight rule, showing how the network's capacity changes 
    as the number of patterns varies.

    The resulting figure is saved as a PNG image with subplots for each network size.

    Args:
        results_df (pandas.DataFrame): A DataFrame containing the results of the Hopfield 
                                        network experiments, including columns for network size, 
                                        number of patterns, weight rule, and the fraction of 
                                        retrieved patterns.

    Saves:
        - The capacity curves are saved as a PNG file at `RESULTS_DIR` with the filename 
          "capacity_curves_subplots.png".
    """
    sizes = results_df["network_size"].unique()
    rows, cols = (len(sizes) + 1) // 2, 2

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 5))
    axes = axes.flatten()

    for i, size in enumerate(sizes):
        ax = axes[i]
        subset = results_df[results_df["network_size"] == size]
        for weight_rule in subset["weight_rule"].unique():
            rule_data = subset[subset["weight_rule"] == weight_rule]
            ax.plot(rule_data["num_patterns"], rule_data["match_frac"], label=weight_rule.capitalize())
        ax.set_title(f"Network Size = {size}")
        ax.set_xlabel("Number of patterns")
        ax.set_ylabel("Fraction of retrieved patterns")
        ax.legend()
        ax.grid(True)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "capacity_curves_subplots.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Capacity curves (subplots) saved as {plot_path}")
    plt.close()

def plot_summary_capacity(results_df):
    """
    Generates and saves a summary of capacity for the Hebbian and Storkey rules.

    This function calculates the capacity of Hopfield networks for different network sizes, 
    based on the number of patterns that can be reliably retrieved with at least 90% success 
    for each weight update rule (Hebbian and Storkey). The capacities are plotted as a function 
    of network size, with both the empirical results and theoretical capacity curves for each rule.

    The resulting summary plot is saved as a PNG image.

    Args:
        results_df (pandas.DataFrame): A DataFrame containing the results of the Hopfield 
                                        network experiments, including columns for network size, 
                                        number of patterns, weight rule, and the fraction of 
                                        retrieved patterns.

    Returns:
        pandas.DataFrame: A DataFrame summarizing the capacity for each network size and weight 
                          rule, including the maximum number of patterns that can be reliably 
                          retrieved with at least 90% success.

    Saves:
        - The summary capacity plot is saved as a PNG file at `RESULTS_DIR` with the filename 
          "summary_capacity.png".
    """
    summary_data = []
    for size in results_df["network_size"].unique():
        for rule in ["hebbian", "storkey"]:
            subset = results_df[(results_df["network_size"] == size) & (results_df["weight_rule"] == rule)]
            capacity = subset[subset["match_frac"] >= 0.9]["num_patterns"].max()
            summary_data.append({"network_size": size, "weight_rule": rule, "capacity": capacity})
    summary_df = pd.DataFrame(summary_data)

    sizes = summary_df["network_size"]
    plt.figure(figsize=(10, 6))
    for rule in ["hebbian", "storkey"]:
        subset = summary_df[summary_df["weight_rule"] == rule]
        plt.plot(subset["network_size"], subset["capacity"], marker='o', label=f"{rule.capitalize()} (empirical)")

    plt.plot(sizes, sizes / (2 * np.log(sizes)), 'k--', label="Hebbian (theoretical)")
    plt.plot(sizes, sizes / np.sqrt(2 * np.log(sizes)), 'k:', label="Storkey (theoretical)")
    plt.xlabel("Network size")
    plt.ylabel("Capacity (# patterns with ≥ 90% success)")
    plt.legend()
    plt.title("Summary Capacity of Hopfield Networks")
    plt.grid()
    plot_path = os.path.join(RESULTS_DIR, "summary_capacity.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Summary plot saved as {plot_path}")
    plt.close()
    return summary_df

if __name__ == "__main__":
    results_df = run_capacity_experiments()
    plot_capacity_curves(results_df)
    summary_df = plot_summary_capacity(results_df)