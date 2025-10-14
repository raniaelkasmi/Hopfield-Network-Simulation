import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

from functions import generate_patterns, perturb_pattern, pattern_match, dynamics, dynamics_async, hebbian_weights, update, update_async, storkey_weights, save_video, create_checkerboard_pattern, compute_energy

from timeit import default_timer as timer
from plots import*
from image_utils import*

def main():

    num_patterns = 80
    pattern_size = 1000
    num_perturb = 200
    max_sync_iter = 20
    max_async_iter = 20000
    convergence_criterion = 3000

    patterns = generate_patterns(num_patterns, pattern_size)
    print(patterns)
    print(patterns.shape)
    print("80 memorized patterns generated.")

    base_pattern = patterns[0]
    perturbed_pattern = perturb_pattern(base_pattern, num_perturb)
    print(f"\nPerturbed base pattern (200 elements modified):\n{perturbed_pattern}")
    print(perturbed_pattern.shape)

    # Test Hebbian rule
    print("\nTesting Hebbian rule.")
    hebb_weights = hebbian_weights(patterns)
    print(hebb_weights)
    print(hebb_weights.shape)
    print("\nWeight matrix calculated with the Hebbian rule.")

    # Synchronous dynamics
    print("\nSynchronous Hebbian dynamics:")
    state_history_sync = dynamics(perturbed_pattern, hebb_weights, max_sync_iter)
    final_state_sync = state_history_sync[-1]
    match_index_sync = pattern_match(patterns, final_state_sync)
    if np.array_equal(final_state_sync, base_pattern):
        print("The synchronous network recovered the original pattern with the Hebbian rule.")
    else:
        print("The synchronous network did not recover the original pattern with the Hebbian rule.")

    # Asynchronous dynamics
    print("\nAsynchronous Hebbian dynamics:")
    state_history_async = dynamics_async(perturbed_pattern, hebb_weights, max_async_iter, convergence_criterion)
    final_state_async = state_history_async[-1]
    match_index_async = pattern_match(patterns, final_state_async)
    if np.array_equal(final_state_async, base_pattern):
        print("The asynchronous network recovered the original pattern with the Hebbian rule.")
    else:
        print("The asynchronous network did not recover the original pattern with the Hebbian rule.")

    # Test Storkey rule
    print("\nTesting Storkey rule.")
    storkey_weights_m = storkey_weights(patterns)
    print(storkey_weights_m)
    print(storkey_weights_m.shape)
    print("\nWeight matrix calculated with the Storkey rule.")

    # Synchronous dynamics
    print("\nSynchronous Storkey dynamics:")
    state_history_sync_storkey = dynamics(perturbed_pattern, storkey_weights_m, max_sync_iter)
    final_state_sync_storkey = state_history_sync_storkey[-1]
    match_index_sync_storkey = pattern_match(patterns, final_state_sync_storkey)
    if np.array_equal(final_state_sync_storkey, base_pattern):
        print("The synchronous network recovered the original pattern with the Storkey rule.")
    else:
        print("The synchronous network did not recover the original pattern with the Storkey rule.")

    # Asynchronous dynamics  
    print("\nAsynchronous Storkey dynamics:")
    state_history_async_storkey = dynamics_async(perturbed_pattern, storkey_weights_m, max_async_iter, convergence_criterion)
    final_state_async_storkey = state_history_async_storkey[-1]
    match_index_async_storkey = pattern_match(patterns, final_state_async_storkey)
    if np.array_equal(final_state_async_storkey, base_pattern):
        print("The asynchronous network recovered the original pattern with the Storkey rule.")
    else:
        print("The asynchronous network did not recover the original pattern with the Storkey rule.")
        
    # Step 1: Generate 50 random patterns of size 2500
    num_patterns = 50
    pattern_size = 2500
    memorized_patterns = generate_patterns(num_patterns, pattern_size)

    # Step 2: Select one of the memorized patterns and perturb it by flipping 1000 elements
    pattern_to_perturb = memorized_patterns[0]  # Choosing the first pattern for perturbation
    num_perturb = 1000
    perturbed_pattern = perturb_pattern(pattern_to_perturb, num_perturb)

    # Check the original and perturbed pattern (for testing purposes)
    print("Original pattern (first 20 elements):", pattern_to_perturb[:20])
    print("Perturbed pattern (first 20 elements):", perturbed_pattern[:20])


    # Step 1: Store the patterns in the network using Hebbian and Storkey weights
    hebbian_weight_matrix = hebbian_weights(memorized_patterns)
    storkey_weight_matrix = storkey_weights(memorized_patterns)

    # Step 2: Define parameters
    max_iterations = 20  # Maximum number of iterations

    # Step 3: Use the perturbed pattern as the initial state for the dynamical evolution
    initial_state = perturbed_pattern.copy()

    # Step 4: Define functions to store state history with synchronous updates

    # Evolve with Hebbian weights
    print("Evolution with Hebbian weights:")
    hebbian_state_history = dynamics(initial_state, hebbian_weight_matrix, max_iterations)

    # Evolve with Storkey weights
    print("\nEvolution with Storkey weights:")
    storkey_state_history = dynamics(initial_state, storkey_weight_matrix, max_iterations)

    # Optional: Display results (e.g., states at each time step or whether convergence was achieved)
    print("\nFinal state after evolution with Hebbian weights (first 20 elements):", hebbian_state_history[-1][:20])
    print("Final state after evolution with Storkey weights (first 20 elements):", storkey_state_history[-1][:20])


    # Parameters for asynchronous evolution
    max_iterations = 30000          # Maximum iterations for the system to evolve
    convergence_num_iter = 10000     # Define convergence as no change for 10,000 iterations
    ""
    # Step 1: Run with Hebbian weights
    print("Asynchronous evolution with Hebbian weights:")
    hebbian_async_state_history = dynamics_async(initial_state, hebbian_weight_matrix, max_iterations, convergence_num_iter)

    # Step 2: Run with Storkey weights
    print("\nAsynchronous evolution with Storkey weights:")
    storkey_async_state_history = dynamics_async(initial_state, storkey_weight_matrix, max_iterations, convergence_num_iter)

    # Optional: Check final states
    print("\nFinal state after asynchronous evolution with Hebbian weights (first 20 elements):", hebbian_async_state_history[-1][:20])
    print("Final state after asynchronous evolution with Storkey weights (first 20 elements):", storkey_async_state_history[-1][:20])

    # Initialisation de la liste pour stocker l'énergie
    hebbian_energy_history = [compute_energy(initial_state, hebbian_weight_matrix)]
    storkey_energy_history = [compute_energy(initial_state, storkey_weight_matrix)]

    # Evolution avec les poids Hebbian
    print("Evolution avec les poids Hebbian :")
    hebbian_state_history = dynamics(initial_state, hebbian_weight_matrix, max_iterations)

    # Calcul de l'énergie après chaque itération pour l'état Hebbian
    for state in hebbian_state_history:
        hebbian_energy_history.append(compute_energy(state, hebbian_weight_matrix))

    # Evolution avec les poids Storkey
    print("\nEvolution avec les poids Storkey :")
    storkey_state_history = dynamics(initial_state, storkey_weight_matrix, max_iterations)

    # Calcul de l'énergie après chaque itération pour l'état Storkey
    for state in storkey_state_history:
        storkey_energy_history.append(compute_energy(state, storkey_weight_matrix))

    # Initialize list to store energy
    hebbian_energy_history = [compute_energy(initial_state, hebbian_weight_matrix)]
    storkey_energy_history = [compute_energy(initial_state, storkey_weight_matrix)]

    # Evolution with Hebbian weights
    print("Evolution with Hebbian weights:")
    hebbian_state_history = dynamics(initial_state, hebbian_weight_matrix, max_iterations)

    # Compute energy after each iteration for Hebbian state
    for state in hebbian_state_history:
        hebbian_energy_history.append(compute_energy(state, hebbian_weight_matrix))

    # Evolution with Storkey weights
    print("\nEvolution with Storkey weights:")
    storkey_state_history = dynamics(initial_state, storkey_weight_matrix, max_iterations)

    # Compute energy after each iteration for Storkey state
    for state in storkey_state_history:
        storkey_energy_history.append(compute_energy(state, storkey_weight_matrix))

    # Initialize list to store energy for asynchronous updates
    hebbian_async_energy_history = [compute_energy(initial_state, hebbian_weight_matrix)]
    storkey_async_energy_history = [compute_energy(initial_state, storkey_weight_matrix)]

    # Asynchronous evolution with Hebbian weights
    print("Asynchronous evolution with Hebbian weights:")
    hebbian_async_state_history = dynamics_async(initial_state, hebbian_weight_matrix, max_iterations, convergence_num_iter)

    # Compute energy after each iteration for Hebbian asynchronous state, with progress prints
    for i, state in enumerate(hebbian_async_state_history):
        if i % 100 == 0:  # Record only every 100th state
            current_energy = compute_energy(state, hebbian_weight_matrix)
            hebbian_async_energy_history.append(current_energy)

    # Asynchronous evolution with Storkey weights
    print("\nAsynchronous evolution with Storkey weights:")
    storkey_async_state_history = dynamics_async(initial_state, storkey_weight_matrix, max_iterations, convergence_num_iter)

    # Compute energy after each iteration for Storkey asynchronous state, with progress prints
    for i, state in enumerate(storkey_async_state_history):
        if i % 100 == 0:  # Record every 100th state
            current_energy = compute_energy(state, storkey_weight_matrix)
            storkey_async_energy_history.append(current_energy)

    # Plot energy evolution for synchronous updates
    plt.plot(hebbian_energy_history, label="Hebbian (synchronous)")
    plt.plot(storkey_energy_history, label="Storkey (synchronous)")

    # Plot energy evolution for asynchronous updates
    plt.plot(hebbian_async_energy_history, label="Hebbian (asynchronous)")
    plt.plot(storkey_async_energy_history, label="Storkey (asynchronous)")

    # Add labels and title
    plt.xlabel('Iterations')
    plt.ylabel('Energy')
    plt.title('Energy evolution during dynamics')
    plt.legend()
    plt.show()

    # Create checkerboard
    num_patterns = 80
    pattern_size = 2500 
    patterns = generate_patterns(num_patterns, pattern_size)
    checkerboard_pattern = create_checkerboard_pattern()
    patterns[0] = checkerboard_pattern
    hebb_weights_checkboard = hebbian_weights(patterns)
    perturbed_checkerboard = perturb_pattern(checkerboard_pattern, 1000)
    state_history_sync = dynamics(perturbed_checkerboard, hebb_weights_checkboard, max_sync_iter)
    state_history_async = dynamics_async(perturbed_checkerboard, hebb_weights_checkboard, max_async_iter, convergence_criterion) 
    state_history_sync = dynamics(perturbed_checkerboard, hebb_weights_checkboard, max_sync_iter)
    state_history_async = dynamics_async(perturbed_checkerboard, hebb_weights_checkboard, max_async_iter, convergence_criterion) 
    state_history_async_sampled = state_history_async[::1000]
    reshaped_sync = [state.reshape((50, 50)) for state in state_history_sync]
    reshaped_async = [state.reshape((50, 50)) for state in state_history_async_sampled]



    # Display first 3 frames synchronous
    for i in range(3):  
        plt.imshow(reshaped_sync[i], cmap='gray')
        plt.title(f"Synchronous state {i}")
        plt.show()

    # Display first 3 frames asynchronous
    for i in range(3):  
        plt.imshow(reshaped_async[i], cmap='gray')
        plt.title(f"Asynchronous state {i}")
        plt.show()
        
    save_video(reshaped_sync, 'VideosPatternConverge/checkerboard_evolution_sync.mp4')
    save_video(reshaped_async, 'VideosPatternConverge/checkerboard_evolution_async.mp4')


if __name__ == "__main__":
    start = timer()
    main()
    end = timer()
    duration = end - start
    print(duration)
