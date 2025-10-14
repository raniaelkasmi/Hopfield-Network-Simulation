from functions import*
import os
import pytest
from moviepy.editor import VideoFileClip
import pandas as pd
from unittest.mock import patch
from capacity_of_hopfield_network import (
    experiment,
    run_capacity_experiments,
    plot_capacity_curves,
    plot_summary_capacity,
)


def test_hebbian() :
    patterns = np.array([[1, 1, -1, -1],
                         [1, 1, -1, 1],
                         [-1, 1, -1, 1]])
    
    W = hebbian_weights(patterns)
    assert W.shape == (4,4)
    expected_W = np.array ([[0.         , 0.33333333, -0.33333333, -0.33333333],
                           [0.33333333 , 0.        , -1.        ,  0.33333333],
                           [-0.33333333, -1        , 0          , -0.33333333],
                           [-0.33333333, 0.33333333, -0.33333333, 0.         ]])
    assert np.allclose(W, expected_W)

def test_storkey() :
    patterns = np.array([[1, 1, -1, -1],
                         [1, 1, -1, 1],
                         [-1, 1, -1, 1]])
    
    S = storkey_weights(patterns)
    expected_S = np.array ([[1.125, 0.25 , -0.25, -0.5 ],
                            [0.25  , 0.625, -1.  , 0.25 ],
                            [-0.25 , -1   , 0.625, -0.25],
                            [-0.5  , 0.25 , -0.25, 1.125]])
    assert np.allclose(S, expected_S)

def test_energy() :
    state = np.array([1, -1, 1])
    weights = np.array([[0, 1, -1], [1, 0, 1], [-1, 1, 0]])
    E = compute_energy(state, weights)
    assert E == 3

def test_activation() :
    # Test with a mix of positive, zero, and negative values
    assert (activation(np.array([0.5, -0.2, 0, -1, 3])) == np.array([1, -1, 1, -1, 1])).all()
    assert (activation(np.array([-1.5, 2.3, -0.1, 0])) == np.array([-1, 1, -1, 1])).all()
    # Test with all positive values
    assert (activation(np.array([1, 2, 3])) == np.array([1, 1, 1])).all()
     # Test with all negative values
    assert (activation(np.array([-1, -2, -3])) == np.array([-1, -1, -1])).all()


def test_generate_patterns() :
    patterns = generate_patterns(3, 4)
    
    # Check the shape of the generated patterns
    assert patterns.shape == (3, 4), "Shape of generated patterns is incorrect."
    
    # Check that all values are either 1 or -1
    assert np.all((patterns == 1) | (patterns == -1)), "Generated patterns contain values other than 1 or -1."


def test_perturb_pattern():
    pattern = np.array([1, 1, -1, -1, 1])
    num_perturb = 2
    perturbed = perturb_pattern(pattern, num_perturb)
    
    # Check that the length of the perturbed pattern is the same as the original
    assert len(perturbed) == len(pattern), "Length of perturbed pattern should match original pattern."
    
    # Check that exactly `num_perturb` elements have been flipped
    assert np.sum(pattern != perturbed) == num_perturb, f"Exactly {num_perturb} elements should be flipped."
    
    # Check that all elements in the perturbed pattern are either 1 or -1
    assert set(perturbed) <= {1, -1}, "All elements in the perturbed pattern should be either 1 or -1."


def test_pattern_match():
    memorized_patterns = np.array([[1, -1, 1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]])

    # Check that the function returns the correct index for a matching pattern
    assert pattern_match(memorized_patterns, np.array([1, -1, 1, -1])) == 0, "Pattern should match at index 0"
    assert pattern_match(memorized_patterns, np.array([1, 1, -1, 1])) == 1, "Pattern should match at index 1"
    
    # Check that the function returns None for a non-matching pattern
    assert pattern_match(memorized_patterns, np.array([1, 1, 1, 1])) is None, "Pattern should not match any memorized pattern"


def test_update():
    patterns = np.array([[1, 1, -1, -1],
                          [1, 1, -1, 1],
                          [-1, 1, -1, 1]])
    weights = hebbian_weights(patterns)
    state = np.array([1, -1, 1, -1])
    
    expected_output = np.array([-1, -1, 1, -1]) #Expected array with the correct calculation
    result = update(state, weights)
    
    assert np.array_equal(result, expected_output), f"Expected {expected_output}, got {result}"



def test_update_async():
    patterns = np.array([[1, 1, -1, -1],
                          [1, 1, -1, 1],
                          [-1, 1, -1, 1]])
    weights = hebbian_weights(patterns)
    state = np.array([1, -1, 1, -1])
    
    new_state = update_async(state, weights)
    
    # Check that the shape of the new state is the same as the original state
    assert new_state.shape == state.shape, "The shape of the updated state should match the original state."

    # Check that maximum one element has changed in the new state
    assert np.sum(new_state != state) in [0,1], "Only one element should change in the asynchronous update."


def test_dynamics():
    patterns = np.array([[1, 1, -1, -1],
                         [1, 1, -1, 1],
                         [-1, 1, -1, 1]])
    weights = hebbian_weights(patterns)
    state = np.array([1, -1, 1, -1])
    
    # Case 1: Convergence within max_iter
    history_converged = dynamics(state, weights, max_iter=10)
    
    # Check that history contains at least two states (initial + at least one update)
    assert len(history_converged) >= 2, "History should contain at least the initial state and one update"
    
    # Verify convergence by checking that the last two states are the same
    assert np.array_equal(history_converged[-1], history_converged[-2]), \
        "The last two states should be the same, indicating convergence"

    # Case 2: No convergence within max_iter
    history_no_convergence = dynamics(state, weights, max_iter=1)  # Set max_iter low to prevent convergence
    
    # Check that history contains at least two states
    assert len(history_no_convergence) >= 2, "History should contain at least the initial state and one update"
    
    # Verify non-convergence by checking that the last two states are different
    assert not np.array_equal(history_no_convergence[-1], history_no_convergence[-2]), \
        "The last two states should differ, indicating no convergence"


def test_dynamics_async_convergence(capfd):
    patterns = np.array([[1, 1, -1, -1],
                         [1, 1, -1, 1],
                         [-1, 1, -1, 1]])
    weights = hebbian_weights(patterns)
    state = np.array([1, -1, 1, -1])

    # Test de convergence
    history = dynamics_async(state, weights, max_iter=100, convergence_num_iter=5)
    
    # Vérifiez la convergence
    assert all(np.array_equal(history[-i-1], history[-i-2]) for i in range(5)), \
        "The last 5 states should be the same, indicating convergence"
    
    captured = capfd.readouterr()
    assert "Convergence achieved." in captured.out, "Expected convergence message was not printed."


def test_dynamics_async_no_convergence(capfd):
    patterns = np.array([[1, 1, -1, -1],
                         [1, 1, -1, 1],
                         [-1, 1, -1, 1]])
    weights = hebbian_weights(patterns)
    state = np.array([1, -1, 1, -1])

    # Exécuter avec des paramètres garantissant la non-convergence
    history_no_convergence = dynamics_async(state, weights, max_iter=5, convergence_num_iter=5)
    
    # Define no convergence
    no_convergence = not all(np.array_equal(history_no_convergence[-i-1], history_no_convergence[-i-2]) for i in range(5))

    # Capture de la sortie
    captured = capfd.readouterr()
    if no_convergence :
        assert "The network did not converge within the maximum number of iterations." in captured.out, \
            "Expected non-convergence message was not printed when last two states differ."


def test_system_convergence_hebbian():
    num_patterns = 1
    pattern_size = 100
    patterns = generate_patterns(num_patterns, pattern_size)
    perturbed_pattern = patterns[0].copy()
    perturbed_pattern[0:10] = -perturbed_pattern[0:10]  

    weights = hebbian_weights(patterns)
    state_history = dynamics(perturbed_pattern, weights, max_iter=100)
    
    final_state = state_history[-1]
    assert np.array_equal(final_state, patterns[0])

def test_system_convergence_storkey():
    num_patterns = 1
    pattern_size = 100
    patterns = generate_patterns(num_patterns, pattern_size)
    perturbed_pattern = patterns[0].copy()
    perturbed_pattern[0:10] = -perturbed_pattern[0:10]  

    weights = storkey_weights(patterns)
    state_history = dynamics(perturbed_pattern, weights, max_iter=100)
    
    final_state = state_history[-1]
    assert np.array_equal(final_state, patterns[0])


def test_energy_non_increasing_hebbian():
    patterns = generate_patterns(5, 50)
    weights = hebbian_weights(patterns)
    
    initial_state = patterns[0].copy()
    state_history = dynamics(initial_state, weights, max_iter=50)
    
    energies = [compute_energy(state, weights) for state in state_history]
    assert all(x >= y for x, y in zip(energies, energies[1:]))

def test_energy_non_increasing_storkey():
    patterns = generate_patterns(5, 50)
    weights = storkey_weights(patterns)  # Utilise les poids de Storkey
    
    initial_state = patterns[0].copy()
    state_history = dynamics(initial_state, weights, max_iter=50)
    
    energies = [compute_energy(state, weights) for state in state_history]
    assert all(x >= y for x, y in zip(energies, energies[1:]))

@pytest.fixture
# To avoid repetition for the tests of save videos parameters
def state_list():
    return [np.ones((50, 50)), -np.ones((50, 50)), np.ones((50, 50)), -np.ones((50, 50))]

@pytest.fixture
def output_path():
    return 'test_output.mp4'
    
def test_save_video_number_of_frames(state_list, output_path):

    save_video(state_list, output_path)

    with VideoFileClip(output_path) as clip:
        assert clip.duration > 0
        num_frames = round(clip.fps * clip.duration)
        assert num_frames == len(state_list), f"Nombre de frames attendu : {len(state_list)}, obtenu : {num_frames}"
    os.remove(output_path)

def test_save_video_file_existance(state_list, output_path):

    save_video(state_list, output_path)
    assert os.path.exists(output_path) == True
    os.remove(output_path)

@pytest.fixture
def checkerboard():
    return create_checkerboard_pattern()

def test_checkerboard_correct_shape(checkerboard):
    assert checkerboard.shape == (2500,)

def test_checkerboard_correct_values(checkerboard):
    assert np.sum(checkerboard == 1) == 1250
    assert np.sum(checkerboard == -1) == 1250

def test_checkerboard_initial_pattern(checkerboard):
    expected_pattern_start = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
    assert np.array_equal(checkerboard[:10], expected_pattern_start)

#BENCHMARK
NUM_PATTERNS = 50
NETWORK_SIZE = 2500
NUM_PERTURBATIONS = 1000
MAX_ITER_SYNC = 20
MAX_ITER_ASYNC = 30000

# Generate random patterns for the network
patterns = np.random.choice([-1, 1], size=(NUM_PATTERNS, NETWORK_SIZE))

# Perturbed pattern for testing
perturbed_pattern = patterns[0].copy()
perturbation_indices = np.random.choice(NETWORK_SIZE, NUM_PERTURBATIONS, replace=False)
perturbed_pattern[perturbation_indices] *= -1

# Benchmarking tests
def test_hebbian_weights_benchmark(benchmark):
    weights = benchmark.pedantic(hebbian_weights, args=(patterns,), rounds=5, iterations=1)
    print(weights)
    assert weights.shape == (NETWORK_SIZE, NETWORK_SIZE)

def test_storkey_weights_benchmark(benchmark):
    weights = benchmark.pedantic(storkey_weights, args=(patterns,), rounds=5, iterations=1)
    assert weights.shape == (NETWORK_SIZE, NETWORK_SIZE)

def test_update_benchmark(benchmark):
    weights = hebbian_weights(patterns)

    def run_synchronous_updates(state, weights, max_iter):
        for _ in range(max_iter):
            state = update(state, weights)
        return state

    result = benchmark.pedantic(
        run_synchronous_updates,
        args=(perturbed_pattern, weights, MAX_ITER_SYNC),
        rounds=5,
        iterations=1
    )
    assert len(result) == NETWORK_SIZE

def test_update_async_benchmark(benchmark):
    weights = hebbian_weights(patterns)

    def run_asynchronous_updates(state, weights, max_iter):
        for _ in range(max_iter):
            state = update_async(state, weights)
        return state

    result = benchmark.pedantic(
        run_asynchronous_updates,
        args=(perturbed_pattern, weights, MAX_ITER_ASYNC),
        rounds=5,
        iterations=1
    )
    assert len(result) == NETWORK_SIZE

def test_energy_function_benchmark(benchmark):
    weights = hebbian_weights(patterns)
    energy_value = benchmark.pedantic(compute_energy, args=(perturbed_pattern, weights), rounds=5, iterations=1)
    assert isinstance(energy_value, float)

@pytest.fixture
def temp_results_dir(tmpdir, monkeypatch):
    """Fixture pour créer un répertoire temporaire et rediriger RESULTS_DIR."""
    temp_dir = tmpdir.mkdir("results")
    monkeypatch.setattr("capacity_of_hopfield_network.RESULTS_DIR", str(temp_dir))
    return str(temp_dir)

# Test rapide pour experiment
def test_experiment():
    """Test rapide de la fonction experiment."""
    result = experiment(size=2, num_patterns=1, weight_rule="hebbian", num_perturb=0, num_trials=1, max_iter=1)
    assert isinstance(result, dict)
    assert "match_frac" in result
    assert 0 <= result["match_frac"] <= 1

# Test rapide pour run_capacity_experiments
def test_run_capacity_experiments(temp_results_dir):
    """Test de run_capacity_experiments avec une version allégée."""
    with patch("capacity_of_hopfield_network.experiment") as mock_experiment:
        # Mock des résultats de experiment pour accélérer le test
        mock_experiment.return_value = {
            "network_size": 2,
            "weight_rule": "hebbian",
            "num_patterns": 1,
            "num_perturb": 0,
            "match_frac": 0.9,
        }
        results_df = run_capacity_experiments()

    # Vérifications
    assert isinstance(results_df, pd.DataFrame)
    assert not results_df.empty
    assert "network_size" in results_df.columns
    assert "match_frac" in results_df.columns

    # Vérification du fichier HDF5
    hdf5_path = os.path.join(temp_results_dir, "hopfield_capacity.h5")
    assert os.path.exists(hdf5_path)

# Test rapide pour plot_capacity_curves
def test_plot_capacity_curves(temp_results_dir):
    """Test rapide de plot_capacity_curves."""
    results_df = pd.DataFrame({
        "network_size": [2],
        "weight_rule": ["hebbian"],
        "num_patterns": [1],
        "match_frac": [0.9],
    })
    with patch("matplotlib.pyplot.savefig"):  # Mock plt.savefig pour éviter l'enregistrement
        plot_capacity_curves(results_df)

# Test rapide pour plot_summary_capacity
def test_plot_summary_capacity(temp_results_dir):
    """Test rapide de plot_summary_capacity."""
    results_df = pd.DataFrame({
        "network_size": [2],
        "weight_rule": ["hebbian"],
        "num_patterns": [1],
        "match_frac": [0.9],
    })
    with patch("matplotlib.pyplot.savefig"):  # Mock plt.savefig pour accélérer
        summary_df = plot_summary_capacity(results_df)

    assert isinstance(summary_df, pd.DataFrame)
    assert not summary_df.empty
