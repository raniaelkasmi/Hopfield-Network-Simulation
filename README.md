# Hopfield Network Project

## Overview

This project implements a Hopfield network, a type of associative memory model proposed by John Hopfield in 1982. This network emulates how a neural network could store representations as weighted connections between neurons and retrieve these patterns from partial or noisy inputs. The Hopfield network used here is recurrent and binary, with each neuron able to be in one of two states: firing (1) or non-firing (-1).

The core functionality includes the construction of a Hopfield network, the Hebbian and Storkey learning rules for training, and synchronous and asynchronous retrieval dynamics.

## Project Requirements

- **Numpy** for efficient numerical operations.
- **Pandas** for storing experimental results and saving them in HDF5 format.
- **Matplotlib** for visualization and creation of animations.
- **Python** for implementation and execution.

Specifially, we are sharing an environment file (hopfield_environment.yml)
```conda env create -f hopfield_environment.yml```

You can activate the environment using :
```conda activate Hopfield```

Create a folder ```VideosPatternConverge``` where the videos of the evolution will be saved

## Network Definition

In this project:
- A Hopfield network with *N* neurons is represented as an *N*-dimensional binary vector in {-1, 1\}<sup>*N*</sup>.
- It is fully connected, with a symmetric weight matrix *W* generated using the Hebbian or Storkey learning rule. 
- Synaptic weights are symmetric without self-connections, so w<sub>*ij*</sub> = w<sub>*ji*</sub> and w<sub>*ii*</sub> = 0.

## Tasks Implemented

### Network Training

1. **Hebbian Learning Rule**: 
   - The weight matrix *W* is constructed based on the principle "Neurons that fire together, wire together."
   - Each weight w<sub>*ij*</sub> is calculated as the average of the contributions from each memorized pattern.

2. **Storkey Learning Rule**:
   - This rule provides an alternative method to construct the weight matrix, allowing for more patterns to be memorized without interference.

### Network Dynamics: Retrieving a Memorized Pattern

Using a partially corrupted input pattern, the network iteratively updates until it converges to one of the stored patterns. Two update mechanisms are used:
- **Synchronous Update**: All neurons are updated simultaneously.
- **Asynchronous Update**: Neurons are updated one at a time in a random sequence.

### Energy Function

The Hopfield network has an associated energy function, where patterns are local minima. This energy function decreases as the network iterates towards convergence.

### Visualization and Animation

The evolution of patterns over time is visualized using matplotlib to demonstrate how the network retrieves stored patterns. A video animation shows the gradual correction of the pattern.

## Model Analysis

You can find the detail of the experiments and the results obtained for the model analysis in the file [summary](./summary.md).

### Evaluating Network Storage and Retrieval Capacity

[PART 1: Capacity Analysis](./summary.md#part-1) in the Summary file.

1. **Experimental Setup**:
   - The experiments evaluated the Hopfield network's ability to store and retrieve patterns under noisy conditions.
   - Tests were conducted for various network sizes (10 to 2500 neurons) and compared two learning rules: Hebbian and Storkey.

2. **Empirical Capacity Analysis**:
   - The fraction of successfully retrieved patterns was analyzed as a function of the number of patterns stored in the network.
   - Results for different network sizes and learning rules were used to evaluate and compare their storage capacities.

3. **Visualization**:
   - **Capacity Curves**: Demonstrated the fraction of retrieved patterns as a function of the number of patterns stored for each network size. These curves highlight how retrieval success decreases with increased storage demands.
   - **Summary Plot**: Compared experimental storage capacities to theoretical predictions, showing the maximum retrievable patterns for each network size and learning rule.
  
### Evaluating Robustness to Perturbations

[PART 2: Robustness Analysis](./summary.md#part-2) in the Summary file.

1. **Experimental Setup**:
   - **The Network Sizes**: Experiments were conducted on network sizes ranging from 10 to 2500 neurons.
   - **Learning Rules**: Hebbian and Storkey rules were compared.
   - **Perturbations**: Initial perturbation starts at 20% noise, increasing in steps of 5%. The system's ability to retrieve the original pattern after perturbation was analyzed.
     
2. **Empirical Robustness Analysis**:
   - The fraction of successfully retrieved patterns was analyzed as a function of increasing perturbations.
   - Results for different network sizes and learning rules were used to evaluate and compare their robustness to noise. The point at which the system stops converging to the original pattern (in at least 9 out of 10 runs) was identified for each scenario.

3. **Visualization**:
   - **Robustness Tables**:Displayed the fraction of retrieved patterns for different perturbation levels across various network sizes and learning rules.
   - **Robustness Curves**: Plotted the fraction of retrieved patterns versus the number of pertubations for each network size and rules.

### Recalling Image Content from Corrupted Images

[PART 3: Recalling Image Content from Corrupted Images](./summary.md#part-3) in the Summary file.

In this task, the network demonstrates its ability to recall a complete image from a partially corrupted version.

## Code Structure

### Functions

- **Pattern Generation and Perturbation**:
  - `generate_patterns(num_patterns, pattern_size)`: Generates random binary patterns.
  - `perturb_pattern(pattern, num_perturb)`: Creates a noisy version of a stored pattern.
  
- **Learning Rules**:
  - `hebbian_weights(patterns)`: Creates the weight matrix using the Hebbian learning rule.
  - `storkey_weights(patterns)`: Creates the weight matrix using the Storkey learning rule.

- **Pattern Matching**:
  - `pattern_match(memorized_patterns, pattern)`: Checks if a pattern matches any stored pattern.

- **Update Dynamics**:
  - `update(state, weights)`: Applies synchronous update to the state.
  - `update_async(state, weights)`: Applies asynchronous update to the state.

- **Energy Calculation**:
  - `energy(state, weights)`: Computes the network's energy for a given state.

- **Visualization**:
  - `save_video(state_list, out_path)`: Creates a video showing the evolution of the network's state.

- **Capacity Experiments**:
  - `experiment(size, num_patterns, weight_rule, num_perturb, num_trials=10, max_iter=100)`: Runs a single experiment to test how well the network retrieves patterns under noise.
  - `run_capacity_experiments()`: Tests capacity for various network sizes and saves the results in an HDF5 file.
  - `plot_capacity_curves(results_df)`: Creates capacity curves (fraction of retrieved patterns vs. number of patterns) for different network sizes and saves the plot as `results/capacity_curves_subplots.png`.
  - `plot_summary_capacity(results_df)`: Summarizes the capacity (patterns retrievable with ≥90% probability) vs. network size and saves the plot as `results/summary_capacity.png`.

- **Robustness Experiments**:
   - `experiment_robustness(size, num_patterns, weight_rule, perturbations, num_trials=10, max_iter=100)`: Conducts an experiment to test the robustness of a Hopfield network by testing its ability to retrieve patterns after perturbation. It returns the fraction of successful retrievals for each perturbation level.
   - `run_robustness_experiments()`: Runs experiments to test the robustness of Hopfield networks for multiple network sizes and learning rules. It saves the results in a HDF5 file and outputs a Markdown table containing the results.
   - `plot_robustness_curves_by_size(results_df)`: Generates robustness curves for different network sizes (Hebbian and Storkey rules) and saves the plot as results/robustness_all_sizes.png

- **Image Binarization**:
  - `load_and_binarize_image(image_path)`: Binarize an image and convert pixel values to +1 and -1.


### Tests
We use [doctests](https://docs.python.org/3/library/doctest.html) and [pytest](https://docs.pytest.org/en/6.2.x/contents.html).

## Doctests

You can (doc)test the functions by:

```python functions.py```

It should have no output (if all tests pass).

You can also get a detailed output pass verbose flag:

```python functions.py -v```

## Pytests and Benchmark

We also created unit-tests and speed measurements with benchmark for the `functions.py` in `test_functions.py`. You can also run the whole test-suite with

```pytest```


You can get a detailed output with :

```pytest -v```

You will see if the tests passed and the speed measurements for some functions.

## Coverage

You can assess the coverage by running:

```
coverage run -m pytest
coverage report
```

## Measuring runtime of code and functions 

You can see the time taken by each function by running:

```
python profiling.py

```
You can also see the execution time of the main at the end of the output with the library timeit when you run:

```
python main.py

```

## How to use our code ?

You can run `main.py` with :
```python main.py```

This will allow you to test the functions with effective values that were given, helping you visualize more concretely what is happening.

For example, you can see that :
- The weight matrices are symmetric, diagonal elements of the Hebbian weight are zero, and weights lie in the range [-1, 1].
- Convergence occurs to the original pattern when noise is limited.
- The energy function decreases over time.
- The images of the video make sense by displaying the first three frames.
- Patterns are generated and perturbed with actual values.
- The checkerboard pattern is manually created for the visualization part.

To evaluate the capacity of a Hopfield network, i.e. how many patterns it can store and successfully retrieve whenn perturbed, you can run :
```
python capacity_of_hopfield_network.py
```

To test the robustness of a Hopfield network by increasing the perturbation on stored patterns and checking retrieval success, you can also run `robustness_of_hopfield_network.py` with :
```
python robustness_of_hopfield_network.py
```

Finally, you can see the retrieval process of corrupted images, including the evolution frames, by running:
```
python recall_image.py
```

  ### Results Directory

  All results, including data files and plots, will be saved in a directory named `results`, which is automatically created by the code if it does not already exist. There is no need for manual setup of this folder.

## Conclusion

This project explores the Hopfield network as an associative memory model, focusing on the Hebbian and Storkey learning rules. It demonstrates how the network can store and retrieve patterns through recurrent dynamics, with the stability of stored patterns achieved by energy minimization. The work highlights the capabilities of the Hopfield network in associative memory tasks, emphasizing the role of learning rules in determining the network's ability to retrieve patterns in noisy conditions.

