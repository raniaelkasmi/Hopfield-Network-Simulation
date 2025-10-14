import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import ArtistAnimation
np.set_printoptions(legacy='1.13')


def activation(x):
    """
    Activation function for a Hopfield network.

    This function applies a step function to the input array `x`, setting 
    each element to 1 if it is non-negative and to -1 otherwise. This 
    activation function helps update the state of neurons in the network.

    Parameters
    ----------
    x : ndarray
        Input array representing the state of the neurons in the network.

    Returns
    -------
    ndarray
        An array where each element is 1 if the corresponding input was 
        non-negative, and -1 otherwise.

    Examples
    --------
    >>> activation(np.array([0.5, -0.2, 0, -1, 3]))
    array([ 1, -1,  1, -1,  1])

    >>> activation(np.array([-1.5, 2.3, -0.1, 0]))
    array([-1,  1, -1,  1])

    >>> activation(np.array([1, 2, 3]))
    array([1, 1, 1])

    >>> activation(np.array([-1, -2, -3]))
    array([-1, -1, -1])
     
    """
    return np.where(x>=0, 1, -1)

def generate_patterns(num_patterns, pattern_size) :
    """
    Generate random patterns for a Hopfield network.

    This function generates `num_patterns` binary patterns, each of size 
    `pattern_size`, with elements randomly chosen between 1 and -1.

    Parameters
    ----------
    num_patterns : int
        The number of random patterns to generate.
    pattern_size : int
        The size of each pattern (i.e., the number of elements in each pattern).

    Returns
    -------
    ndarray
        An array of shape (num_patterns, pattern_size) where each row is a 
        random binary pattern.

    Examples
    --------
    >>> patterns = generate_patterns(3, 4)
    >>> patterns.shape == (3, 4)  # Check the shape of the result
    True
    >>> np.all((patterns == 1) | (patterns == -1)) # Check all values are either 1 or -1
    True

    """
    return np.random.choice([1, -1], size=(num_patterns, pattern_size))


def perturb_pattern(pattern, num_perturb):
    """
    Perturb a pattern by flipping a specified number of its elements.

    This function creates a copy of the input `pattern` and flips the value 
    of `num_perturb` randomly selected elements (from 1 to -1 or vice versa).

    Parameters
    ----------
    pattern : ndarray
        The original pattern to be perturbed. It should be a 1-D array.
    num_perturb : int
        The number of elements to flip in the pattern.

    Returns
    -------
    ndarray
        A copy of the input pattern with `num_perturb` elements flipped.

    Examples
    --------
    >>> pattern = np.array([1, 1, -1, -1, 1])
    >>> perturbed = perturb_pattern(pattern, 2)
    >>> len(perturbed) == len(pattern)  # Length remains the same
    True
    >>> np.sum(pattern != perturbed) == 2  # Exactly 2 elements flipped
    True
    >>> set(perturbed) <= {1, -1}  # Elements are either 1 or -1
    True

    """
    pertubed_pattern = pattern.copy()
    indices_to_flip = np.random.choice(len(pattern), size=num_perturb, replace=False)
    pertubed_pattern[indices_to_flip] *= -1
    return pertubed_pattern


def pattern_match(memorized_patterns, pattern) :
    """
    Check if a given pattern matches any of the memorized patterns.

    This function compares the input `pattern` to each pattern in the 
    `memorized_patterns` array and returns the index of the matching pattern 
    if found, or `None` if no match is found.

    Parameters
    ----------
    memorized_patterns : ndarray
        A 2-D array where each row is a memorized pattern.
    pattern : ndarray
        The pattern to be matched against the memorized patterns.

    Returns
    -------
    int or None
        The index of the matching pattern if found; otherwise, None.

    Examples
    --------
    >>> memorized_patterns = np.array([[1, -1, 1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]])
    >>> pattern_match(memorized_patterns, np.array([1, -1, 1, -1])) # Check the pattern match with memorized pattern 0
    0
    >>> pattern_match(memorized_patterns, np.array([1, 1, -1, 1])) # Check the pattern match with memorized pattern 1
    1
    >>> pattern_match(memorized_patterns, np.array([1, 1, 1, 1])) is None # Check the pattern match with none of the memorized patterns
    True

    """
    for i in range(len(memorized_patterns)) :
        if np.array_equal(memorized_patterns[i], pattern):
            return i
    
    return None
        
def dynamics(state, weights, max_iter) :
    """
    Update the state of a Hopfield network synchronously.

    This function computes the next state of the network by calculating the 
    dot product of the `weights` matrix and the current `state`, then applies 
    the activation function to the result. It continues until the state converges 
    or the maximum number of iterations is reached.

    Parameters
    ----------
    state : ndarray
        The current state of the network, represented as a 1-D array.
    weights : ndarray
        The weight matrix of the network, where each entry represents the 
        connection weight between pairs of neurons.
    max_iter : int
        The maximum number of iterations to perform.

    Returns
    -------
    list of ndarray
        A list containing the history of states visited during the iterations.

    Examples
    --------
    # Example 1: Convergence within the maximum number of iterations
    >>> patterns = np.array([[1, 1, -1, -1],
    ...                      [1, 1, -1, 1],
    ...                      [-1, 1, -1, 1]])
    >>> weights = hebbian_weights(patterns)
    >>> state = np.array([1, -1, 1, -1])
    >>> history = dynamics(state, weights, max_iter=10)
    Convergence achieved.

    >>> len(history) >= 2
    True
    >>> np.array_equal(history[-1], history[-2])
    True

    # Example 2: No convergence within the maximum number of iterations
    >>> state = np.array([1, -1, 1, -1])
    >>> history = dynamics(state, weights, max_iter=1)  # Setting max_iter low to force no convergence
    The network did not converge within the maximum number of iterations.

    >>> len(history) >= 2
    True

    """
    state_history = [state.copy()]
    for i in range(max_iter) :
        next_state = update(state, weights)
        state_history.append(next_state.copy())
        if np.array_equal(next_state, state):
            print(f"Convergence achieved.")
            break   
        state = next_state
    else:
        print("The network did not converge within the maximum number of iterations.")
    return state_history


def dynamics_async(state, weights, max_iter, convergence_num_iter):
    """
    Update the state of a Hopfield network asynchronously.

    This function updates a single, randomly chosen element of the `state` 
    based on the corresponding row of the `weights` matrix and applies the 
    activation function. This process simulates asynchronous updating in 
    the network.

    Parameters
    ----------
    state : ndarray
        The current state of the network, represented as a 1-D array.
    weights : ndarray
        The weight matrix of the network, where each entry represents the 
        connection weight between pairs of neurons.
    max_iter : int
        The maximum number of iterations to run.
    convergence_num_iter : int
        The number of iterations with unchanged state to declare convergence.

    Returns
    -------
    list of ndarray
        The whole state history from the initial state until convergence or the maximum number of iterations.

    Examples
    --------
    # Example 1: Convergence within the maximum number of iterations
    >>> patterns = np.array([[1, 1, -1, -1],
    ...                      [1, 1, -1, 1],
    ...                      [-1, 1, -1, 1]])
    >>> weights = hebbian_weights(patterns)
    >>> state = np.array([1, -1, 1, -1])
    >>> history = dynamics_async(state, weights, max_iter=100, convergence_num_iter=5)
    Convergence achieved.

    >>> len(history) >= 2
    True
    >>> all(np.array_equal(history[-i-1], history[-i-2]) for i in range(5))
    True

    # Example 2: No convergence within the maximum number of iterations
    >>> state = np.array([1, -1, 1, -1])
    >>> history = dynamics_async(state, weights, max_iter=1, convergence_num_iter=5)
    The network did not converge within the maximum number of iterations.

    >>> len(history) >= 2
    True

    """
    state_history = [state.copy()]
    unchanged_count = 0

    for i in range(max_iter) :

        next_state = update_async(state, weights)
        state_history.append(next_state.copy())

        if np.array_equal(next_state, state):
             unchanged_count += 1 
        else:
            unchanged_count = 0

        if unchanged_count >= convergence_num_iter:
            print(f"Convergence achieved.")
            break

        state = next_state
    else:
        print("The network did not converge within the maximum number of iterations.")

    return state_history


def hebbian_weights(patterns):
    """
    Compute the Hebbian weight matrix for a set of patterns.

    This function computes the weight matrix `W` for a Hopfield network using 
    the Hebbian learning rule. The weights are calculated as the average 
    outer product of the patterns.

    Parameters
    ----------
    patterns : ndarray
        A 2-D array of shape (M, N), where M is the number of patterns and 
        N is the size of each pattern.

    Returns
    -------
    ndarray
        A 2-D weight matrix of shape (N, N) where the diagonal elements are 
        set to zero.

    Examples
    --------
    >>> patterns = np.array([[ 1, 1, -1, -1],
    ...              [ 1, 1, -1,  1],
    ...              [-1, 1, -1,  1]])
    >>> hebbian_weights(patterns)
    array([[ 0.        ,  0.33333333, -0.33333333, -0.33333333],
           [ 0.33333333,  0.        , -1.        ,  0.33333333],
           [-0.33333333, -1.        ,  0.        , -0.33333333],
           [-0.33333333,  0.33333333, -0.33333333,  0.        ]])
    """
    M,N = patterns.shape
    W = np.dot(patterns.transpose(), patterns)/M
    np.fill_diagonal(W, 0)
    return W

def update(state, weights) :
    """
    Perform a synchronous update of the state in a Hopfield network.

    This function computes the new state of the network by taking the dot 
    product of the current state and the weight matrix and applying the 
    activation function.

    Parameters
    ----------
    state : ndarray
        The current state of the network, represented as a 1-D array.
    weights : ndarray
        The weight matrix of the network, where each entry represents the 
        connection weight between pairs of neurons.

    Returns
    -------
    ndarray
        The updated state of the network after one synchronous step.

    Examples
    --------
    >>> patterns = np.array([[1, 1, -1, -1],
    ...                      [1, 1, -1, 1],
    ...                      [-1, 1, -1, 1]])
    >>> weights = hebbian_weights(patterns)
    >>> state = np.array([1, -1, 1, -1])
    >>> update(state, weights) 
    array([-1, -1,  1, -1])

    """
    new_state = np.dot(weights,state)
    return activation(new_state)

def update_async(state, weights) :
    """
    Perform an asynchronous update of the state in a Hopfield network.

    This function updates a single, randomly chosen element of the state 
    based on the corresponding row of the weights matrix and applies the 
    activation function. This simulates asynchronous updating of neurons.

    Parameters
    ----------
    state : ndarray
        The current state of the network, represented as a 1-D array.
    weights : ndarray
        The weight matrix of the network, where each entry represents the 
        connection weight between pairs of neurons.

    Returns
    -------
    ndarray
        The updated state of the network after one asynchronous step.

    Examples
    --------
    >>> patterns = np.array([[1, 1, -1, -1],
    ...                      [1, 1, -1, 1],
    ...                      [-1, 1, -1, 1]])
    >>> weights = hebbian_weights(patterns)
    >>> state = np.array([1, -1, 1, -1])
    >>> new_state = update_async(state, weights)
    >>> new_state.shape == state.shape  # Check shape remains the same
    True
    >>> np.sum(new_state != state) in [0, 1]  # Maximum one element should change
    True

    """
    new_state = state.copy()
    i = np.random.randint(len(state))
    w_sum = np.dot(weights[i], state)
    new_state[i] = activation(w_sum)
    return new_state

def storkey_weights(patterns) :
    """
    Compute the Storkey weight matrix for a set of patterns.

    This function computes the weight matrix `W` using the Storkey learning 
    rule, which incorporates both Hebbian learning and local inhibition to 
    prevent runaway weights. This method ensures better performance for 
    larger numbers of stored patterns.

    Parameters
    ----------
    patterns : ndarray
        A 2-D array of shape (M, N), where M is the number of patterns and 
        N is the size of each pattern.

    Returns
    -------
    ndarray
        A 2-D weight matrix of shape (N, N) where the diagonal elements are 
        set to zero.

    Examples
    --------
    >>> patterns = np.array([[1, 1, -1, -1],
    ...                      [1, 1, -1, 1],
    ...                      [-1, 1, -1, 1]])
    >>> storkey_weights(patterns)
    array([[ 1.125,  0.25 , -0.25 , -0.5  ],
           [ 0.25 ,  0.625, -1.   ,  0.25 ],
           [-0.25 , -1.   ,  0.625, -0.25 ],
           [-0.5  ,  0.25 , -0.25 ,  1.125]])

    """

    N = len(patterns[0])
    W = np.zeros((N, N))
    for mu in patterns :
        q = np.array(mu).reshape(N,1)
        diag = np.diag(W)
        H = np.dot(W, q) + np.zeros(N) - (q*(diag).reshape(N,1) + np.zeros(N)) - (W - np.diag(diag))*mu
        W += 1/N*(q*mu - H*mu - (H.T)*q)
    return W


def create_checkerboard_pattern():
    """
    Generates a 50x50 checkerboard pattern with alternating 5x5 blocks of 1 and -1, then flattens the matrix to a 1D array.

    This function creates a 50x50 numpy array in a checkerboard pattern, where each 5x5 block alternates between
    values of 1 and -1 to form a recognizable visual pattern. The array is then flattened to a 1D vector.

    Parameters
    ----------
    None

    Returns
    -------
    numpy.ndarray
        A 1D array of length 2500, representing the flattened 50x50 checkerboard pattern with alternating blocks of 1 and -1.

    Examples
    >>> checkerboard = create_checkerboard_pattern()
    >>> checkerboard.shape
    (2500,)
    >>> checkerboard[0:10] 
    array([ 1.,  1.,  1.,  1.,  1., -1., -1., -1., -1., -1.])
    >>> np.sum(checkerboard == 1.0)
    1250
    >>> np.sum(checkerboard == -1.0)
    1250
    """
    base_pattern = np.array([[1, -1] * 5, [-1, 1] * 5] * 5)
    checkerboard = np.kron(base_pattern, np.ones((5, 5)))  
    return checkerboard.flatten()


def save_video(state_list, out_path, img_shape=None):
    """
    Creates and saves a video of the evolution of network states using matplotlib's ArtistAnimation.

    This function compiles a sequence of 2D numpy arrays representing network states over time
    into an animation, saved as a video file. Each state is visualized as a frame in grayscale.

    Parameters
    ----------
    state_list : list of numpy.ndarray
        List of 2D arrays (e.g., 50x50) representing network states at each time step.
    out_path : str
        File path, including filename, where the output video will be saved (e.g., 'video.mp4').
    img_shape : tuple of int
        The shape (height, width) to reshape the states before visualization.
    Returns
    -------
    None

    Examples
    --------
    >>> import numpy as np
    >>> import os
    >>> state_list = [np.ones((50, 50)), -np.ones((50, 50))]
    >>> save_video(state_list, 'output.mp4')
    >>> os.path.exists('output.mp4')
    True
    >>> os.remove('output.mp4')
    """
    fig, ax = plt.subplots()
    frames = [[ax.imshow(state.reshape(img_shape), cmap='gray', animated=True)] for state in state_list]
    animation = ArtistAnimation(fig, frames)
    animation.save(out_path, writer='ffmpeg')


def compute_energy(state, weights):
    """
    Compute the energy of a given state in a Hopfield network.

    This function calculates the energy associated with a particular network state
    and weight matrix based on the Hopfield energy function.

    Parameters
    ----------
    state : array_like
        Input state vector of the network. This should be a 1-D array representing
        the state of each neuron in the network, with each entry being either 1 or -1.
    weights : array_like
        Weight matrix of the Hopfield network. This should be a 2-D square array
        where each entry represents the synaptic weight between pairs of neurons.

    Returns
    -------
    energy_value : float
        The computed energy value associated with the provided state.
    
    Examples
    --------
    >>> state = np.array([1, -1, 1])
    >>> weights = np.array([[0, 1, -1], [1, 0, 1], [-1, 1, 0]])
    >>> (compute_energy(state, weights))
    3.0

    """
    return -0.5 * np.sum(weights * np.outer(state, state))


if __name__ == "__main__": # pragma: no cover
    import doctest
    print("Starting doctest") # pragma: no cover
    doctest.testmod() # pragma: no cover

























