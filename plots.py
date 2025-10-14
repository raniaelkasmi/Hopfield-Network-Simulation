import matplotlib.pyplot as plt

def visualize_dynamics(states, image_shape):
    """
    Visualizes the network dynamics at each iteration.

    Parameters:
    - states: List of states representing the network at each iteration.
    - image_shape: Tuple representing the shape to reshape the state for visualization (e.g., (width, height)).

    Displays the first 10 steps of the network dynamics in a grid of subplots.
    """
    num_frames = len(states)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i, state in enumerate(states[:10]): 
        ax = axes[i//5, i%5]
        ax.imshow(state.reshape(image_shape), cmap='gray')
        ax.set_title(f"Step {i+1}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()