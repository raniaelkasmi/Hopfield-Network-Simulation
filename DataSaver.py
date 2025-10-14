from functions import*

class DataSaver:

    def __init__(self):
        self.states = []
        self.energies = []

    def reset(self):
        self.states = []
        self.energies = []

    def compute_energy(self, state, weights):
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
        >>> compute_energy(state, weights)
        3.0

        """
        return -0.5 * np.sum(weights * np.outer(state, state))

    def store_iter(self, state, weights):
        self.states.append(state.copy())
        self.energies.append(self.compute_energy(state, weights))


    def get_data(self):
        return self.states, self.energies

    def save_video(self, out_path, img_shape):
        """
        Creates and saves a video of the network states evolution using matplotlib's ArtistAnimation.

        This function compiles a sequence of 2D numpy arrays representing network states over time
        into an animation, which is then saved as a video file. Each state is visualized as a frame in grayscale.

        Parameters
        ----------
        out_path : str
        The file path, including filename, where the output video will be saved (e.g., 'video.mp4').
        img_shape : tuple of int, optional
        The shape (height, width) to reshape the states before visualization. 

        Returns
        -------
        None

        Examples
        --------
        >>> data_saver.save_video('output.mp4', img_shape=(50, 50))
        >>> os.path.exists('output.mp4')
        True
        >>> os.remove('output.mp4')
        """
        fig, ax = plt.subplots()
        frames = [[ax.imshow(state.reshape(img_shape), cmap='gray', animated=True)] for state in self.state_list]
        animation = ArtistAnimation(fig, frames)
        animation.save(out_path, writer='ffmpeg')


    def plot_energy(self):
        """
        Plots the evolution of the network's energy over time.

        This function generates a plot representing how the energy of the system changes during the network's evolution.
        The energy values are plotted against the iteration number (or time steps).

        Parameters
        ----------
        None

        Returns
        -------
        None

        Examples
        --------
        >>> data_saver.plot_energy()  # Plots the energy over iterations
        """
        plt.plot(self.energies)
        plt.xlabel('Itérations')  
        plt.ylabel('Énergie')  
        plt.title('Évolution de l\'énergie')  
        plt.show()  