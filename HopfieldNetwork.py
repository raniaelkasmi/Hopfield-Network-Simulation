from functions import*
from DataSaver import DataSaver

class HopfieldNetwork:
    def __init__(self, patterns, rule="hebbian"):
        N = (patterns[0])
        self.weights = np.zeros((N, N))  # Initialize weights to zero
        if rule == "hebbian":
            self.hebbian_weights(patterns)
        elif rule == "storkey":
            self.storkey_weights(patterns)
        else:
            raise ValueError("Unknown learning rule: choose 'hebbian' or 'storkey'.")

        
    def hebbian_weights(self, patterns):
        N = len(patterns[0])
        self.weights = np.zeros((N, N))
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)
        return self.weights

    def storkey_weights(self, patterns):
        N = len(patterns[0])
        self.weights = np.zeros((N, N))
        for mu in patterns :
            q = np.array(mu).reshape(N,1)
            diag = np.diag(self.weights)
            H = np.dot(self.weights, q) + np.zeros(N) - (q*(diag).reshape(N,1) + np.zeros(N)) - (self.weights - np.diag(diag))*mu
            self.weights += 1/N*(q*mu - H*mu - (H.T)*q)
        return self.weights
    
    def update(self, state):
        new_state = np.dot(self.weights,state)
        return activation(new_state)
    
    def update_async(self, state):
        new_state = state.copy()
        i = np.random.randint(len(state))
        w_sum = np.dot(self.weights[i], state)
        new_state[i] = activation(w_sum)
        return new_state
    
    def dynamics(self, state, saver, max_iter=20):
        saver.reset()
        for i in range(max_iter) :
            saver.store_iter(state, self.weights)
            new_state = self.update(state)
            if np.array_equal(new_state, state):
                print(f"Convergence achieved.")
                break   
            state = new_state
        else:
            print("The network did not converge within the maximum number of iterations.")
        saver.store_iter(state, self.weights)
    
    def dynamics_async(self, state, saver, max_iter=1000, convergence_num_iter=100, skip=10):
        saver.reset()
        previous_states = []
        for iteration in range(max_iter):
            if iteration % skip == 0:
                saver.store_iter(state, self.weights)
            state = self.update_async(state)
            previous_states.append(state)
            if len(previous_states) > convergence_num_iter:
                previous_states.pop(0)
                if all(np.array_equal(s, previous_states[0]) for s in previous_states):
                    break
        saver.store_iter(state, self.weights)
