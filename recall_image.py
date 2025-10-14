import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

from functions import*

from timeit import default_timer as timer
from plots import*
from image_utils import*

image_path = 'images/image.jpg'
binarized_image = load_and_binarize_image(image_path)

patterns = binarized_image
plt.imshow(binarized_image, cmap='gray')
plt.title("Binarized image")
plt.show()
patterns=np.squeeze(patterns.flatten()).reshape(1,-1)
print(patterns)
corrupted_image=perturb_pattern(pattern=patterns.flatten(),num_perturb=3000)
weights = hebbian_weights(patterns)
final_states = dynamics(corrupted_image, weights, max_iter=100)
print(final_states)
final_states = [state.reshape((100, 100)) for state in final_states]
save_video(final_states, 'VideosPatternConverge/recall_image_sync.mp4')

states = []
state = corrupted_image
for _ in range(100):
    states.append(state)
    state = update(state, weights)
    if np.array_equal(state, states[-1]):
        break

visualize_dynamics(states, binarized_image.shape)