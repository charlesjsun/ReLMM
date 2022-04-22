from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt

buffer = None

def show_obs(i):
    plt.imshow(buffer._observations[i]);plt.show()

def load_buffer(path, size=5000):
    global buffer
    buffer = ReplayBuffer(10000, (60, 60, 3), 1, (2,))
    buffer.load(path)
    print("num samples", buffer.num_samples)

