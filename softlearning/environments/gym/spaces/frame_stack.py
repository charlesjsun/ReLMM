import numpy as np

from collections import OrderedDict, deque

from gym.spaces import Box

class FrameStack(Box):
    class Stack:
        def __init__(self, frames):
            self.frames = frames

        def numpy(self):
            return np.concatenate(self.frames, axis=-1)

    class Queue:
        def __init__(self, num_stack):
            self.num_stack = num_stack
            self.frames = deque(maxlen=self.num_stack)

        def reset(self):
            self.frames.clear()

        def append(self, frame):
            if len(self.frames) == 0:
                for _ in range(self.num_stack):
                    self.frames.append(frame)
            else:
                self.frames.append(frame)
            
        def stack(self):
            return FrameStack.Stack(list(self.frames))

    def __init__(self, frame_shape, num_stack):
        super().__init__(low=0, high=255, dtype=np.uint8,
            shape=frame_shape[:-1] + (frame_shape[-1] * num_stack,))

    @classmethod
    def process_batch(cls, batch):
        """ 
            batch: [batch_size, 1] array of FrameStack.Stack objects
            returns: [batch_size, shape] array
        """
        return np.concatenate([stack.numpy()[np.newaxis, ...] for stack in batch.squeeze()], axis=0)