import numpy as np
import tensorflow as tf


class dataloader():
    def __init__(self, 
                 coeff = 1,
                 batch_size=20):
        self.batch_size = batch_size
        self.N = 2000
        self.coeff = coeff
        self._reset()

    def __iter__(self):
        return self

    def __len__(self):
        #N = len(self.dataset)
        N = self.N
        b = self.batch_size
        return N // b + bool(N % b)

    def _reset(self):
        self._idx = 0

    def __next__(self):
        if self._idx >= self.N:
            self._reset()
            raise StopIteration()

        ### collocation points and governing equations
        x = (5 * np.random.rand(self.batch_size))[:, np.newaxis].astype(np.float32)
        y = self.coeff * x
        ###

        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)
        self._idx += self.batch_size
        return x, y
