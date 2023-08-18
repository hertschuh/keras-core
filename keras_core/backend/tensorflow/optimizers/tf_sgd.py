import tensorflow as tf

from keras_core import ops
from keras_core import optimizers


class SGD(optimizers.SGD):
    def _sparse_update_step(self, gradient, variable, learning_rate):
        """Update step given gradient and the associated model variable."""
        learning_rate = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        m = None
        if self.momentum != 0:
            m = self.momentums[self._get_variable_index(variable)]

        add_value = tf.IndexedSlices(
            -gradient.values * learning_rate, gradient.indices
        )
        if m is not None:
            momentum = ops.cast(self.momentum, variable.dtype)
            m.assign(m * momentum)
            print("###nesterov m scatter_add")
            m.scatter_add(add_value)
            if self.nesterov:
                print("###nesterov scatter_add")
                variable.scatter_add(add_value)
                variable.assign_add(m * momentum)
            else:
                variable.assign_add(m)
        else:
            print("###scatter_add")
            variable.scatter_add(add_value)
