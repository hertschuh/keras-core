from keras_core import ops
from keras_core.api_export import keras_core_export
from keras_core.backend.common.keras_tensor import KerasTensor
from keras_core.layers.merging.base_merge import Merge


@keras_core_export("keras_core.layers.Add")
class Add(Merge):
    """Performs elementwise addition operation.

    It takes as input a list of tensors, all of the same shape,
    and returns a single tensor (also of the same shape).

    Examples:

    >>> input_shape = (2, 3, 4)
    >>> x1 = np.random.rand(*input_shape)
    >>> x2 = np.random.rand(*input_shape)
    >>> y = keras_core.layers.Add()([x1, x2])

    Usage in a Keras model:

    >>> input1 = keras_core.layers.Input(shape=(16,))
    >>> x1 = keras_core.layers.Dense(8, activation='relu')(input1)
    >>> input2 = keras_core.layers.Input(shape=(32,))
    >>> x2 = keras_core.layers.Dense(8, activation='relu')(input2)
    >>> # equivalent to `added = keras_core.layers.add([x1, x2])`
    >>> added = keras_core.layers.Add()([x1, x2])
    >>> out = keras_core.layers.Dense(4)(added)
    >>> model = keras_core.models.Model(inputs=[input1, input2], outputs=out)

    """

    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = ops.add(output, inputs[i])
        return output

    def compute_output_spec(self, inputs):
        output_shape = self.compute_output_shape([x.shape for x in inputs])
        output_sparse = all(x.sparse for x in inputs)
        return KerasTensor(
            output_shape, dtype=self.compute_dtype, sparse=output_sparse
        )


@keras_core_export("keras_core.layers.add")
def add(inputs, **kwargs):
    """Functional interface to the `keras_core.layers.Add` layer.

    Args:
        inputs: A list of input tensors with the same shape.
        **kwargs: Standard layer keyword arguments.

    Returns:
        A tensor as the sum of the inputs. It has the same shape as the inputs.

    Examples:

    >>> input_shape = (2, 3, 4)
    >>> x1 = np.random.rand(*input_shape)
    >>> x2 = np.random.rand(*input_shape)
    >>> y = keras_core.layers.add([x1, x2])

    Usage in a Keras model:

    >>> input1 = keras_core.layers.Input(shape=(16,))
    >>> x1 = keras_core.layers.Dense(8, activation='relu')(input1)
    >>> input2 = keras_core.layers.Input(shape=(32,))
    >>> x2 = keras_core.layers.Dense(8, activation='relu')(input2)
    >>> added = keras_core.layers.add([x1, x2])
    >>> out = keras_core.layers.Dense(4)(added)
    >>> model = keras_core.models.Model(inputs=[input1, input2], outputs=out)

    """
    return Add(**kwargs)(inputs)
