"""E2E model test for numerical correctness verification.

We use a small keras application model (VGG16) as baseline, with
pretrained weights, and deterministic dataset (MNIST). The loss/
metric value will be compared against the golden value.
"""

import warnings
import tensorflow as tf

from keras_core import activations
from keras_core import backend
from keras_core import layers
from keras_core import models
from keras_core.utils import file_utils


def VGG16(
    include_top=True,
    weights="imagenet",
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    if not (weights in {"imagenet", None} or tf.io.gfile.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded.  Received: "
            f"weights={weights}"
        )

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError(
            'If using `weights` as `"imagenet"` with `include_top` '
            "as true, `classes` should be 1000.  "
            f"Received `classes={classes}`"
        )
    # Determine proper input shape
    input_shape = obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights,
    )

    img_input = layers.Input(shape=input_shape)
    # Block 1
    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block1_conv1"
    )(img_input)
    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block1_conv2"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2
    x = layers.Conv2D(
        128, (3, 3), activation="relu", padding="same", name="block2_conv1"
    )(x)
    x = layers.Conv2D(
        128, (3, 3), activation="relu", padding="same", name="block2_conv2"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3
    x = layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3_conv1"
    )(x)
    x = layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3_conv2"
    )(x)
    x = layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3_conv3"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

    # Block 4
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4_conv1"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4_conv2"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4_conv3"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

    # Block 5
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block5_conv1"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block5_conv2"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block5_conv3"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)

    if include_top:
        # Classification block
        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(4096, activation="relu", name="fc1")(x)
        x = layers.Dense(4096, activation="relu", name="fc2")(x)

        validate_activation(classifier_activation, weights)
        x = layers.Dense(
            classes, activation=classifier_activation, name="predictions"
        )(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D()(x)

    inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name="vgg16")

    # Load weights.
    if weights == "imagenet":
        if include_top:
            weights_path = file_utils.get_file(
                "vgg16_weights_tf_dim_ordering_tf_kernels.h5",
                WEIGHTS_PATH,
                cache_subdir="models",
                file_hash="64373286793e3c8b2b4e3219cbf3544b",
            )
        else:
            weights_path = file_utils.get_file(
                "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
                WEIGHTS_PATH_NO_TOP,
                cache_subdir="models",
                file_hash="6d6bbae143d832006294945121d1f1fc",
            )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def obtain_input_shape(
    input_shape,
    default_size,
    min_size,
    data_format,
    require_flatten,
    weights=None,
):
    """Internal utility to compute/validate a model's input shape.

    Args:
      input_shape: Either None (will return the default network input shape),
        or a user-provided shape to be validated.
      default_size: Default input width/height for the model.
      min_size: Minimum input width/height accepted by the model.
      data_format: Image data format to use.
      require_flatten: Whether the model is expected to
        be linked to a classifier via a Flatten layer.
      weights: One of `None` (random initialization)
        or 'imagenet' (pre-training on ImageNet).
        If weights='imagenet' input channels must be equal to 3.

    Returns:
      An integer shape tuple (may include None entries).

    Raises:
      ValueError: In case of invalid argument values.
    """
    if weights != "imagenet" and input_shape and len(input_shape) == 3:
        if data_format == "channels_first":
            if input_shape[0] not in {1, 3}:
                warnings.warn(
                    "This model usually expects 1 or 3 input channels. "
                    "However, it was passed an input_shape with "
                    + str(input_shape[0])
                    + " input channels.",
                    stacklevel=2,
                )
            default_shape = (input_shape[0], default_size, default_size)
        else:
            if input_shape[-1] not in {1, 3}:
                warnings.warn(
                    "This model usually expects 1 or 3 input channels. "
                    "However, it was passed an input_shape with "
                    + str(input_shape[-1])
                    + " input channels.",
                    stacklevel=2,
                )
            default_shape = (default_size, default_size, input_shape[-1])
    else:
        if data_format == "channels_first":
            default_shape = (3, default_size, default_size)
        else:
            default_shape = (default_size, default_size, 3)
    if weights == "imagenet" and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError(
                    "When setting `include_top=True` "
                    "and loading `imagenet` weights, "
                    f"`input_shape` should be {default_shape}.  "
                    f"Received: input_shape={input_shape}"
                )
        return default_shape
    if input_shape:
        if data_format == "channels_first":
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        "`input_shape` must be a tuple of three integers."
                    )
                if input_shape[0] != 3 and weights == "imagenet":
                    raise ValueError(
                        "The input must have 3 channels; Received "
                        f"`input_shape={input_shape}`"
                    )
                if (
                    input_shape[1] is not None and input_shape[1] < min_size
                ) or (input_shape[2] is not None and input_shape[2] < min_size):
                    raise ValueError(
                        f"Input size must be at least {min_size}"
                        f"x{min_size}; Received: "
                        f"input_shape={input_shape}"
                    )
        else:
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        "`input_shape` must be a tuple of three integers."
                    )
                if input_shape[-1] != 3 and weights == "imagenet":
                    raise ValueError(
                        "The input must have 3 channels; Received "
                        f"`input_shape={input_shape}`"
                    )
                if (
                    input_shape[0] is not None and input_shape[0] < min_size
                ) or (input_shape[1] is not None and input_shape[1] < min_size):
                    raise ValueError(
                        "Input size must be at least "
                        f"{min_size}x{min_size}; Received: "
                        f"input_shape={input_shape}"
                    )
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == "channels_first":
                input_shape = (3, None, None)
            else:
                input_shape = (None, None, 3)
    if require_flatten:
        if None in input_shape:
            raise ValueError(
                "If `include_top` is True, "
                "you should specify a static `input_shape`. "
                f"Received: input_shape={input_shape}"
            )
    return input_shape


def validate_activation(classifier_activation, weights):
    """validates that the classifer_activation is compatible with the weights.

    Args:
      classifier_activation: str or callable activation function
      weights: The pretrained weights to load.

    Raises:
      ValueError: if an activation other than `None` or `softmax` are used with
        pretrained weights.
    """
    if weights is None:
        return

    classifier_activation = activations.get(classifier_activation)
    if classifier_activation not in {
        activations.get("softmax"),
        activations.get(None),
    }:
        raise ValueError(
            "Only `None` and `softmax` activations are allowed "
            "for the `classifier_activation` argument when using "
            "pretrained weights, with `include_top=True`; Received: "
            f"classifier_activation={classifier_activation}"
        )

