import tensorflow as tf
import numpy as np


class DeepDream:
    def __init__(self, end_layers = ["mixed3", "mixed5"]):
        base_model = tf.keras.applications.InceptionV3(
            include_top=True, weights="imagenet"
        )
        layers = [base_model.get_layer(name).output for name in end_layers]
        self.dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

    def dream(
        self,
        img: np.ndarray,
        steps_per_octave: int = 10,
        step_size: float = 0.02,
        n_octaves: int = 4,
        octave_scale: float = 1.5,
    ):
        img = tf.convert_to_tensor(img)
        base_shape = tf.shape(img)
        img = tf.keras.utils.img_to_array(img)
        img = tf.keras.applications.inception_v3.preprocess_input(img)

        octaves = range(1 - n_octaves, 1)
        for octave in octaves:
            # Scale the image based on the octave
            new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32) * (
                octave_scale**octave
            )
            new_size = tf.cast(new_size, tf.int32)
            img = tf.image.resize(img, new_size)

            for _ in range(steps_per_octave):
                gradients = self._get_tiled_gradients(img, new_size)
                img = img + gradients * step_size
                img = tf.clip_by_value(img, -1, 1)

        img = tf.image.resize(img, base_shape[:2])
        result = tf.cast(255 * (img + 1.0) / 2.0, tf.uint8)
        return result.numpy()

    @tf.function
    def _get_tiled_gradients(self, img, img_size, tile_size=512):
        shift, img_rolled = self._random_roll(img, tile_size)

        # Initialize the image gradients to zero.
        gradients = tf.zeros_like(img_rolled)

        # Skip the last tile, unless there's only one tile.
        xs = tf.range(0, img_size[1], tile_size)[:-1]
        if not tf.cast(len(xs), bool):
            xs = tf.constant([0])
        ys = tf.range(0, img_size[0], tile_size)[:-1]
        if not tf.cast(len(ys), bool):
            ys = tf.constant([0])

        for x in xs:
            for y in ys:
                # Calculate the gradients for this tile.
                with tf.GradientTape() as tape:
                    # This needs gradients relative to `img_rolled`.
                    # `GradientTape` only watches `tf.Variable`s by default.
                    tape.watch(img_rolled)

                    # Extract a tile out of the image.
                    img_tile = img_rolled[y : y + tile_size, x : x + tile_size]
                    activation = self._dream_activation(img_tile)

                    # Update the image gradients for this tile.
                    gradients = gradients + tape.gradient(activation, img_rolled)

        # Undo the random shift applied to the image and its gradients.
        gradients = tf.roll(gradients, shift=-shift, axis=[0, 1])

        # Normalize the gradients.
        gradients /= tf.math.reduce_std(gradients) + 1e-8

        return gradients

    def _dream_activation(self, img: tf.Tensor):
        img_batch = tf.expand_dims(img, axis=0)
        layer_activations = self.dream_model(img_batch)
        if len(layer_activations) == 1:
            layer_activations = [layer_activations]

        activations = []
        for activation in layer_activations:
            activation = tf.math.reduce_mean(activation)
            activations.append(activation)

        return tf.reduce_sum(activations)

    def _random_roll(self, img: tf.Tensor, maxroll: int):
        # Randomly shift the image to avoid tiled boundaries.
        shift = tf.random.uniform(
            shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32
        )
        img_rolled = tf.roll(img, shift=shift, axis=[0, 1])
        return shift, img_rolled
