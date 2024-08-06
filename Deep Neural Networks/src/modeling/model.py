import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class ClassificationModel(object):

    def create_model(self, input_shape: tuple, learning_rate: int = 0.1) -> Sequential:
        num_classes = 5

        model = Sequential([
            layers.experimental.preprocessing.Rescaling(
                1./255, input_shape=input_shape),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])

        opt = tf.keras.optimizers.Adam()

        model.compile(optimizer=opt,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])

        model.summary()

        return model

    def load_model_from_checkpoint(self, checkpoint_path: str, input_shape: tuple, learning_rate: int = 0.1):
        model = self.create_model(input_shape, learning_rate)

        model.load_weights(checkpoint_path)

        return model

    def load_train_ds(self, data_dir: str, validation_split: float, subset: str, seed: int, image_size: tuple, batch_size: float):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=image_size,
            batch_size=batch_size
        )

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

        return train_ds

    def load_val_ds(self, data_dir: str, validation_split: float, subset: str, seed: int, image_size: tuple, batch_size: float):
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=image_size,
            batch_size=batch_size
        )

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        return val_ds
