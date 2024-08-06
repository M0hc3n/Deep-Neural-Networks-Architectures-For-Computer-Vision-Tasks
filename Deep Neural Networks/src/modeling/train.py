from modeling.model import ClassificationModel
import os
import tensorflow as tf
import logging
from core.loggging import logger
import argparse
import datetime


def process(data_dir: str,
            batch_size: int,
            learning_rate: int,
            img_height: str,
            img_width: str,
            epochs: str,
            checkpoint_path: str,
            log_dir: str = "logs/fit/"):

    logger.info('TRAINING ---> Starting training ...')
    classification = ClassificationModel()

    train_ds = classification.load_train_ds(
        data_dir, 0.2, "training", 123, (img_height, img_width), batch_size)
    logger.info('TRAINING ---> Loaded training dataset')

    val_ds = classification.load_val_ds(
        data_dir, 0.2, "validation", 123, (img_height, img_width), batch_size)
    logger.info('TRAINING ---> Loaded validation dataset')

    model = classification.create_model(
        (img_height, img_width, 3), learning_rate)

    log_dir = os.path.join(
        log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    logger.info('TRAINING ---> Starting training model')

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[cp_callback, tensorboard_callback]
    )

    logger.info('TRAINING ---> Successfully training model ...')

    logger.info(
        'PACKING ---> Starting load trained model from: {}'.format(checkpoint_path))

    model = classification.load_model_from_checkpoint(
        checkpoint_path, (img_height, img_width, 3), learning_rate)

    logger.info('PACKING ---> Load trained model success!')
