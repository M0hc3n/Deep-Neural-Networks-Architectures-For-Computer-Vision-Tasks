import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib
import shutil
from core.loggging import logger


class FlowerExtract(object):
    def __init__(self, dataset_url: str, data_dir: str):
        if dataset_url is None or data_dir is None:
            raise Exception("Please input right dataset_url and data_dir")

        self.dataset_url = dataset_url
        self.data_dir = data_dir

    def run(self) -> str:
        data_set_name = "flower_photos"

        processed_dir = os.path.join(self.data_dir, data_set_name)

        if os.path.exists(processed_dir):
            logger.info('return processed dataset {}'.format(processed_dir))
            return processed_dir

        output_data_dir = tf.keras.utils.get_file(
            data_set_name, origin=self.dataset_url, untar=True)

        output_data_dir = pathlib.Path(output_data_dir)

        image_count = len(list(output_data_dir.glob('*/*.jpg')))
        logger.info('collected: {}'.format(image_count))

        shutil.move(output_data_dir.as_posix(), self.data_dir)

        return os.path.join(self.data_dir, data_set_name)


def process(dataset_url: str, data_dir: str) -> str:
    ### prepare data independ with training ###
    logger.info("PREPARE ---> Starting data preparation for training ...")
    data_processor = FlowerExtract(dataset_url, data_dir)

    processed_data_dir = data_processor.run()
    logger.info(
        'PREPARE ---> Processed training data dir: {}'.format(processed_data_dir))

    return processed_data_dir
