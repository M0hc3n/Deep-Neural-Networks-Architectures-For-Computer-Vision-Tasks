import argparse
import sys
from modeling import train
from preparation import extract
import os
from core.loggging import logger


class ArgumentFactory(object):

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Program command help',
            usage='''<command> [<args>]

            The most commonly used git commands are:
            train     Training your model
            prepare      Prepare your data and running engineering features before training
            ''')
        parser.add_argument('command', help='Subcommand to run')

        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):
        parser = argparse.ArgumentParser(description='Traing your model cmd')

        parser.add_argument('command', help='Subcommand to run')
        parser = argparse.ArgumentParser(add_help=False)

        # hyperparameters sent by the client are passed as command-line arguments to the script.
        parser.add_argument('--epochs', type=int, default=10,
                                  help='model hyperparameter')
        parser.add_argument('--batch_size', type=int, default=32,
                                  help='model hyperparameter')
        parser.add_argument('--learning_rate', type=float,
                                  default=0.01, help='model hyperparameter')

        # input data and model directories
        parser.add_argument('--model_dir', type=str, default='training')
        parser.add_argument(
            '--model_name', type=str, default="model.ckpt")
        parser.add_argument(
            '--train', type=str, default='data/processed')
        parser.add_argument('--test', type=str, default='data/processed')

        # model training configurations
        parser.add_argument('--img_height', type=int, default=180)
        parser.add_argument('--img_width', type=int, default=180)

        args, _ = parser.parse_known_args()

        dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
        data_dir = args.train
        batch_size = args.batch_size
        learning_rate = args.learning_rate
        img_height = args.img_height
        img_width = args.img_width
        epochs = args.epochs
        model_dir = args.model_dir
        model_name = args.model_name
        checkpoint_path = os.path.join(model_dir, model_name)

        train.process(data_dir=data_dir,
                      batch_size=batch_size,
                      learning_rate=learning_rate,
                      img_height=img_height,
                      img_width=img_width,
                      epochs=epochs,
                      checkpoint_path=checkpoint_path,
                      )

    def prepare(self):
        parser = argparse.ArgumentParser(
            description='Prepare your data and running engineering features before training')
        parser.add_argument('command', help='Subcommand to run')

        parser.add_argument('--data_dir', type=str, default='data/processed')
        parser.add_argument('--dataset_url', type=str,
                            default='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz')

        args = parser.parse_args()

        data_dir = args.data_dir
        dataset_url = args.dataset_url

        data_dir = extract.process(dataset_url, data_dir)
        logger.info('processed data dir: {}'.format(data_dir))


if __name__ == '__main__':
    ArgumentFactory()
