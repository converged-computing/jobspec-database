import argparse
import logging
import os
import sys

import pandas as pd
from fastai.text import TextLMDataBunch, URLs, language_model_learner, csv, AWD_LSTM, Transformer, load_learner, Path
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

def str2bool(v):
    """Convert a string to a bool, this makes sure argument parsing goes right,
     based on https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description='Script to generate Trump tweets')
    parser.add_argument('--job_id', '-j', metavar='STRING', default="", help="Job_id used for saving files")
    parser.add_argument('--data', '-d', metavar='STRING', default='./tweets/real_trump_2011.csv',
                        help="The CSV with the tweets")
    parser.add_argument("--train", '-t', type=str2bool, nargs='?', const=True, default=True,
                        help="Do we need to train a model?")
    parser.add_argument('--output_path', '-o', metavar='STRING', default='./output',
                        help="Where the output will be stored")
    parser.add_argument('--use_pretrained', '-p', metavar='BOOL', default=True,
                        help="Do we use a pretrained model?")
    parser.add_argument('--model_path', '-mp', metavar='STRING',
                        default=os.path.join(os.getcwd(), 'model/'),
                        help="Where the model is or should be stored")
    parser.add_argument('--model_name', '-m', metavar='STRING',
                        default='finetune_trump.pkl', help="Which model to load, if any")
    parser.add_argument('--n_tweets', '-n', metavar='STRING',
                        default=10, help="How many tweets to generate")
    parser.add_argument('--n_words', '-w', metavar='STRING',
                        default=300, help="How many words should be generated at a time")
    parser.add_argument('--architecture', '-a', metavar='STRING',
                        default='AWD_LSTM', help='Which architecture do we want to use?')
    parser.add_argument('--epochs', '-e', metavar='INT',
                        default=10, help='Amount of epochs in training')
    return parser.parse_args()


class TextGeneration:
    """
    Class for text generation deep learning networks, training, generation and evaluation.
    Partly based on https://github.com/granilace/TweetGenerator
    """
    def __init__(self):
        # Here we init stuff
        self.args = get_args()
        self.trained = False

        self.dropout = 0.5
        self.epochs = int(self.args.epochs)
        self.batch_size = 32

        self.train_df = None
        self.validation_df = None
        self.test_df = None
        self.model = None
        self.data_lm = None
        logger.info(self.args)

        # Make sure the path for saving the model exists
        if not os.path.exists(self.args.model_path):
            os.makedirs(self.args.model_path)

    def load_data(self):
        """Load the data from file and append to list,
         split it and store into dataframes"""
        logger.info("Start loading data")
        with open(self.args.data, 'r') as file:
            reader = csv.reader(file)
            data = []
            for row in reader:
                if len(row[0]) > 1:
                    # Check if tweet does not start with quote
                    if not (row[0].startswith('"') or ord(row[0][0]) == 820):
                        data.append(row[0])
            logger.info('The number of tweets: {}'.format(len(data)))

        # Split datset into set for training and for validation and testing
        train_data, validation_test_data = train_test_split(
            list(data),
            test_size=0.1,
            random_state=1
        )

        # Split validation and testing set into two sets
        validation_data, test_data = train_test_split(
            list(validation_test_data),
            test_size=0.1,
            random_state=1
        )

        self.train_df = pd.DataFrame({'tweet': train_data})
        self.validation_df = pd.DataFrame({'tweet': validation_data})
        self.test_df = pd.DataFrame({'tweet': test_data})

        print(len(self.train_df))
        print(len(self.validation_df))
        print(len(self.test_df))

    def train(self, epochs=1, batch_size=32):
        """Train language model, uses two one cycle processes on first iteration"""

        # Setup the databunch
        self.data_lm = TextLMDataBunch.from_df(
            path='data',
            train_df=self.train_df,
            valid_df=self.validation_df,
            test_df=self.test_df,
            text_cols='tweet',
            bs=batch_size
        )

        # Check if it has to initialise the model
        if not self.trained:
            logger.info("Using a pretrained_model to finetune: " + str(self.args.use_pretrained))

            # Select the correct architecture
            if self.args.architecture.lower() == 'awd_lstm':
                self.model = language_model_learner(self.data_lm, arch=AWD_LSTM,
                                                    pretrained=self.args.use_pretrained, drop_mult=self.dropout)
            else:
                self.model = language_model_learner(self.data_lm, arch=Transformer,
                                                    pretrained=self.args.use_pretrained, drop_mult=self.dropout)
            # Finetune last layers
            self.model.fit_one_cycle(1, 1e-2)
            # Unfreeze whole model for training
            self.model.unfreeze()
            # Train whole model
            self.model.fit_one_cycle(1, 1e-3)

            self.trained = True

        # Train
        self.model.fit(epochs, lr=1e-3, wd=1e-7)

    def test(self):
        """Validates the model on the test dataset"""
        test_metric = self.model.validate(self.data_lm.test_dl)
        logger.info("Test loss: " + str(test_metric[0]))
        logger.info("Test accuracy: " + str(test_metric[1]))

    @staticmethod
    def prettify_tweet(tweet):
        """Prettifies tweet by removing spaces around some tokens, mostly interpunction"""

        # Set up character sets
        no_leading_space = ['?', '!', ',', '.', '\'', '’', '”', 'n\'t', 'n’t', '%', ')', ':']
        no_trailing_space = ['$', '#', '“', '(']

        # Remove space in front of characters
        for char in no_leading_space:
            tweet = tweet.replace(' ' + char, char)

        # Remove space after characters
        for char in no_trailing_space:
            tweet = tweet.replace(char + ' ', char)
        return tweet

    def generate(self, count=10, max_words=280):
        """Generates new tweets with the language model"""
        logger.info("Generating tweets")
        generated_tweets = []

        # Generate text until we have enough tweets
        while len(generated_tweets) < count:
            # Generate text
            raw_generated = self.model.predict("xxbos", n_words=max_words, temperature=0.8)

            # Split on begin of sentence token
            raw_tweets = raw_generated.split("xxbos ")[1:]

            # Iterate through tweets, prettify the tweets and save the tweets
            for tweet in raw_tweets[:-1]:  # Skip last xxbos as it is 99% chance it is an incomplete tweet
                tweet = self.prettify_tweet(tweet)
                if tweet and len(tweet) <= 280:
                    generated_tweets.append(tweet)
        return generated_tweets

    def run(self):
        """Main part of program, handles all the different steps"""
        if self.args.train:
            self.load_data()

            logger.info("Start training the model")
            # self.train(epochs=3, batch_size=32)
            self.train(epochs=self.epochs, batch_size=64)

            logger.info("Start testing")
            self.test()

            if self.args.job_id == "":
                model_name = self.args.model_path
            else:
                model_name = str(self.args.job_id) + ".pkl"
            self.model.export(Path(os.path.join(self.args.model_path, model_name)))
        else:
            logger.info("Loading a pretrained model")
            self.model = load_learner(Path(self.args.model_path), self.args.model_name)
        generated_tweets = self.generate(int(self.args.n_tweets), int(self.args.n_words))
        for tweet in generated_tweets:
            print("Tweet:\n" + tweet)


def main():
    generation = TextGeneration()
    generation.run()


if __name__ == '__main__':
    main()
