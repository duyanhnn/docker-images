from MeCab import Tagger
import unicodedata
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model


class Tokenizer(Tagger):
    word_index = {}
    max_len = 0

    def __init__(self):
        Tagger.__init__(self)

    def tokenize(self, sequence):
        """
        Tokenize a string using mecab
        """

        sequence = sequence.replace(' ', '')
        sequence = ''.join([unicodedata.normalize('NFKC', char) for char in sequence])
        sequence = self.parse(sequence).splitlines()
        sequence = [line.split('\t')[0] for line in sequence]
        sequence = [token for token in sequence if token != 'EOS']
        sequence = [list(token) if token.isnumeric() else [token] for token in sequence]
        sequence = [token for sublist in sequence for token in sublist]
        return sequence

    def fit_on_texts(self, sentences):
        """
        Build dictionary word-index.
        Key is word, value is index
        """

        all_tokens = []
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            all_tokens.extend(tokens)
            if len(tokens) > self.max_len:
                self.max_len = len(tokens)
        unique_tokens = np.unique(all_tokens)

        for i, token in enumerate(unique_tokens):
            self.word_index[token] = i + 1

    def texts_to_sequences(self, sentences):
        """
        Map array of words in sentence to array of index
        """

        all_tokens = []
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            tokenized = []
            for token in tokens:
                id = 0
                if token in self.word_index:
                    id = self.word_index[token]
                tokenized.append(id)
            all_tokens.append(tokenized)
        return all_tokens


def predict(sentences, vocabulary_path, model_path):
    """
    Predict class of sentences

    :param sentences: list of sentence
    :param vocabulary_path: path of word_index file
    :param model_path: path of trained model file
    :return: class of sentences
    """

    # load vocabulary
    with open(vocabulary_path, 'rb') as input:
        word_index = pickle.load(input)

    if word_index is None:
        raise ValueError('No vocabulary for model')

    max_len = 20
    tokenizer = Tokenizer()
    tokenizer.word_index = word_index
    tokenized = tokenizer.texts_to_sequences(sentences)
    X_test = pad_sequences(tokenized, maxlen=max_len)
    model = load_model(model_path)
    pred = model.predict(X_test, batch_size=1024, verbose=1)
    predictions = np.round(np.argmax(pred, axis=1)).astype(int)
    return predictions