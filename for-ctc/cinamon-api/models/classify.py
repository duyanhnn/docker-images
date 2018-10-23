import os
import json
import pickle
import numpy as np
from glob import glob
from MeCab import Tagger
from collections import namedtuple, Counter
from unicodedata import normalize
from keras.models import Input, Model, load_model
from keras.layers import TimeDistributed, Bidirectional, LSTM, Dense, \
    Conv1D, AveragePooling1D, Reshape, GlobalAveragePooling1D, MaxPooling1D, \
    Dropout, Concatenate, Dot, Softmax, Multiply, Reshape, Lambda, RepeatVector
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
import ipdb

HOME = os.path.expanduser('~')
IMG_DIR = 'data/input_v1'
#IMG_BLACKLIST = 'data/img_blacklist.txt'
IMG_BLACKLIST = None
OCR_DIR = 'data/ocr_v2'
VOCAB_PATH = 'data/line_clf_vocab.txt'
#VOCAB_PATH = 'data/vocab.txt'
VOCAB_PICKLE_PATH = os.path.splitext(VOCAB_PATH)[0] + '.pkl'
EMBEDDINGS_PATH = os.path.join(HOME, 'datasets/fasttext/ja.txt')
MODEL_PATH = f'models/line_classification/model.h5'
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.01
EPOCHS = 20

Token = namedtuple('Token', 'token, label')

class Tokenizer(Tagger):
    
    def __init__(self):
        Tagger.__init__(self)
    
    def tokenize(self, sequence):
        '''Tokenize a string using mecab'''
        sequence = sequence.replace(' ', '')
        sequence = ''.join([normalize('NFKC', char) for char in sequence])
        sequence = self.parse(sequence).splitlines()
        sequence = [line.split('\t')[0] for line in sequence]
        sequence = [token for token in sequence if token != 'EOS']
        sequence = [list(token) if token.isnumeric() else [token] for token in sequence]
        sequence = [token for sublist in sequence for token in sublist]
        return sequence
    
class Vocab:
    
    def __init__(self, vocab_path, vocab_pickle_path):
        self.tokenizer = Tokenizer()
        if os.path.exists(vocab_pickle_path):
            self.embeddings = pickle.load(open(vocab_pickle_path, 'rb'))
        else:
            if type(vocab_path) == str:
                vocab_lines = open(vocab_path, 'r').readlines()
            elif type(vocab_path) == list:
                vocab_files = [open(f, 'r').readlines() for f in vocab_path]
                vocab_lines = [line for f in vocab_files for line in f]
            else:
                raise Exception('Invalid Vocab Path')
            self.vocab = [
                token for line in vocab_lines for token in self.tokenizer.tokenize(line)]
            self.vocab = self.vocab + [token.lower() for token in self.vocab]
            self.vocab = self.vocab + [char for token in self.vocab for char in token]
            self.vocab = self.vocab + ['GO', 'EOS', 'UNK', 'PAD']
            self.vocab = set(self.vocab)
            self.embeddings = self._parse_embeddings(EMBEDDINGS_PATH)
            pickle.dump(self.embeddings, open(vocab_pickle_path, 'wb'))
        self.vocab = sorted(self.embeddings.keys())
        self.characters = list(set(char for token in self.vocab for char in token))
        self.size = len(self.vocab)
        self.embedding_size = len(self.get_embed('GO'))
        
    def __str__(self):
        return self.vocab.__str__()
    
    __repr__ = __str__
    
    def has(self, token):
        return token in self.vocab
    
    def get_index(self, token):
        if self.has(token):
            return self.vocab.index(token)
        else:
            return self.get_index('UNK')
    
    def get_token(self, index):
        return self.vocab[index]
    
    def get_embed(self, token):
        return self.embeddings[token] if self.has(token) else self.embeddings['UNK']
    
    def tokenize(self, sequence):
        sequence = self.tokenizer.tokenize(sequence)
        sequence = [
            [token] if self.has(token) else list(token) for token in sequence]
        sequence = [token for sublist in sequence for token in sublist]
        return sequence

    def sample(self, length):
        '''Sample random string of len length from Vocab'''
        indices = np.random.randint(len(self.characters), size=length)
        sample = [self.characters[index] for index in indices]
        sample = ''.join(sample)
        return sample

    
    def _parse_embeddings(self, embeddings_path):
        embeddings = dict()
        with open(embeddings_path, 'r') as f:
            f.readline()
            for i, line in enumerate(f):
                line = line.split(' ')
                token = line[0]
                if self.has(token):
                    embedding = np.array(list(map(float, line[1:])))
                    embeddings[token] = embedding
        return embeddings


class Box:
    
    size = 8
    
    def __init__(self, coords):
        '''Coordinates order goes from top-left -> top-right -> bottom-right -> bottom-left'''
        self.x0, self.y0, self.x1, self.y1, self.x2, self.y2, self.x3, self.y3 = coords
    
    def as_array(self, translate_x=None, translate_y=None, scale_x=None, scale_y=None):
        '''Return box coordinates as a 1D numpy array with values scaled by `scale`'''
        xs = np.array([self.x0, self.x1, self.x2, self.x3]) + translate_x
        xs = xs * scale_x
        ys = np.array([self.y0, self.y1, self.y2, self.y3]) + translate_y
        ys = ys * scale_y
        # Reference: https://stackoverflow.com/questions/5347065/interweaving-two-numpy-arrays
        result = np.empty((xs.size + ys.size,), dtype=xs.dtype)
        result[0::2] = xs
        result[1::2] = ys
        return result


class Example:
    
    def __init__(self, text, target, vocab, target_dict):
        self.vocab = vocab
        self.target_dict = target_dict
        self.reverse_target_dict = {v: k for k, v in self.target_dict.items()}
        self.input = text
        self.target = target
        self.encoded_input = self._encode_input()
        self.encoded_target = self._encode_target()
        self.sequence_len = self.encoded_input.shape[0]
        
    def __str__(self):
        return f'Input: "{self.input}", Target: "{self.target}"'
    
    __repr__ = __str__

    def _encode_input(self):
        encoded_input = self.vocab.tokenize(self.input)
        encoded_input = [self.vocab.get_embed(token) for token in encoded_input]
        encoded_input = np.array(encoded_input)
        return encoded_input
    
    def _encode_target(self):
        if not self.target:
            return None
        encoded_target = np.zeros(len(self.target_dict))
        encoded_target[self.target_dict[self.target]] = 1
        return encoded_target
        
class Dataset:
    
    def __init__(self, data, vocab, target_dict):
        '''Create dataset wrapper
        Params:
            data: dict mapping input string to target class
            vocab: Vocab object to manage vocabulary
            target_dict: dictionary mapping target class name to index'''
        self.vocab = vocab
        self.embedding_size = self.vocab.embedding_size
        self.target_dict = target_dict
        self.target_num = len(self.target_dict)
        self.reverse_target_dict = {index: target for target, index in target_dict.items()}
        self.data = [
            Example(text, target, self.vocab, self.target_dict) for text, target in data.items()]
        self.min_sequence_len = min(example.sequence_len for example in self.data)
        self.max_sequence_len = max(example.sequence_len for example in self.data)
        self.size = len(self.data)
        self.X = self._encode_input() 
        self.y = self._encode_target()
        
    def __str__(self):
        return "X's shape: {}, y's shape: {}".format(
            self.X.shape, self.y.shape)
    
    __repr__ = __str__
        
    def _encode_input(self):
        '''Vectorize the data'''
        pad_value = self.vocab.get_embed('PAD')
        X = [example.encoded_input for example in self.data]
        X = pad_sequences(X, pad_value, self.max_sequence_len)
        return X
    
    def _encode_target(self):
        y = np.array([example.encoded_target for example in self.data])
        return y
    
def pad_sequences(sequences, value, max_len=None):
    '''Pad all sequences to the same length
    
    Args:
        sequences: list of rank 2 ndarray
        value: 1D ndarray with length matching the last dimension of each sequence
        
    Returns:
        result: 3D ndarray of padded sequences
    '''
    if max_len is None:
        max_len = max(sequence.shape[0] for sequence in sequences)
    result = []
    for sequence in sequences:
        pad_len = max_len - sequence.shape[0]
        padding = np.tile(value, (pad_len, 1))
        try:
            sequence = np.concatenate((sequence, padding))
        except:
            sequence = padding
        result.append(sequence)
    result = np.stack(result)
    return result

def build_model(dataset):
    input_layer = Input(shape=(None, dataset.embedding_size))
    out = Conv1D(64, 3, activation='relu')(input_layer)
    out = Conv1D(64, 3, activation='relu')(out)
    # out = MaxPooling1D(3)(out)
    out = Conv1D(128, 3, activation='relu')(out)
    # out = Conv1D(128, 3, activation='relu')(out)
    out = GlobalAveragePooling1D()(out)
    out = Dropout(0.5)(out)
    # out = AveragePooling1D(pool_size=2, strides=None, padding='valid')(inp)
    # out = Reshape([-1])(out)
    # out = Bidirectional(LSTM(units=64, dropout=0.4, recurrent_dropout=0.4))(inp)
    # out = Dense(64, activation='relu')(out)
    out = Dense(dataset.target_num, activation='softmax')(out)
    model = Model(input_layer, out) 
    return model
    
def predict_once(model, dataset, i):
    X = dataset.X[i:i+1]
    y = dataset.y[i:i+1]
    pred = model.predict(X)
    confidence = np.max(pred)
    pred = np.argmax(pred)
    pred = dataset.reverse_target_dict[pred]
    true = dataset.data[i]
    return pred, true, confidence

result_template = {
    'company_name': '',
    'company_address':'',
    'tel':'',
    'fax':'',
    'bank1': {
        'bank': '',
        'branch': '',
        'type_of_account': '',
        'account': '',
    },
    'bank2': {
        'bank': '',
        'branch': '',
        'type_of_account': '',
        'account': '',
    },
    'bank3': {
        'bank': '',
        'branch': '',
        'type_of_account': '',
        'account': '',
    },
    'bank4': {
        'bank': '',
        'branch': '',
        'type_of_account': '',
        'account': '',
    }
}

class InferenceModel:
    '''Model for inerence from text'''

    def __init__(self, model, vocab, target_dict):
        self.model = model if type(model) != str else load_model(model)
        self.vocab = vocab if type(vocab) != str else Vocab(None, vocab)
        self.target_dict = target_dict if type(target_dict) != str else json.load(open(target_dict))

    def predict(self, text):
        example = Example(text, None, self.vocab, self.target_dict)
        pad_value = self.vocab.get_embed('PAD')
        pred = [example.encoded_input]
        pred = pad_sequences(pred, pad_value, 35)
        pred = self.model.predict(pred)
        confidence = np.max(pred)
        pred = np.argmax(pred)
        pred = example.reverse_target_dict[pred]
        return pred, confidence

def add_null_class(data, vocab):
    '''Add random string as null class
    Params:
        data: dict holding text to class mappings
        vocab: Vocab object holding vocabulary to sample characters from
    Returns:
        data: new dict with added null class'''
    if any(value == 'null' for value in data.values()):
        raise Exception('Null class already present')
    min_sequence_len = min(len(text) for text in data.keys())
    max_sequence_len = max(len(text) for text in data.keys())
    target_classes = set(data.values())
    class_count = []
    for target_class in target_classes:
        target_class_values = [1 for value in data.values() if value == target_class]
        target_class_count = len(target_class_values)
        class_count.append(target_class_count)
    mean_class_count = np.mean(class_count).astype('int')
    for i in range(mean_class_count):
        sequence_len = np.random.randint(min_sequence_len, max_sequence_len+1)
        sample = vocab.sample(sequence_len)
        data[sample] = 'null'
    return data

def fill_imbalanced_classes(fillee, filler):
    '''Add data to make balanced classes
    Args:
        fillee: data dict to be filled
        filler: data dict to take data from
    Returns:
        result: fillee dict with balanced classes'''
    counter = Counter(fillee.values())
    [(most_common_label, most_common_count)] = counter.most_common(1)
    labels_to_fill = set(fillee.values()) - {most_common_label}
    for label in labels_to_fill:
        values = [k for k, v in filler.items() if v == label]
        current_label_count = counter[label]
        num_label_to_fill = most_common_count - current_label_count
        indices = np.random.randint(len(values), size=num_label_to_fill)
        fillee.update((values[i], label) for i in indices)
    return fillee

if __name__ == '__main__':
    VOCAB_PATH = glob('data/ocr_v3/0*.txt')
    VOCAB_PATH.append('data/line_clf_vocab_db.txt')
    VOCAB_PICKLE_PATH = f'data/line_clf_vocab_{VOCAB_PATH[0].split("/")[1]}_db_merged.pkl'
    LINE_CLF_DATA_PATH = 'data/line_clf_data_ocr_v3.json'
    LINE_CLF_DATA_DB_PATH = 'data/line_clf_data_db.json'
    vocab = Vocab(VOCAB_PATH, VOCAB_PICKLE_PATH)
    #vocab_to_sample_from = Vocab(TRAIN_DATA_VOCAB_PATH, TRAIN_DATA_VOCAB_PICKLE_PATH)
    data = json.load(open(LINE_CLF_DATA_PATH, 'r'))
    #data = {k: v for k, v in data.items() if v == 'address' or v == 'null'}
    data_from_db = json.load(open(LINE_CLF_DATA_DB_PATH, 'r'))
    data = fill_imbalanced_classes(data, data_from_db)
    ipdb.set_trace()
    #data = add_null_class(data, vocab_to_sample_from)
    target_set = sorted(set(data.values()))
    target_dict = {target: index for index, target in enumerate(target_set)}
    dataset = Dataset(data, vocab, target_dict)

    K.clear_session()
    is_training = True

    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        is_training = False
    else:
        model = build_model(dataset)
        model.compile(
                Adam(lr=0.005, decay=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    test_set_size = int(VALIDATION_SPLIT*dataset.size)
    test_indices = np.random.choice(dataset.size, test_set_size)
    train_indices = np.setdiff1d(np.arange(dataset.size), test_indices)

    X_train = dataset.X[train_indices]
    y_train = dataset.y[train_indices]
    X_test = dataset.X[test_indices]
    y_test = dataset.y[test_indices]

    if is_training:
        model.fit(
                X_train, y_train, batch_size=BATCH_SIZE, 
                validation_data=(X_test, y_test), 
                epochs=EPOCHS)
        model.save(MODEL_PATH)

    inference_model = InferenceModel(model, vocab, target_dict)
    file_path = 'data/ocr_v2/00001.txt'
    with open(file_path, 'r') as f:
        for line in f.readlines():
            result, conf = inference_model.predict(line)
            if result != 'null':
                print(line, result, conf)
