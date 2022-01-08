#!/usr/bin/python
info="""
##############################################################################
# TS-NN: Neural Network on base of Keras providing a generic interface to create
# one of:
# - ANN                (Artifical Neural Network)
# - CNN                (Convolutional Neural Network)
# - RNN (LSTM)         (Recurrent NN with Long-Short-Time-Memory)
# - GAN                (Generative Adversarial Networks)
# - Autoencoder (AED)  (Auto En-/Decoder Network)
# - RIF                (Reinforcment Network to optimize the base network)
#
# selecting by given learing data and property yml file: ts-nn.yml
# the hyper-properties of the neural net will be optimized through a reinforced 
# extra neural network. Hyperproperties are:
# - layer definitions (with unit count)
# - loss function selection
# - activation function selection
# - metrics
# - optimizier
#
# For further informations, see: ts-nn.yml
#
# NOTE: Another solution would be to optimize the hyper paramerers through evolutional
#       algorithms like in tsl2.nano.gp.
# NOTE: as some functions use eval(), run this script only in a trusted content!
#
# CURRENT STATE: minimized implementation for ANN, CNN
# TODO: encapsulate implementations per NN type
# TODO: have a look at https://keras.io/api/keras_tuner/
#
# cp Thomas Schneider / Jan-2022
#
# Tested with:
# images: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets
# csv   : https://www.kaggle.com/wordsforthewise/lending-club
#
# References:
# https://www.baeldung.com/cs/reinforcement-learning-neural-network
# https://awjuliani.medium.com/super-simple-reinforcement-learning-tutorial-part-2-ded33892c724
# https://gitlab.com/snieda/tsl2nano/-/tree/master/tsl2.nano.gp
##############################################################################
"""

import os, sys, yaml, joblib, json
import pickle #import dump, load
from random import randint
from datetime import date
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, LSTM, InputLayer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.utils import text_dataset_from_directory
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

name: str = ''
p = {} # load properties from ts-nn.yml
binary_data: bool = None

def main():
    global name
    name = sys.argv[0][0:-3]
    model_name = os.path.split(sys.argv[0])[-1][0: -3]
    global p
    p = load_properties(name)
    _print('='*80)
    for i, arg in enumerate(sys.argv):
        _print(f'Argument {i:>6}: {arg}')
    _print(p, 2)
    _print('='*80)
    if (len(sys.argv) == 2 and sys.argv[1] == '--help'):
        usage = """
        <model | --help> [predict[props]]
        
        with:
            model  : model file name without extension
            predict: tuple or list to predict or
                     starting with 'file:' loading csv file to predict or
                     starting with 'image:' loading image file to predict
            props  : property dictionary to override the loaded yml-properties
            --help : print this info

        examples:
            ts-nn.py my-model [1, 2, 3] {"verbose": 0}
            ts-nn.py my-model (1, 2, 3)
            ts-nn.py my-model image: my-image.png
            ts-nn.py my-model file: my-predict.csv {"verbose": 0}
        """
        print('usage: ' + sys.argv[0] + usage + info)
    else:
        model = create_network(sys.argv[1] if len(sys.argv) > 1 else model_name)
        if (len(sys.argv) > 2):
            if (len(sys.argv) > 3 and sys.argv[3].startswith('{')):
                p.update(json.loads(sys.argv[3]))
            predict_args(model, sys.argv[2])
        else:
            print('NO PREDICTION DATA GIVEN -> DOING NOTHING!')

def load_properties(name) -> dict:
    with open(name + '.yml', 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def create_network(model_name) -> Sequential:
    _print("creating or loading model: " + name)
    model=load(name)
    if (model == None):
        input_shape, classifications, train_data, test_data, expection_type = arrange_data()
        model = create_optimize_model(input_shape, classifications, train_data, test_data, expection_type)
        save(model)
    else:
        _print(model.summary(), 1)
    return model

def create_optimize_model(input_shape, classifications, train_xy, test_xy, expection_type) -> Sequential:
    rif = create_reinforcer()
    nn = get_model_type(train_xy)
    model: Sequential = None

    if (p['hyper'] == 'config'):
        return create_learn_model(nn, input_shape, train_xy, test_xy, expection_type)
    elif (p['hyper'] == 'reinforce'):
        model_observer = ModelObserver(nn, input_shape, train_xy, test_xy, expection_type)
        qlearn(rif, model_observer)
        return model_observer.model
    elif (p['hyper'] == 'evolution'):
        raise BaseException('evolutional algorithm are not implemented yet')
    else:
        raise BaseException('hyper must be one of: config, reinforce, evolution')
    # i: int = 0
    # while (optimize(rif, model)):
    #     _print('-'*80 + '\nMODEL ITERATION STEP ' + str(i) + '\n' + '-'*80)
    #     model = create_learn_model(nn, input_shape, train_data, test_data)
    #     model.save(name, str(i))
    # return model

# def qlearn_observer(nn, input_shape, train_data, test_data, action):
#     pass
    
# def optimize(reinforment, model) -> bool:
#     # TODO: implement main construct to learn best configuration of model...
#     return True if model == None else False

class ModelObserver:

    def __init__(self, nn: str, input_shape, train_data, test_data, expection_type):
        self.nn = nn
        self.input_shape = input_shape
        self.train_data = train_data
        self.test_data = test_data
        self.expection_type = expection_type
        self.status = 0
        self.model: Sequential
        self.i = 0

    def observe_next(self, action):
        self.i += 1
        _print('-'*80 + '\nMODEL OBSERVATION STEP ' + str(self.i) + '\n' + '-' *80)
        self.status += self.status
        self.model = create_learn_model(self.nn, self.input_shape, self.train_data, self.test_data, self.expection_type)
        loss = self.model.history.history['loss'][-1] if len(self.model.history.history) > 0 else 9999
        return self.status, 1 / loss, loss < 0.1 or self.i >= p['epochs'], None

"""
does the reinforced q-learning on the RIF network with n episodes
    rif_model   : reinforcement neural network to optimize a target model
    observe_next: callback function to evaluate next step on given action.
                  returns next_step: int, reward: int, done: bool
"""
def qlearn(rif_model: Sequential, observer: ModelObserver):
    rif = p['RIF']
    eps = rif['eps']
    observations = rif['observations']
    discount = rif['discount']
    for i in range(rif['episodes']):
        state = 0
        eps *= rif['eps_decay']
        done = False
        while not done:
            if np.random.random() < eps:
                action = np.random.randint(0, rif['actions'])
            else:
                action = np.argmax(rif_model.predict(np.identity(observations)[state:state + 1]))
            new_state, reward, done, _ = observer.observe_next(action)
            target = reward + discount * np.max(rif_model.predict(np.identity(observations)[new_state:new_state + 1]))
            target_vector = rif_model.predict(np.identity(observations)[state:state + 1])[0]
            target_vector[action] = target
            rif_model.fit(
                np.identity(observations)[state:state + 1], 
                target_vector.reshape(-1, rif['actions']), 
                epochs=1, verbose=0)
            state = new_state

def create_reinforcer() -> Sequential:
    return create_learn_model('RIF', (1, p['RIF']['observations']), None, None, 'regression')

"""
    The core method to create the model after preprocessing
"""
def create_learn_model(nn: str, input_shape, train_xy, test_xy, expection_type) -> Sequential:
    _print("creating network model for type: " + nn)
    early_stop = EarlyStopping(monitor='val_loss',patience=2)
    
    model = Sequential()
    for l in get_layers(nn, input_shape):
        _print('adding layer: ' + str(l))
        model.add(l)
    model.build(input_shape)
    model.compile(loss=rule('loss')[expection_type], optimizer=p[nn]['optimizer'], metrics=p[nn]['metrics'])
    _print(model.summary())

    if (train_xy is not None and test_xy is not None):
        results = model.fit(
            train_xy[0] if isinstance(train_xy, tuple) else train_xy,
            train_xy[1] if isinstance(train_xy, tuple) else None, 
            epochs=p['epochs'],
            validation_data=test_xy,
            callbacks=[early_stop])

    if (is_verbose() and model.history is not None):
        print_model_info(model, test_xy, expection_type)

    return model

# TODO: best part to be encapsulated into subclasses
def get_layers(type: str, input_shape: tuple) -> list:
    if (type == 'ANN'):
        return [
            Dense(get_randunits(input_shape)),
            Dropout(p['dropout']),
            Dense(get_randunits(input_shape)),
            Dropout(p['dropout']),
            Dense(get_randunits(input_shape)),
            Dropout(p['dropout']),
            Activation(p[type]['activation']),
            Dense(1, activation='sigmoid')]
    elif (type == 'CNN'):
        m = (input_shape[0] + input_shape[1]) // 2
        return [
            Conv2D(filters=randint(2, m), kernel_size=(randint(2, 12), randint(2, 12)), input_shape=input_shape),
            MaxPooling2D(pool_size=(randint(2, 8), randint(2, 8))),
            Flatten(),
            Dense(get_randunits(input_shape)),
            Activation(p[type]['activation']),
            Dropout(p['dropout']),
            Dense(1),
            Activation('sigmoid')]
    elif (type == 'RNN'):
        return [
            LSTM(get_randunits(input_shape), activation=p[type]['activation']),
            Dense(1)]
    elif (type == 'GAN'):
        raise BaseException(type + ' not implemented yet')
    elif (type == 'AED'):
        encoder = Sequential()
        minu = p['AED']['min-units']
        u = input_shape[0]
        it = [u]
        while (u > minu):
            u = u // 2
            it.append(u)
            encoder.add(Dense(u))
        decoder = Sequential()
        for u in reversed(it):
            encoder.add(Dense(u))
        return [encoder, decoder] 
    elif (type == 'RIF'):
        return [
            InputLayer(batch_input_shape=input_shape),
            Dense(20, activation=p[type]['activation']),
            Dense(p[type]['actions'], activation='linear')
        ]
    else:
        raise BaseException(type + ' not allowed')

def predict_args(model: Sequential, arg: str) -> np.array:
    X = None
    if (arg.startswith('(') or arg.startswith('[')):
        X = eval(arg)
    elif arg.startswith('image:'):
        X = imread(arg[6:])
    elif arg.startswith('file:'):
        X = pd.read_csv(arg[5:])
    predict(model, X)
    _print(prediction)

def predict(model: Sequential, to_predict: np.array) -> np.array:
    scaler = pickle.load(open(f'{name}-scaler.dump', 'rb'))
    scaler.transform(to_predict)
    return model.predict(to_predict)

def rule(name: str):
    return p['rules'][name]

def print_data_info(train_path, classifications):
    _print('-' * 80)
    _print('train-path    : ' + train_path)
    _print('classifcations: ' + str(classifications))
    _print('-' * 80)

def print_model_info(model: Sequential, test_xy: tuple, expection_type: str):
    losses = pd.DataFrame(model.history.history)
    losses[['loss','val_loss']].plot()

    X = test_xy[0] if isinstance(test_xy, tuple) else test_xy
    y = test_xy[1] if isinstance(test_xy, tuple) else test_xy.classes
    pred_probabilities = model.predict(X)
    if (expection_type == 'regression'):
        return
    predictions = pred_probabilities > 0.5
    _print(y, 2)
    _print(predictions, 2)
    _print(classification_report(y, predictions))
    _print(confusion_matrix(y, predictions))

""" load, classify and preprocess data """
def arrange_data():
    train_path = os.path.join(p['data-dir'], p['train-dir'])
    test_path = os.path.join(p['data-dir'], p['test-dir'])
    classifications = get_classifications(train_path)
    input_shape = None

    if (is_verbose()):
        print_data_info(train_path, classifications)

    if (is_binary_data(train_path, classifications)):
        train_xy, test_xy, input_shape = generate_transformed_images(train_path, test_path, classifications)
    else:
        train_xy, test_xy, input_shape = load_textual_data(train_path, test_path, classifications)

    # type to evaluate best loss function
    if (len(classifications) > 0):
        expection_type = 'binary' if len(classifications) == 2 else 'categorical'
    elif (isinstance(train_xy, tuple) and len(train_xy[1]) != train_xy[1].nunique() and train_xy[1].nunique() < p['max-categories']):
        expection_type = 'binary' if train_xy[1].nunique() == 2 else 'categorical'
    else:
        expection_type = 'regression'

    _print('input shape: ' + str(input_shape))
#    _print(train_data.class_indices)
    return input_shape, classifications, train_xy, test_xy, expection_type

def get_classifications(path: str) -> list:
    classifications = []
    for d in os.listdir(path):
        if (not d.startswith('.') and os.path.isdir(os.path.join(path, d))):
            classifications.append(d)
    return classifications if len(classifications) > 1 else []

def load_textual_data(train_path, test_path, classifications):
    df = pd.DataFrame()
    if not classifications:
        read_pars = ', parse_dates=' + str(p['parse-dates'])
        read_pars += ', delimiter=\'' + p['read-delimiter'] + '\''
        if (p['read-index-col'] and not str(p['y-column']).__contains__(':')):
            read_pars += ', index_col=' + str(p['y-column'])
        
        for f in [p['data-file']] if 'data-file' in p else os.listdir(train_path):
            try:
                cmd_read = 'pd.read_' + f[-3:] + '(\'' + os.path.abspath(train_path + f) + '\'' + read_pars + ')'
                d = eval(cmd_read)
            except BaseException as e:
                _print('\t' + str(e) + ' -> ' + cmd_read, 1)
                continue
            print(cmd_read + ' -> ' + str(d))
            df = pd.concat([df, d])
        if df.empty:
            raise BaseException('couldn''t load data file from ' + os.path.curdir)
        if (is_verbose()):
            _print('row-count: ' + str(len(df)))
            _print(df.info())
            _print(df.corr())
            _print('null values: \n' + str(df.isnull().sum()))
            _print('not-number-columns: ' + str(df.select_dtypes(['object']).columns))
            plt.figure(figsize=(12,7))
            sns.heatmap(df.corr(),annot=True,cmap='viridis')
            plt.ylim(10, 0)

        X_train, y_train, X_test, y_test = preprocess_textual_data(df)
        if (get_model_type((X_train, y_train)) == 'RNN' and isinstance(y_train[0], date)):
            return generate_time_series(X_train, y_train, X_test, y_test)
        else:
            return (X_train, y_train), (X_test, y_test), X_train.shape
    else:
        # ON CONSTRUCTION - NOT TESTED YET. TODO: preprocess data, evaluate input_shape
        train_set = text_dataset_from_directory(train_path)
        test_set = text_dataset_from_directory(test_path)
        return train_set, test_set, None
        # raise BaseException('folder classified textual data is not implemented yet!')

def preprocess_textual_data(df: pd.DataFrame):
    if (df.shape[1] > df.shape[0]):
        df = df.transpose()

    # drop or transform columns by configuration
    df = df.drop(labels=p['drop-columns'], axis=1)
    for k in p['transform']:
        if (isinstance(p['transform'][k], dict)):
            df[k] = df[k].map(p['transform'][k])
        else:
            df[k] = eval('df[k].apply(' + p['transform'][k] + ')')
            
    # remove rows without numerical values
    if (p['drop-not-number-columns']):
        df = df.drop(labels=df.select_dtypes(['object']).columns, axis=1)

    # remove rows with incomplete values
    if (p['drop-incomplete-rows']):
        df = df.dropna()
    
    # on big data, use only a sample
    if (p['sample-rows'] is not None and len(df) > p['sample-rows']):
        df = df.sample(n = p['sample-rows'], random_state=p['random-state'])

    _print('remaining shape after pre-processing: ' + str(df.shape))
    _print('columns: ' + str(df.columns))

    # extract X data and one y column
    y = df.iloc[:, p['y-column']] if isinstance(p['y-column'], int) else df[p['y-column']]
    # if (isinstance(y[0], date)): # the timeseriesgenerator needs y column in the data set
    #     X = df.values
    # else:
    X = df.drop(df.columns[p['y-column']] if isinstance(p['y-column'], int) else p['y-column'], axis=1).values
    
    # split train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p['test-size'], random_state=p['random-state'])

    # normalize data
    scaler = MinMaxScaler()
    scaler.fit_transform(X_train)
    scaler.transform(X_test)
    pickle.dump(scaler, open(f'{name}-scaler.dump', 'wb'))

    return X_train, y_train, X_test, y_test

def generate_time_series(X_train, y_train, X_test, y_test):
    # TODO: not working yet - on fit() there is a problem
    train_xy = np.insert(X_train, 0, y_train, axis=1)
    test_xy = np.insert(X_test, 0, y_test, axis=1)
    l = length=len(y_test) // 2
    s = train_xy.shape
    rows = s[0]
    cols = s[1]
    generator = TimeseriesGenerator(train_xy, train_xy, l, batch_size=p['RNN']['batch-size'])
    validation_generator = TimeseriesGenerator(test_xy, test_xy, length=l, batch_size=p['RNN']['batch-size'])
    return generator, validation_generator, (rows, l, cols)

"""
image data will be loaded from given paths. additionally, transformed (rotated, scaled, translated) images 
will be generated. the images must be stored in classification folders!
"""
def generate_transformed_images(train_path, test_path, classifications) -> tuple:
    input_shape = evaluate_image_shape(os.path.join(train_path, classifications[0]))
    # TODO: extract parameter to configuration yaml
    image_gen = ImageDataGenerator(
        rotation_range=20, # rotate the image 20 degrees
        width_shift_range=0.10, # Shift the pic width by a max of 5%
        height_shift_range=0.10, # Shift the pic height by a max of 5%
        rescale=1/255, # Rescale the image by normalzing it.
        shear_range=0.1, # Shear means cutting away part of the image (max 10%)
        zoom_range=0.1, # Zoom in by 10% max
        horizontal_flip=True, # Allow horizontal flipping
        fill_mode='nearest' # Fill in missing pixels with the nearest filled value
        )
    image_gen.flow_from_directory(train_path)

    train_data = image_gen.flow_from_directory(
        train_path,
        target_size=input_shape[:2],
        color_mode='rgb',
        batch_size=p['batch-size'],
        class_mode='binary')
    test_data = image_gen.flow_from_directory(
        test_path,
        target_size=input_shape[:2],
        color_mode='rgb',
        batch_size=p['batch-size'],
        class_mode='binary',shuffle=False)
    
    if (is_verbose()):
        image_name = os.listdir(os.path.join(train_path, classifications[0]))[0]
        image = imread(os.path.join(train_path, classifications[0], image_name))
        _print(image_name + ":" + str(image.shape))
        plt.imshow(image)

    return train_data, test_data, input_shape

def evaluate_image_shape(path) -> ():
    dim1 = []
    dim2 = []
    img = get_first_image(path)
    if img is None:
        raise BaseException('no image found in path: ' + path)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)
    
    if (is_verbose()):
        sns.jointplot(dim1,dim2)

    return (int(np.mean(dim1)), int(np.mean(dim2)), 3)

def get_first_image(path) -> np.array:
    img = None
    dir = os.listdir(path)
    _print('trying to load first image in dir ' + path + '(' + str(len(dir)) + ')')
    for image_filename in dir:
        try:
            img = imread(os.path.join(path, image_filename))
            _print('image found: ' + image_filename)
            return img
        except BaseException as ex:
            print("not an image: " + image_filename + 'Exception: + ' + str(ex))
            continue
    return None

def is_supervised(data):
    pass

def is_classifying(data):
    pass

def is_verbose() -> bool:
    return int(p['verbose']) > 0

def is_binary_data(train_path: str = None, classifications: list = None) -> bool:
    global binary_data
    if binary_data is None:
        path = os.path.join(train_path, classifications[0]) if classifications else None
        binary_data = True if path is not None and get_first_image(path) is not None else False
        _print('loading data as binary!')
    return binary_data

def get_data_type(data) -> str:
    return 'binary' if is_binary_data() else 'text'

def get_model_type(data) -> str:
    if (p['network-type'] is not None):
        return p['network-type']

    type = get_data_type(data)
    if (type == "binary"):
        return "CNN"
    elif (isinstance(data, TimeseriesGenerator) or len(data[1]) == data[1].nunique()):
        return "RNN"
    else: # text
        return "ANN"
    # GAN and AED must be defined by 'network-type'!

def get_randunits(input_shape: tuple) -> int:
    return min(p['max-units'], randint(input_shape[0], input_shape[0] * input_shape[1]))

def save(model, str_iteration: str=''):
    file = name + str_iteration + '.' + p['save-format']
    _print(f'saving model to : {file}')
    model.save(file, save_format=p['save-format'])

def load(name) -> Sequential:
    f = name + '.' + p['save-format']
    if (os.path.exists(f)):
        _print(f'loading model from {f}')
        return load_model(f)
    else:
        return None

def _print(obj, verbose=1):
    if (verbose <= int(p['verbose'])):
        print(obj)
    pass

##############################################################################
# main function
##############################################################################
if __name__ == '__main__':
    main()