n
import gzip
from typing import List
import numpy as np
from scipy.special import expit
from scipy.ndimage import rotate, zoom
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
import pickle
import os
import time

# import sys
# sys.path.append(os.path.dirname(os.path.abspath("../Neuronales_Netz_optimieren")))
# import importlib
# importlib.import_module(os.path.dirname(os.path.abspath("../Neuronales_Netz_optimieren/Lernkurve_plotten")))

# TODO (Thomas Schneider / 12-2023):
# * (v) learning rate as function of epoch
# * (v) restructure into classes
# * (v) generalize to have any count of layers with given dimension
# * (v) stop if specific accurrency is reached
# * define optional transformations of input:
#   * (v) color normalization
#   * (v) shifting (translation)
#   * (v) rotation
#   * (v) resizing
# * (v) define graphical monitoring
# * ( ) construct a generic net, learning its hyper parameter through itself(a network layer)

class Logger:
    LEVELS = {'ERROR': 1, 'WARN': 2, 'INFO': 3, 'DEBUG': 4, 'TRACE': 5, 'DRAW': 6, 'DRAWANDSAVE': 7}
    mode = LEVELS.get('DRAW', 3)
    start = time.time()
    silentClasses = []

    @staticmethod
    def info(txt: str):
        Logger.log(None, 'INFO', txt, None)

    @staticmethod
    def info(txt: str, items: None):
        Logger.log(None, 'INFO', txt, items)

    @staticmethod
    def log(instance, level: str, txt: str, items):
        if (instance and type(instance).__name__ in Logger.silentClasses):
            return
        if (Logger.mode >= Logger.LEVELS.get(level, 3)):
            if (callable(items)):
                items = items()
            if (isinstance(items, np.ndarray) and Logger.LEVELS.get(level, 2) >= Logger.LEVELS.get('TRACE', 3)):
                Image.fromarray(items).show()
                time.sleep(1.5)
            else:
                t = round(time.time() - Logger.start, 2)
                print(str(t) + ": " + txt, items)

class Storage(object):
    def __init__(self):
        self.set_current_dir()

# set working directroy to find relative resource files
    def set_current_dir(self):
        current_dir_origin = os.getcwd()
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        Logger.info("cwd: " + current_dir_origin + " => " + dname, None)
        os.chdir(dname)

    def open_images(self, filename):
        with gzip.open(filename, "rb") as file:
            data = file.read()
            return np.frombuffer(data, dtype=np.uint8, offset=16)\
                .reshape(-1, 28, 28)\
                .astype(np.float32)

    def open_labels(self, filename):
        with gzip.open(filename, "rb") as file:
            data = file.read()
            return np.frombuffer(data, dtype=np.uint8, offset=8)

    def load_data(self, file_data: str, file_pred, reshape_2: int=784, y_encoding: bool=False):
        X = self.open_images(file_data).reshape(-1, reshape_2)
        y = self.open_labels(file_pred)
        y_oh = None
        if (y_encoding):
            oh = OneHotEncoder()
            y_oh = oh.fit_transform(y.reshape(-1, 1)).toarray()
        return X, y, y_oh
    
    def shift_images_random(self, X_train, index_from: int, count: int=1000, max_random: int=3, reshape_1: int=28, reshape_2: int=28):
        images = X_train[index_from:(index_from + count), :] / 255.
        shift_x = np.random.randint(-max_random, max_random)
        shift_y = np.random.randint(-max_random, max_random)
        images = np.roll(images.reshape(-1, reshape_1, reshape_2), (shift_x, shift_y), axis=(1, 2))
        return images.reshape(-1, reshape_1 * reshape_2)

 
    def rot_images_random(self, X_train, index_from: int, count: int=1000, max_random: int=20, reshape: int=28):
        images = X_train[index_from:(index_from + count), :] / 255.
        images = rotate(images.reshape(-1, reshape, reshape), angle=np.random.randint(-max_random, max_random), axes=(1, 2), reshape=False)
        Logger.log(self, "DRAW", "images", lambda : images[0] * 255)
        return images.reshape(-1, reshape * reshape)

    def scale_images_random(self, X_train, index_from: int, count: int=1000, max_random: float=1.2, reshape: int=28):
        images = X_train[index_from:(index_from + count), :].reshape(-1, reshape, reshape) / 255.
        for i in range(0, count):
            image = Image.fromarray(images[i])
            images[i] = self.clipped_zoom(images[i], np.random.rand() * max_random)
        Logger.log(self, "DRAW", "images", lambda : images[0] * 255)
        return images.reshape(-1, reshape * reshape)

    def clipped_zoom(self, img, zoom_factor, **kwargs):

        h, w = img.shape[:2]

        # For multichannel images we don't want to apply the zoom factor to the RGB
        # dimension, so instead we create a tuple of zoom factors, one per array
        # dimension, with 1's for any trailing dimensions after the width and height.
        zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

        # Zooming out
        if zoom_factor < 1:

            # Bounding box of the zoomed-out image within the output array
            zh = int(np.round(h * zoom_factor))
            zw = int(np.round(w * zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2

            # Zero-padding
            out = np.zeros_like(img)
            out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

        # Zooming in
        elif zoom_factor > 1:

            # Bounding box of the zoomed-in region within the input array
            zh = int(np.round(h / zoom_factor))
            zw = int(np.round(w / zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2

            out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

            # `out` might still be slightly larger than `img` due to rounding, so
            # trim off any extra pixels at the edges
            trim_top = ((out.shape[0] - h) // 2)
            trim_left = ((out.shape[1] - w) // 2)
            out = out[trim_top:trim_top+h, trim_left:trim_left+w]

        # If zoom_factor == 1, just return the input array
        else:
            out = img
        return out

class Layer(object):
    def __init__(self, name: str, input_size: int, output_size: int, lr: float = 0.25):
        Logger.info("creating layer " + name + "[" + str(input_size) + "x" + str(output_size) + " (learn-rate: " + str(lr) + ")]", None)
        self.name = name
        self.lr = lr
        self.w = np.random.randn(output_size, input_size)

    def activation(self, X):
        return expit(X)

    def predict(self, X):
        return self.activation(self.w @ X.T).T
    
    def back_propagation(self, X, pred, e, lr):
        dw = (e * pred * (1 - pred)).T @ X / len(X)
        assert dw.shape == self.w.shape
        if not np.any(dw):
          Logger.log(self, 'DRAW', 'input: X', X)
          Logger.info(" not learning anything!")
        # e = e @ self.w # the original calcluates the error with old weights
        self.w = self.w + lr * dw
        return e @ self.w


class NeuralNetwork(object):
    def __init__(self, layers: List[Layer], lr = lambda x: 0.1):
        self.learn_rate = lr
        self.layers: List[Layer] = layers

    def train(self, epoch: int, X, y):
        t = [X]
        # each layer gives its output as input to the following layer
        for l in self.layers:
            t.append(l.predict(t[-1]))

        # error as diff between expection and last prediction/output
        # back propagation corrects weights of each layer given by error from layer behind
        pred = t[-1]
        e = y - pred
        lr = 0.1#round(self.learn_rate(epoch), 2)
        cost = self.cost(e)
        Logger.info(str(epoch), [lr, cost])
        for l in reversed(self.layers):
            tl = t.pop()
            e = l.back_propagation(t[-1], tl, e, lr)
        return cost

    def predict(self, X):
        al = X
        for l in self.layers:
            al = l.predict(al)

    def cost(self, e):
        s = (1 / 2) * e ** 2
        return np.mean(np.sum(s, axis=1))

# Logger.mode = Logger.LEVELS.get("DRAW")
store = Storage()
X_train, y_train, y_train_oh = store.load_data("../mnist/train-images-idx3-ubyte.gz", "../mnist/train-labels-idx1-ubyte.gz", 784, True)
X_test, y_test, y_test_oh = store.load_data("../mnist/t10k-images-idx3-ubyte.gz", "../mnist/t10k-labels-idx1-ubyte.gz", 784)

limits = [40000, 60000]
test_accs = []
train_accs = []
step = 10000
max_epochs = 50

for max_learn in limits:
    # model0 = test.NeuralNetwork()
    model = NeuralNetwork([Layer("first", 784, 100), Layer("second", 100, 10)], lambda x: (x +4) / ((x +4) ** 1.4))

    for i in range(0, max_epochs):
        for d in range(0, max_learn, step):
            # images = store.shift_images_random(X_train, d, step)
            images = store.scale_images_random(X_train, d, step)
            cost = model.train(i, images , y_train_oh[d:(d + step), :])

        if (cost < 0.01):
            Logger.info("stopping training in cause of pretty good predictions")
            break

        y_test_pred = model.predict(X_test / 255.)
        y_test_pred = np.argmax(y_test_pred, axis=0)
        test_acc = np.mean(y_test_pred == y_test)

        y_train_pred = model.predict(X_train / 255.)
        y_train_pred = np.argmax(y_train_pred, axis=0)
        train_acc = np.mean(y_train_pred == y_train)

        test_accs.append(test_acc)
        train_accs.append(train_acc)

import matplotlib.pyplot as plt

plt.plot(limits, train_accs, label="Training")
plt.plot(limits, test_accs, label="Test")

plt.legend()
plt.show()

print("finished")
