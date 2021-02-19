"""
==================================================
Class implementing a NN using keras
==================================================
"""

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import AlphaDropout
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from collections import OrderedDict
from itertools import product
from copy import deepcopy
import logging
import time
import os
import copy
import json
import numpy as np
import pandas as pd
import queue
import random
import matplotlib.pyplot as plt
import shap
import csv
import warnings
import tensorflow as tf
import my_funcs as mf
from MetricsCallback import MetricsCallback
from Metrics import Metrics

os.environ["KERAS_BACKEND"] = "tensorflow"

class MlpHelper:
    """
    Class implementing fraud detection mlp using keras_tools.

    :ivar output_dir: Output directory : output_dir
    :ivar semlp_selued: Random seed for reproducibility: seed
    """

    def __init__(self, output_dir="./", seed=0):

        self.logger = logging.getLogger(self.__class__.__name__)
       
        # Create initial folders and set thye seed
        self.output_dir = output_dir
        self.model_dir = os.path.join(output_dir, "model")
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)
        self.seed = seed

        # Set folder for tensorflow callbacks
        self._tb_callback = TensorBoard(log_dir=os.path.join(output_dir, "logs/{}".format(time.time())))

        # Properties that gets initialized by method calls
        self._X = None
        self._Y = None
        self._feature_columns = None
        self._label = None
        self._activation = None
        self._lr = None
        self._dropout_perc = None
        self._model_type = None
        self._layers = None
        self._input_dim = None
        self._train_X = None
        self._train_Y = None
        self._test_X = None
        self._test_Y = None
        self._predictions = None
        self._metrics = None
        self._metrics_callback = None
        self._model_json = None
        self._l1 = None
        self._l2 = None
        self._kernel_regularizer = None
        self._training_history = {}
        self._best_params_list = None
        self._multi_class_flag = None
        self._num_classes = 2

        # Model could be a property but it's more convenient it is not because we want auto-completion in ipython shell
        self.model = None

    def load_dataset(self, label, input_fpath=None, feature_columns=None, df=pd.DataFrame(), scale_dataset=False ):
        """
        Load dataset into X and Y arrays.

        X is the feature array and Y is the target array.

        :param input_fpath: Path to input file
        :type input_fpath: ``str``
        :param label: Specify classification label
        :type label: ``str``
        :param feature_columns: Specify feature set
        :type feature_columns: ``list``
        :param no_header: True means there is no header in the input file
        :type no_header: ``bool``
        :param df: Input dataframe
        :type df: ``pandas.DataFrame``
        :param scale_dataset: If True it scales the dataset
        :type scale_dataset: ``boolean``
        :returns: An instance of FraudDetection mlp
        :rtype: ``core_scripts.nn.keras_tools.MlpHelper``
        """

        # Load file if df is not specified
        if df.empty:
            if input_fpath is None:
                self.logger.error("No DataFrame and no path given.")
            df, _ = mf.load_file(input_fpath, limit=None, to_be_sorted=False, index_col=0, header=0)

        # Define feature columns
        if feature_columns is None:
            feature_columns = list(df.columns)
            try:
                feature_columns.remove(label)
            except ValueError:
                # Convert label to integer and re-try
                try:
                    label = int(label)
                    feature_columns.remove(label)
                except ValueError:
                    self.logger.error("Cannot find column {0} in {1}".format(label, feature_columns))

        # Init features and labels arrays
        X = df[feature_columns].values
        Y = df[label].values

        # Check if target column is binary or multi-class
        Y_distr = np.unique(Y)
        if len(Y_distr) > 2:
            # Set multi_class_flag to True
            multi_class_flag = True
            # Set num_classes
            self.num_classes = len(Y_distr)
        elif len(Y_distr) == 2:
            # Set multi_class_flag_to_False
            multi_class_flag = False
            # Make sure Y array contains only 0s and 1s
            vfunc = np.vectorize(lambda x: 1 if x > 0 else 0)
            Y = vfunc(Y)
        else:
            error_msg = "The specified label column {0} is neither binary nor multi-class".format(label)
            self.logger.error(error_msg)
            raise Exception(error_msg)

        # Update X, Y, features, and target
        self.X = X
        self.Y = Y
        self.feature_columns = feature_columns
        self.label = label
        self.multi_class_flag = multi_class_flag
        self.train_X, self.train_Y, self.test_X, self.test_Y = (None, None, None, None)
        if scale_dataset:
          self.scale_dataset()

        return self

    def build_model(self, layers, input_dim, activation="relu", lr=0.01, dropout_perc=0.5, model_type="mlp", l1=0., l2=0.):
        """
        Buid prediction model.
        Activation function can be relu or tanh.
        Available model types are:
        mlp = standard multi layer perceptron
        mlp_output_dropout = mlp with dropout in output layer
        ml_dropout = mlp with dropout in each layer
        :param layers: Layers composing the neural network architecture
        :type layers: ``list``
        :param input_dim: Size of input record
        :type input_dim: ``int``
        :param activation: Activation function
        :type activation: ``str``
        :param lr: Learning rate
        :type lr: ``float``
        :param dropout_perc: Dropout percentage
        :type dropout_perc: ``float``
        :param model_type: Model type.
        :type model_type: ``str``
        :param l1: L1 regularization coefficient
        :type l1: ``float``
        :param l2: L2 regularization coefficient
        :type l2: ``float``
        :returns: An instance of FraudDetection mlp
        :rtype: ``core_scripts.nn.keras_tools.MlpHelper``
        """

        # Make sure layers gets properly initialized
        if not layers:
            self.logger.warning("Layers cannot be empty list.\n"
                                "Setting layers = [100]")
            layers = [100]
        layers = deepcopy(layers)
        layers.append(1)

        self.layers = layers
        self.logger.info("Layers = {}".format(self.layers))
        self.input_dim = input_dim
        self.activation = activation
        self.lr = lr
        self.dropout_perc = dropout_perc
        self.model_type = model_type
        self.l1 = l1
        self.l2 = l2
        self._reg = regularizers.l1_l2(l1=self.l1, l2=self.l2)

        if self.model_type:
            self.destroy_model()

        if model_type == "mlp":
            self.logger.info("Model is mlp.")
            self.model = self._mlp()
        elif model_type == "mlp_output_dropout":
            self.logger.info("Model is mlp with dropout on output layer.")
            self.model = self._mlp_output_dropout()
        elif model_type == "mlp_dropout":
            self.logger.info("Model is mlp with dropout on each layer.")
            self.model = self._mlp_dropout()
        elif model_type == "mlp_selu":
            self.logger.info("Model is mlp selu.")
            self.model = self._mlp_selu()
        else:
            self.logger.error("Cannot build model.\n"
                              "Available model types are: mlp, mlp_output_dropout, and mlp_dropout.")

        return self

    def _add_last_layer(self, model):
        if self.multi_class_flag:
            self.logger.info("multi-class softmax classification.")
            model.add(Dense(self.num_classes, activation="softmax", kernel_regularizer=self._reg))
            optimizer = Adam(lr=self.lr)
            model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        else:
            self.logger.info("binary classification.")
            model.add(Dense(self.layers[-1], activation="sigmoid", kernel_regularizer=self._reg))
            optimizer = Adam(lr=self.lr)
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def _mlp(self):
        """
        Create Multi-Layer Perceptron model.

        :returns: model
        :rtype: keras.models.Sequential
        """

        model = Sequential()
        model.add(Dense(self.layers[0], input_dim=self.input_dim, activation=self.activation,
                        kernel_regularizer=self._reg))
        for i in range(1, len(self.layers) - 1):
            model.add(Dense(self.layers[i], activation=self.activation, kernel_regularizer=self._reg))
        self._add_last_layer(model)

        return model

    def _mlp_output_dropout(self):
        """
        Create Multi-Layer Perceptron model with dropout in output layer.

        :returns: model
        :rtype: keras.models.Sequential
        """

        model = Sequential()
        model.add(Dense(self.layers[0], input_dim=self.input_dim, activation=self.activation,
                        kernel_regularizer=self._reg))
        for i in range(1, len(self.layers) - 1):
            model.add(Dense(self.layers[i], activation=self.activation, kernel_regularizer=self._reg))
        model.add(Dropout(self.dropout_perc))
        self._add_last_layer(model)

        return model

    def _mlp_dropout(self):
        """
        Create Multi-Layer Perceptron model with dropout in all layers.

        :returns: model
        :rtype: keras.models.Sequential
        """

        model = Sequential()
        model.add(Dense(self.layers[0], input_dim=self.input_dim, activation=self.activation,
                        kernel_regularizer=self._reg))
        model.add(Dropout(self.dropout_perc))
        for i in range(1, len(self.layers) - 1):
            model.add(Dense(self.layers[i], activation=self.activation, kernel_regularizer=self._reg))
            model.add(Dropout(self.dropout_perc))
        self._add_last_layer(model)

        return model

    def _mlp_selu(self):
        """
        Create Multi-Layer Perceptron model scaled exponential lineary units and alpha dropout.
        :returns: model
        :rtype: keras.models.Sequential
        """

        kernel_initializer = "lecun_normal"
        activation = "selu"

        model = Sequential()
        model.add(Dense(self.layers[0], input_dim=self.input_dim, activation=activation,
                        kernel_regularizer=self._reg, kernel_initializer=kernel_initializer))
        model.add(AlphaDropout(self.dropout_perc))
        for i in range(1, len(self.layers) - 1):
            model.add(Dense(self.layers[i], activation=activation, kernel_regularizer=self._reg,
                            kernel_initializer=kernel_initializer))
            model.add(AlphaDropout(self.dropout_perc))
        self._add_last_layer(model)

        return model

    def destroy_model(self):
        """
        Destroy existing model.
        :returns: An instance of FraudDetection mlp
        :rtype: ``core_scripts.nn.keras_tools.MlpHelper``
        """

        if self.model:
            del self.model
            self.model = None

        return self


    def train(self, nff=1, epochs=10, batch_size=50, class_weight=None, verbose=1):
        """
        Train neural network model.

        Training verbosity level: 0 = silent, 1 = progress bar, 2 = one line per epoch
        :param epochs: Training epochs
        :type epochs: ``int``
        :param batch_size: Size of training batches
        :type batch_size: ``int``
        :param class_weight:  Class weights to apply during training
        :type class_weight: ``dict``
        :param verbose: Verbosity level.
        :type verbose: ``int``
        :returns: An instance of FraudDetection mlp
        :rtype: ``core_scripts.nn.keras_tools.MlpHelper``
        """

        if not self._check_data():
            return self

        if not self._check_model():
            return self

        self.model.fit(self.X, self.Y, epochs=epochs, batch_size=batch_size, class_weight=class_weight,
                       callbacks=[self.tb_callback], verbose=verbose)
        return self


    def train_and_validate(self, epochs=10, batch_size=50, class_weight=None, verbose=1,
                           test_perc=10, create_split=True, train_index=None, test_index=None, lauc_fct=0.2):
        """
        Train and validate neural network model.
        Training verbosity level: 0 = silent, 1 = one line per epoch
        Here keras verbosity is disabled to allow proper formatting.
        :param epochs: Training epochs
        :type epochs: ``int``
        :param batch_size: Size of training batches
        :type batch_size: ``int``
        :param class_weight:  Class weights to apply during training
        :type class_weight: ``dict``
        :param verbose: Verbosity level.
        :type verbose: ``int``
        :test_perc: Percentage of dataset for testing
        :type test_perc: ``int``
        :param create_split: If True creates train and test splits
        :type create_split: ``bool``
        :param train_index: train index
        :type train_index: ``list`` or ``array-like``
        :param test_index: test index
        :type test_index: ``list`` or ``array-like``
        :param lauc_fct: Fraction of true positives to be included in auc calculation
        :type lauc_fct: ``float``
        :returns: An instance of FraudDetection mlp
        :rtype: ``core_scripts.nn.keras_tools.MlpHelper``
        """

        if train_index is None:
            train_index = []
        if test_index is None:
            test_index = []
        if not self._check_model():
            self.logger.info("Hey! You must build a model first")
            return self

        if create_split:
            self.create_train_test_splits(test_perc)
        elif len(train_index) + len(test_index) == len(self.Y):
            self.create_train_test_splits_by_index(train_index, test_index)

        validation_data = (self.test_X, self.test_Y)

        self._init_metrics_callback(list(validation_data), lauc_fct, batch_size=batch_size)

        self.model.fit(self.train_X, self.train_Y, epochs=epochs, batch_size=batch_size, validation_data=validation_data, class_weight=class_weight, callbacks=[self.metrics_callback], verbose=verbose)

        self._dump_training_history()

        return self

    def _init_metrics_callback(self, validation_data, lauc_fct, val_steps=0, batch_size=50):
        if self.multi_class_flag:
            self.logger.error("Error, you need MetricsCallbackMultiClass.") 
        else:
            self.metrics_callback = MetricsCallback(validation_data, lauc_fct, self.model_dir, validation_steps=val_steps, batch_size=batch_size)

    def _dump_training_history(self):

        training_history_fpath = os.path.join(self.output_dir, "logs/{}.txt".format(time.time()))
        if not os.path.isdir(os.path.dirname(training_history_fpath)):
            os.mkdir(os.path.dirname(training_history_fpath))
        df = pd.DataFrame(data=self.metrics_callback.training_history)
        self.training_history = df.round(decimals=6)
        self.training_history.to_csv(training_history_fpath, index=False)

    def create_train_test_splits(self, test_perc):
        """
        Create train and test splits.
        train_X(numpy.ndarray) is the feature matrix for training.
        train_Y(numpy.ndarray) is the array of labels for training.
        test_X(numpy.ndarray) is the feature matrix for testing.
        test_Y(numpy.ndarray) is the array of labels for testing
        :param test_perc: Percentage of test data
        :type test_perc: ``int``
        :returns: An instance of FraudDetection mlp
        :rtype: ``core_scripts.nn.keras_tools.MlpHelper``
        """

        if not self._check_data():
            return self

        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(self.X, self.Y,
                                                                                test_size=test_perc / 100,
                                                                                random_state=self.seed)

        return self


    def scale_dataset(self):
        """
        Scale the dataset.
        :returns: An instance of FraudDetection mlp
        :rtype: ``core_scripts.nn.keras_tools.MlpHelper``
        """

        if not self._check_data():
            return self
        if self.train_X is not None:
            scaler = preprocessing.StandardScaler().fit(self.train_X)
            self.train_X = scaler.transform(self.train_X)
            self.test_X = scaler.transform(self.test_X)
        else:
            scaler = preprocessing.StandardScaler().fit(self.X)
            self.train_X = scaler.transform(self.X)
        return self

    def create_train_test_splits_by_index(self, train_index, test_index):
        """
        Create train and test splits by explicit index

        :param train_index: train index
        :type train_index: ``list`` or ``array-like``
        :param test_index: test index
        :type test_index: ``list`` or ``array-like``
        :returns: An instance of FraudDetection mlp
        :rtype: ``core_scripts.nn.keras_tools.MlpHelper``
        """

        if not self._check_data():
            return self

        self.train_X, self.test_X, self.train_Y, self.test_Y = self.X[train_index], self.X[test_index], self.Y[train_index], self.Y[test_index]
        return self

    def predict(self, X, batch_size=50, verbose=2):
        """
        Predict probability of fraud using built model.

        Training verbosity level: 0 = silent, 1 = progress bar, 2 = one line per epoch

        :param X: Input data
        :type X: ``np.ndarray``
        :param batch_size: Size of training batches
        :type batch_size: ``int``
        :param verbose: Verbosity level.
        :type verbose: ``int``
        :returns: An instance of FraudDetection mlp
        :rtype: ``core_scripts.nn.keras_tools.MlpHelper``
        """

        if not self._check_model():
            return self

        self.predictions = self.model.predict_proba(X, batch_size=batch_size, verbose=verbose)

        return self

    def compute_metrics(self, target, predictions, fct_vals=None):
        """
        Compute metrics.
        :param target: Array of target values
        :type target: ``np.ndarray``
        :param predictions: Array of predicted values
        :type predictions: ``np.ndarray``
        :param fct_vals: Fractions of true positives to be included in auc calculation
        :type fct_vals: ``list``
        :returns: An instance of FraudDetection mlp
        :rtype: ``core_scripts.nn.keras_tools.MlpHelper``
        """

        if fct_vals is None:
            fct_vals = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        if self.multi_class_flag:
            accuracy = accuracy_score(target, np.argmax(predictions, axis=1))

            auc = []
            precision = []
            recall = []
            f1 = []
            target = to_categorical(target)
            for i in range(self.num_classes):
                auc.append(roc_auc_score(target[:, i], predictions[:, i]))
                precision.append(precision_score(target[:, i], np.rint(predictions[:, i])))
                recall.append(recall_score(target[:, i], np.rint(predictions[:, i])))
                f1.append(f1_score(target[:, i], np.rint(predictions[:, i])))
        else:
            accuracy = accuracy_score(target, np.rint(predictions))
            auc = roc_auc_score(target, predictions)
            precision = precision_score(target, np.rint(predictions))
            recall = recall_score(target, np.rint(predictions))
            f1 = f1_score(target, np.rint(predictions))

        if self.multi_class_flag:
            laucs = []
            for i in range(self.num_classes):
                laucs.append(dict([(fct, Metrics.compute_lauc(target[:, i], predictions[:, i], fct)) for fct in fct_vals]))
        else:
            laucs = dict([(fct, Metrics.compute_lauc(target, predictions, fct)) for fct in fct_vals])

        self.metrics = {
            "accuracy": accuracy,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "laucs": laucs
        }

        return self

    def print_metrics(self):
        """
        Print metrics.

        :returns: An instance of FraudDetection mlp
        :rtype: ``core_scripts.nn.keras_tools.MlpHelper``
        """

        if self.metrics:
            self.logger.info("Metrics at end of training epochs.")
            self.logger.info("{0:15}: {1:4.2f}".format("Test Accuracy", self.metrics["accuracy"]))
            self.logger.info("")
            if self.multi_class_flag:
                for i in range(self.num_classes):
                    self.logger.info("{0:15}: {1:4.2f}".format("Test AUC-%d" % i, self.metrics["auc"][i]))
                    self.logger.info("{0:15}: {1:4.2f}".format("Test Precision-%d" % i, self.metrics["precision"][i]))
                    self.logger.info("{0:15}: {1:4.2f}".format("Test Recall-%d" % i, self.metrics["recall"][i]))
                    self.logger.info("{0:15}: {1:4.2f}".format("Test F1-%d" % i, self.metrics["f1_score"][i]))
                    self.logger.info("")

                if self.metrics_callback:
                    for i in range(self.num_classes):
                        self.logger.info("{0:15}: {1:4.6f} at epoch {2}".format("Best LAUC-%d" % i, self.metrics_callback.lauc[i],
                                                                                self.metrics_callback.best_lauc_epoch))

                    self.logger.info("{0:15}: {1:4.6f} at epoch {2}".format("Best LOG LOSS", self.metrics_callback.log_loss,
                                                                            self.metrics_callback.best_log_loss_epoch))

                self.logger.info("")
                for i in range(self.num_classes):
                    for key, value in self.metrics["laucs"][i].items():
                        self.logger.info("{0:13}: {1:5}: {2:9}: {3:4.6f}".format("FP fraction", key, "Test LAUC-%d" % i, value))
                    self.logger.info("")
            else:
                self.logger.info("{0:15}: {1:4.2f}".format("Test AUC", self.metrics["auc"]))
                self.logger.info("{0:15}: {1:4.2f}".format("Test Precision", self.metrics["precision"]))
                self.logger.info("{0:15}: {1:4.2f}".format("Test Recall", self.metrics["recall"]))
                self.logger.info("{0:15}: {1:4.2f}".format("Test F1", self.metrics["f1_score"]))

                if self.metrics_callback:
                    self.logger.info("{0:15}: {1:4.6f} at epoch {2}".format("Best LAUC", self.metrics_callback.lauc,
                                                                            self.metrics_callback.best_lauc_epoch))
                    self.logger.info("{0:15}: {1:4.6f} at epoch {2}".format("Best LOG LOSS", self.metrics_callback.log_loss,
                                                                            self.metrics_callback.best_log_loss_epoch))

                for key, value in self.metrics["laucs"].items():
                    self.logger.info("{0:13}: {1:5}: {2:9}: {3:4.6f}".format("FP fraction", key, "Test LAUC", value))

        return self

    def save_model(self, name="model.h5"):
        """
        Save model.

        :param name: Model name
        :type name: ``str``
        :returns: An instance of FraudDetection mlp
        :rtype: ``core_scripts.nn.keras_tools.MlpHelper``
        """

        if self._check_model():
            model_fpath = os.path.join(self.model_dir, name)
            try:
                self.model.save(os.path.join(self.model_dir, name))
                self.logger.info("Model saved to: {}".format(model_fpath))
            except IOError:
                self.logger.error("Cannot save model to: {}".format(model_fpath))

        return self

    def load_model(self, model_fpath):
        """
        Load model from file. File format is h5.

        :param model_fpath: Model file path
        :type model_fpath: ``str``
        :returns: An instance of FraudDetection mlp
        :rtype: ``core_scripts.nn.keras_tools.MlpHelper``
        """

        if self._check_model():
            del self.model

        self.model = load_model(model_fpath)

        return self


    def _get_base_params(self):

        params = {
            "activation": ["relu"],
            "lr": [0.001],
            "dropout_perc": [0.5],
            "model_type": ["mlp_dropout"],
            "l1": [0.],
            "l2": [0.],
            "nff": [1],
            "epochs": [10],
            "batch_size": [50],
            "class_weight": [{0: 1.0, 1: 1.0}],
            "test_perc": [10],
            "lauc_fct": [0.2],
            "create_split": [True]
        }

        return params

    def build_train_and_validate(self, _params, verbose=1, seed=0, validate=True):
        """
        Build model and run train and validate

        Required parameters:

        .. code-block:: text

            params = {
            "layers": [50]
            }

        Optional parameters and their default values:

        .. code-block:: text

            params = {
            "activation": "relu",
            "lr": 0.001,
            "dropout_perc": 0.5,
            "model_type": "mlp_dropout",
            "l1": 0.,
            "l2" : 0.,
            "nff": 1,
            "epochs": 10,
            "batch_size": 50,
            "class_weight": {0: 1.0, 1: 1.0},
            "test_perc": 10,
            "lauc_fct": 0.2,
            "create_split": True
            }

        Example:

        .. code-block:: text

            params = {
            "layers": [30],
            "lr": 0.0001
            }

        :param _params: Input parameters
        :type _params: ``dict``
        :param verbose: Verbosity level.
        :type verbose: ``int``
        :param seed: Random seed for reproducibility
        :type seed: ``int``
        :param validate: If True run validation along training
        :type validate: ``bool``
        :returns:
        """

        # Merge base params with user params
        params = self._get_base_params()
        params = {key: value[0] for (key, value) in params.items()}
        params.update(_params)

        self.seed = seed
        self.build_model(params["layers"], self.X.shape[1], activation=params["activation"], lr=params["lr"],
                         dropout_perc=params["dropout_perc"], model_type=params["model_type"], l1=params["l1"],
                         l2=params["l2"])
        if validate:
            self.train_and_validate(nff=params["nff"], epochs=params["epochs"], batch_size=params["batch_size"],
                                    class_weight=params["class_weight"], verbose=verbose, test_perc=params["test_perc"],
                                    create_split=params["create_split"], lauc_fct=params["lauc_fct"])
        else:
            verbose = verbose if verbose == 0 else 2
            self.train(nff=params["nff"], epochs=params["epochs"], batch_size=params["batch_size"],
                       class_weight=params["class_weight"], verbose=verbose)

    def plot_history(self):
        """
        Plot training history.

        If are on jupyter notebook, make sure you run %matplotlib inline
        """

        if not self.training_history.empty:
            ax = plt.gca()
            self.training_history.plot(kind="line", x="Epoch", y="Train Loss", ax=ax)
            self.training_history.plot(kind="line", x="Epoch", y="Val Loss", color="red", ax=ax)
            if self.multi_class_flag:
                for i in range(self.num_classes):
                    self.training_history.plot(kind="line", x="Epoch", y="Val LAUC %d" % i, ax=ax)
            else:
                self.training_history.plot(kind="line", x="Epoch", y="Val LAUC", color="green", ax=ax)
            plt.show()

    def save_scores(self, name="scores.csv"):
        """
        Save scores to file.

        :param name: Name of scores file
        :type name: ``str``
        :returns: An instance of FraudDetection mlp
        :rtype: ``core_scripts.nn.keras_tools.MlpHelper``
        """

        if type(self.Y) != np.ndarray or type(self.predictions) != np.ndarray:
            self.logger.error("save_scores should be called after predict.")
            return

        save_scores(self.output_dir, self.Y, self.predictions.squeeze(), name=name, multi_class_flag=self.multi_class_flag)

        return self

    def _check_model(self):

        if type(self.model) != Sequential:
            return False
        else:
            return True

    def _check_data(self):

        if (type(self.X) != np.ndarray or type(self.Y) != np.ndarray) and \
                (type(self.X) != list or type(self.Y) != list):
            self.logger.info("Hey! You must load some data first.")
            return False
        else:
            return True


    # You have 2 methods with the same name. The decorator allows you to do class.X='A' and class.X without confusion.
    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        self._X = value

    @property
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, value):
        self._Y = value

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, value):
        self._activation = value

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, value):
        self._lr = value

    @property
    def dropout_perc(self):
        return self._dropout_perc

    @dropout_perc.setter
    def dropout_perc(self, value):
        self._dropout_perc = value

    @property
    def model_type(self):
        return self._model_type

    @model_type.setter
    def model_type(self, value):
        self._model_type = value

    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, value):
        self._layers = value

    @property
    def input_dim(self):
        return self._input_dim

    @input_dim.setter
    def input_dim(self, value):
        self._input_dim = value

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        self._output_dir = value

    @property
    def tb_callback(self):
        return self._tb_callback

    @tb_callback.setter
    def tb_callback(self, value):
        self._tb_callback = value

    @property
    def train_X(self):
        return self._train_X

    @train_X.setter
    def train_X(self, value):
        self._train_X = value

    @property
    def train_Y(self):
        return self._train_Y

    @train_Y.setter
    def train_Y(self, value):
        self._train_Y = value

    @property
    def test_X(self):
        return self._test_X

    @test_X.setter
    def test_X(self, value):
        self._test_X = value

    @property
    def test_Y(self):
        return self._test_Y

    @test_Y.setter
    def test_Y(self, value):
        self._test_Y = value

    @property
    def model_dir(self):
        return self._model_dir

    @model_dir.setter
    def model_dir(self, value):
        self._model_dir = value

    @property
    def metrics_callback(self):
        return self._metrics_callback

    @metrics_callback.setter
    def metrics_callback(self, value):
        self._metrics_callback = value

    @property
    def predictions(self):
        return self._predictions

    @predictions.setter
    def predictions(self, value):
        self._predictions = value

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, value):
        self._metrics = value

    @property
    def model_json(self):
        return self._model_json

    @model_json.setter
    def model_json(self, value):
        self._model_json = value

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        np.random.seed(self._seed)
        tf.random.set_seed(self._seed)

    @property
    def feature_columns(self):
        return self._feature_columns

    @feature_columns.setter
    def feature_columns(self, value):
        self._feature_columns = value

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    def l1(self):
        return self._l1

    @l1.setter
    def l1(self, value):
        self._l1 = value

    @property
    def l2(self):
        return self._l2

    @l2.setter
    def l2(self, value):
        self._l2 = value

    @property
    def training_history(self):
        return self._training_history

    @training_history.setter
    def training_history(self, value):
        self._training_history = value

    @property
    def best_params_list(self):
        return self._best_params_list

    @best_params_list.setter
    def best_params_list(self, value):
        self._best_params_list = value

    @property
    def multi_class_flag(self):
        return self._multi_class_flag

    @multi_class_flag.setter
    def multi_class_flag(self, value):
        self._multi_class_flag = value

    @property
    def num_classes(self):
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value):
        self._num_classes = value
