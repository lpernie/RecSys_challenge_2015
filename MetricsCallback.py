import numpy as np
import os
import time
import logging
import warnings

from Metrics import Metrics
from tensorflow.keras.callbacks import Callback


class MetricsCallback(Callback):
    """
    Metrics callback for Keras model.fit

    :ivar validatation_set: Validation set : validation_set
    :ivar fct: Fraction of false positives : fct
    :ivar model_dir: Model directory : model_dir
    :ivar validation_steps: If > 0, assume validatation_set is a generator: validation_set
    """

    def __init__(self, validation_set, fct, model_dir, verbose=1, validation_steps=0, batch_size=50):

        self.logger = logging.getLogger(self.__class__.__name__)
        super().__init__()
        best_lauc_model_fpath = os.path.join(model_dir, "model_best_lauc_{}_fct.h5".format(fct))
        best_log_loss_model_fpath = os.path.join(model_dir, "model_best_log_loss.h5")

        self.validation_set = validation_set
        self.fct = fct
        self.best_lauc_model_fpath = best_lauc_model_fpath
        self.best_lauc_epoch = 0
        self.best_log_loss_model_fpath = best_log_loss_model_fpath
        self.best_log_loss_epoch = 0
        self.lauc = 0.0
        self.log_loss = np.inf
        self.verbose = verbose
        self.validation_steps = validation_steps
        self.training_history = {}
        self.epoch_time_start = 0
        self.batch_size = batch_size

    def on_train_begin(self, logs=None):
        """
        Performs the following operations before training begins.

        :param logs: Current logs
        :type logs: ``dict``
        """

        if logs is None:
            logs = {}
        self.training_history = {
                                "Epoch": [],
                                "Train Loss": [],
                                "Train Acc": [],
                                "Val Loss": [],
                                "Val Acc": [],
                                "Val LAUC": []
                                }

    def on_epoch_begin(self, batch, logs=None):
        """
        Performs the following operations before training epoch begins.
        :param batch: Batch size
        :type batch: ``int``
        :param logs: Current logs
        :type logs: ``dict``
        """

        if logs is None:
            logs = {}
        self.epoch_time_start = time.time()

    def _compute_y(self):
        if self.validation_steps > 0:
            with warnings.catch_warnings():
                y_true, y_pred = [], []
                warnings.filterwarnings("ignore", category=UserWarning)
                steps = 0
                for batch in self.validation_set:
                    if steps >= self.validation_steps:
                        break

                    y_true.append(batch[1])
                    y_pred.append(self.model.predict_on_batch(batch[0]).numpy())

                    steps += 1

                y_true = np.array(y_true).flatten()
                y_pred = np.array(y_pred).flatten()
        else:
            y_true = self.validation_set[1]
            x = self.validation_set[0]
            y_pred = self.model.predict_proba(x, batch_size=self.batch_size, verbose=0)

        return y_true, y_pred

    def on_epoch_end(self, epoch, logs=None):
        """
        Performs the following operations at the end of each training epoch.

        :param epoch: Current epoch
        :type epoch: ``int``
        :param logs: Current logs
        :type logs: ``dict``
        """
        if logs is None:
            logs = {}
        y_true, y_pred = self._compute_y()

        cur_lauc = Metrics.compute_lauc(y_true, y_pred, self.fct)
        cur_log_loss = logs["val_loss"]

        if cur_lauc > self.lauc:
            self.model.save(self.best_lauc_model_fpath)
            self.lauc = cur_lauc
            self.best_lauc_epoch = epoch

        if cur_log_loss < self.log_loss:
            self.model.save(self.best_log_loss_model_fpath)
            self.log_loss = cur_log_loss
            self.best_log_loss_epoch = epoch

        if self.verbose >= 1:
            self.logger.info("Epoch {}".format(epoch))
            self.logger.info("")
            self.logger.info("{0:10}: {1:.6f}".format("Train Loss", logs["loss"]))
            self.logger.info("{0:10}: {1:.6f}".format("Train Acc", logs["accuracy"]))
            self.logger.info("{0:10}: {1:.6f}".format("Val Loss", logs["val_loss"]))
            self.logger.info("{0:10}: {1:.6f}".format("Val Acc", logs["val_accuracy"]))
            self.logger.info("{0:10}: {1:.6f}".format("Val LAUC", cur_lauc))
            self.logger.info("{0:10}: {1:.6f}".format("Time (sec)", time.time()-self.epoch_time_start))
            self.logger.info("{0:10}: {1:.6f} at Epoch: {2}".format("Best LAUC", self.lauc, self.best_lauc_epoch))
            self.logger.info("{0:10}: {1:.6f} at Epoch: {2}".format("Best LOSS", self.log_loss, self.best_log_loss_epoch))

        self.training_history["Epoch"].append(epoch)
        self.training_history["Train Loss"].append(logs["loss"])
        self.training_history["Train Acc"].append(logs["accuracy"])
        self.training_history["Val Loss"].append(logs["val_loss"])
        self.training_history["Val Acc"].append(logs["val_accuracy"])
        self.training_history["Val LAUC"].append(cur_lauc)
