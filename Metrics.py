"""
=======================================================
Class containing metrics utilities
=======================================================
"""

import sys
import os
import csv
import copy
import logging
import numpy as np
from collections import Counter, OrderedDict

class Metrics:
    """
    Class containing utilities to compute metrics utilities
    """
    def __init__(self):
        self.logger = logging.getLogger("Metrics")

        self.perf_metrics = None
        self.formatted_perf_metrics = None

    @staticmethod
    def compute_lauc(y_true, y_pred, max_fp_fct):
        """
        Compute Left Area Under the ROC Curve (LAUC).

        :param y_true: True binary labels or binary label indicators.
        :type y_true: ``numpy.ndarray``
        :param y_pred: Prediction scores, typically probability estimates of the positive class.
        :type y_pred: ``numpy.ndarray``
        :param max_fp_fct: maximum fraction of false positive samples.
        :type max_fp_fct: ``list`` or ``float``
        :returns A: LAUC
        :rtype: ``float``
        """

        if not isinstance(max_fp_fct, list):
            max_fp_fct = [max_fp_fct]
        num_entries = len(max_fp_fct)
        # ensure that the lower bound is slightly larger than 0 and upper bound is 1.0
        max_fp_fct = np.clip(max_fp_fct, sys.float_info.epsilon, 1.0)
        # Get counts of all elements in the list
        counts = Counter(max_fp_fct)
        ordered_counts = sorted(counts.items())
        # Ensure entries are unique and sorted in increasing order.
        # In order to ensure that the final number of LAUC values corresponds
        # to the input number of FPR values, we will map calculated LAUC values
        # before returning the list of LAUC values at the end
        max_fp_fct = sorted(list(set(max_fp_fct)))

        # fp refers to false positive count
        # fpprev refers to fp for previous sample
        # tp refers to true positive count
        # tpprev refers to tp of previous sample
        # A refers to Area Under the Curve
        # datprev refers to prediction for previous sample

        fp = 0.0
        fpprev = 0.0
        tp = 0.0
        tpprev = 0.0
        A = 0.0
        A_values = []
        laucs = []
        datprev = -np.inf

        # Reshape y_true and y_pred
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()

        # Reshape y_true and y_pred
        np.random.seed(0)
        random_numbers = np.random.uniform(10 ** (-9), 2 * 10 ** (-9), size=y_pred.shape[0])
        y_pred = y_pred + random_numbers

        # Compute n0 and n1
        n0 = 0.0
        n1 = 0.0
        for i in range(y_true.shape[0]):
            if y_true[i] > 0:
                n1 += 1.0
            else:
                n0 += 1.0
        max_fps = [i * n0 for i in max_fp_fct]

        # Sort y_pred in descending order
        index = np.argsort(y_pred)[::-1]

        # Use trapezoidal rule to compute area under the curve
        for i in range(y_true.shape[0]):
            if y_pred[index[i]] != datprev:
                additional_area = (fp - fpprev) * (tp + tpprev) * 0.5
                if fp <= max_fps[0]:
                    A += additional_area
                else:
                    delta = additional_area * (max_fps[0] - fpprev) / (fp - fpprev)
                    # We go up to max_fp_fct only in false positive
                    # so only add a fraction of the additional area.
                    A_values.append(A+delta)
                    if len(max_fps) == 1:
                        break
                    else:
                        # Update A since we are continuing to calculate values
                        # for the next fpr
                        A += additional_area
                        # remove the fpr for which fp > fpr
                        max_fps.pop(0)
                datprev = y_pred[index[i]]
                fpprev = fp
                tpprev = tp

            if y_true[index[i]] > 0:
                tp += 1.0
            else:
                fp += 1.0

        if i == len(y_true)-1:
            # At this point we may have one or more entries left in max_fps array
            # Force update A at the end
            # A += (n0 - fpprev) * (n1 + tpprev) * 0.5 * (max_fp - fpprev) / (n0 - fpprev)

            for max_fp in max_fps:
                A_values.append(A + (max_fp - fpprev) * (n1 + tpprev) * 0.5)

        for A, fpr in zip(A_values, max_fp_fct):
            A /= (n0 * n1 * fpr)
            laucs.append(A)

        if num_entries == 1:
            return laucs[0]
        else:
            mapped_lauc_values = []
            for i, elem in enumerate(ordered_counts):
                mapped_lauc_values.extend([laucs[i]]*elem[1])
            return mapped_lauc_values

    def compute_perf_metrics(self, y_true=None, y_pred=None, mode=1, score_thresholds=None, input_metrics=None):
        """
        Compute perf metrics for the list of score thresholds

        :param y_true: True binary labels or binary label indicators.
        :type y_true: ``numpy.ndarray``
        :param y_pred: Prediction scores, typically probability estimates of the positive class.
        :type y_pred: ``numpy.ndarray``
        :param mode: Flag indicating the score range:

           ::

            - 0: [0,999]
            - 1: [0,1]
            - 2: [-1,1]

        :type mode: ``int``
        :param score_thresholds: List of score thresholds used for calculation
        :type score_thresholds: ``list``
        :param input_metrics: Type of the metrics that will be calculated. If none specified, use all available
        :type input_metrics: ``list``
        :returns: Perf metrics values
        :rtype: ``OrderedDict``
        """
        if score_thresholds is None:
            score_thresholds = []
        if input_metrics is None:
            input_metrics = []
        allowed_metrics = ['tp', 'tn', 'fn', 'fp', 'fpr', 'tpr', 'rr', 'accuracy', 'precision', 'f1', 'lauc']
        if len(input_metrics) == 0:
            input_metrics = allowed_metrics
        else:
            if 'fpr' not in input_metrics or 'tpr' not in input_metrics:
                self.logger.error('fpr and tpr are mandatory metrics')
                return self
            for elem in input_metrics:
                if elem not in allowed_metrics:
                    self.logger.error('{} is not yet supported!'.format(elem))
                    self.logger.info('Supported metrics: tp,tn,fn,fp,fpr,tpr,rr,accuracy,precision,f1,lauc')
                    return self

        if self._check_scores(y_pred, mode=mode):
            # If score_thresholds are not provided, use bin width=10
            # for mode =0 and bin width 0.01 for mode 1 and 2
            if score_thresholds is None or len(score_thresholds) == 0:
                if mode == 0:
                    score_thresholds = np.arange(0, 1010, 10)
                elif mode == 1:
                    score_thresholds = np.linspace(0, 1, 101, endpoint=True)
                else:
                    score_thresholds = np.linspace(-1, 1, 101, endpoint=True)
        else:
            return self

        all_keys = ['score']
        all_keys.extend(allowed_metrics)
        tp, tn, fn, fp, fpr, tpr, rr, accuracy, precision, f1, lauc = ([None]*len(score_thresholds) for i in range(len(allowed_metrics)))
        # Calculate tp, tn, fn, fp since they are needed in all calculations
        total_target_1 = y_true.sum(dtype=np.int64)
        total_target_0 = len(y_true) - total_target_1

        desc_score_indices = np.argsort(y_pred)
        sorted_predictions = y_pred[desc_score_indices]
        sorted_labels = y_true[desc_score_indices]

        for i, score_threshold in enumerate(score_thresholds):
            # ind_geq_threshold represents indices of the array elements where element
            # is greater or equal than the threshold
            # sorted_predictions = [0.    0.107 0.158 0.264 0.294 0.327 0.384 0.486 0.514 0.762]
            # sorted_labels = [0., 0., 0., 1., 1., 0., 0., 0., 1., 1.]
            # score_threshold=0.4
            # ind_geq_threshold = [7,8,9]
            ind_geq_threshold = np.where(sorted_predictions >= score_threshold)[0]

            # ind_lt_threshold represents indices of the array elements where element
            # has values lower than the threshold
            ind_lt_threshold = np.setdiff1d(desc_score_indices, ind_geq_threshold)

            # True Positive count represents the count
            # of y_pred=1 at scores greater or equal to the threshold.
            # From the example above: sorted_labels[ind_geq_threshold] = [0,1,1]
            tp[i] = sorted_labels[ind_geq_threshold].sum(dtype=np.int64)

            # False Negative count represents number of y_pred=1 at score lower than threshold
            fn[i] = sorted_labels[ind_lt_threshold].sum(dtype=np.int64)

            # False Positive count represents number of y_pred=0 at scores greater or equal to the threshold
            # Therefore, in order to avoid calculating another sum, we can use the fact that
            # we already know number of ones for scores greater than or equal to the threshold (tp)
            fp[i] = len(ind_geq_threshold)-tp[i]

            # True Negative count represents the number of y_pred=0 at scores lower
            # than the threshold. Using logic similar to the one used in fp:
            tn[i] = len(ind_lt_threshold)-fn[i]

            fpr[i] = fp[i]/(fp[i]+tn[i])
            tpr[i] = tp[i]/(tp[i]+fn[i])

            if 'accuracy' in input_metrics:
                accuracy[i] = (tp[i]+tn[i])/(tp[i]+tn[i]+fp[i]+fn[i])

            if 'precision' in input_metrics:
                # If tp+fp=0, precision is not calculated
                # and remains None
                if (tp[i]+fp[i]).sum() != 0:
                    precision[i] = tp[i]/(tp[i]+fp[i])

            if 'f1' in input_metrics:
                f1[i] = 2*tp[i]/(2*tp[i]+fp[i]+fn[i])

            if 'rr' in input_metrics:
                # ind_gt_threshold: indices of elements greater than threshold
                ind_gt_threshold = np.where(sorted_predictions > score_threshold)[0]
                target_1_gt_threshold = sorted_labels[ind_gt_threshold].sum(dtype=np.int64)
                target_0_gt_threshold = len(ind_gt_threshold)-target_1_gt_threshold
                rr[i] = (target_1_gt_threshold + target_0_gt_threshold)/(total_target_1+total_target_0)

            if 'lauc' in input_metrics and i == len(score_thresholds)-1:
                lauc = Metrics.compute_lauc(y_true, y_pred, fpr)
                lauc = lauc[::-1]

        perf_metrics = OrderedDict(zip(all_keys, [score_thresholds, tp, tn, fn, fp, fpr, tpr, rr, accuracy, precision, f1, lauc]))
        keys_to_remove = [key for key in allowed_metrics if key not in input_metrics]
        for key in keys_to_remove:
            perf_metrics.pop(key, None)
        self.perf_metrics = perf_metrics

    def save_perf_metrics(self, output_dir="./", delimiter=',', file_name="perf_metrics.csv"):
        """
        Saves the results of compute_perf_metrics

        :param output_dir: Directory where the results will be saved
        :type output_dir: ``str``
        :param delimiter: Delimiter for the output file
        :type delimiter: ``str``
        :param file_name: Name of the file where performance metrics will be saved
        :type file_name: ``str``
        :returns:
        :rtype:
        """
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        if self.perf_metrics is None:
            self.logger.error("You need to generate perf metrics before calling save_perf_metrics")
            return self

        self._format_values()

        out_file = os.path.join(output_dir, file_name)

        keys, values = [], []
        for key, value in self.formatted_perf_metrics.items():
            keys.append(key)
            values.append(value)

        with open(out_file, "w") as f:
            wr = csv.writer(f, delimiter=delimiter)
            wr.writerow(list(self.formatted_perf_metrics))
            wr.writerows(zip(*self.formatted_perf_metrics.values()))

        self.logger.info('Perf metrics written to {}'.format(out_file))
        return self

    def _format_values(self, keys_to_exclude=None):
        """
        Format all floats

        :param keys_to_exclude: List of indices to exclude from formatting. Default [0] (score threshold)
        :type keys_to_exclude: ``list``
        """
        if keys_to_exclude is None:
            keys_to_exclude = ['tp', 'tn', 'fn', 'fp']
        formatted_values = copy.deepcopy(self.perf_metrics)

        for key, value in formatted_values.items():
            if key in keys_to_exclude:
                continue
            values_to_format = formatted_values[key]
            for i, elem in enumerate(values_to_format):
                if isinstance(elem, int) or isinstance(elem, np.int64):
                    values_to_format[i] = elem
                elif isinstance(elem, float) or isinstance(elem, np.float64):
                    if key == 'score':
                        values_to_format[i] = format(elem, '.3f').rstrip('0')
                    else:
                        values_to_format[i] = format(elem, '.6f').rstrip('0')

            formatted_values[key] = values_to_format

        self.formatted_perf_metrics = formatted_values
        return self

    def _check_scores(self, y_pred=None, mode=1):
        """
        Check validity of score ranges
        """
        if y_pred is None:
            y_pred = []
        if y_pred is None or len(y_pred) == 0:
            self.logger.error("You must provide np.array of scores as input")
        min_prediction = y_pred.min()
        max_prediction = y_pred.max()

        if mode == 0:
            if max_prediction > 999 or min_prediction < 0.0:
                error_message = "Found scores outside of [0, 999]"
                self.logger.error(error_message)
                return False
        elif mode == 1:
            if max_prediction > 1 or min_prediction < 0:
                error_message = "Found scores outside of [0, 1]"
                self.logger.error(error_message)
                return False
        elif mode == 2:
            if max_prediction > 1 or min_prediction < -1:
                error_message = "Found scores outside of [-1, 1]"
                self.logger.error(error_message)
                return False
        else:
            raise Exception("Unknown mode option")

        return True
