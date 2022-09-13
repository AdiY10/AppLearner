"""
***********************************************************************************************************************
    imports
***********************************************************************************************************************
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from sktime.performance_metrics.forecasting import MeanSquaredError
from src.framework__data_set import get_data_set

"""
***********************************************************************************************************************
    plot_result
***********************************************************************************************************************
"""


def plot_result(original, prediction_as_np_array):
    original_as_series = original["sample"].copy()
    predicted_as_series = pd.Series(prediction_as_np_array)
    x_axis = [time for time in original["time"]]
    original_as_series.index = x_axis
    predicted_as_series.index = x_axis[-len(prediction_as_np_array):]
    ax = original_as_series.plot(color="blue", label="Samples")
    predicted_as_series.plot(ax=ax, color="red", label="Predictions")
    plt.legend()
    plt.show()


"""
***********************************************************************************************************************
    Test Bench Class
***********************************************************************************************************************
"""


class TestBench:
    """
    Class that takes some other class of a time series forecasting architecture, and tests it on
    multiple types of data.
    """

    def __init__(
            self,
            class_to_test,
            path_to_data,
            tests_to_perform
    ):
        self.__class_to_test = class_to_test
        self.__path_to_data = path_to_data
        for dictionary in tests_to_perform:
            assert "metric" in dictionary
            assert "app" in dictionary
            assert "prediction length" in dictionary
            assert "sub sample rate" in dictionary
            assert "data length limit" in dictionary
            assert dictionary["data length limit"] > dictionary["prediction length"]
        self.__tests_to_perform = tests_to_perform
        self.__msg = "[TEST BENCH]"
        # mutable variable
        self.length_to_predict = None

    """
    *******************************************************************************************************************
        Data and model related functions
    *******************************************************************************************************************
    """

    def __get_data(self, dictionary):
        """
        @param dictionary: a specified test (keys are the definitions of the tests: the metrics, app name and more)
        @return: train and test datasets
        """
        metric = dictionary["metric"]
        app = dictionary["app"]
        ss_rate = dictionary["sub sample rate"]
        dl_limit = dictionary["data length limit"]
        self.length_to_predict = dictionary["prediction length"]
        dataset = get_data_set(
            metric=metric,
            application_name=app,
            path_to_data=self.__path_to_data
        )
        print(self.__msg, f"Subsampling data from 1 sample per 1 minute to 1 sample per {ss_rate} minutes.")
        dataset.sub_sample_data(sub_sample_rate=ss_rate)
        print(self.__msg, f"Throwing out data that is less than {dl_limit * ss_rate / 60} hours long.")
        dataset.filter_data_that_is_too_short(data_length_limit=dl_limit)
        print(self.__msg, "Scaling data.")
        dataset.scale_data()
        print(self.__msg, "Splitting data into train and test.")
        train, test = dataset.split_to_train_and_test(length_to_predict=self.length_to_predict)
        assert len(train) == len(test)
        assert min([len(df) for df in train] + [len(df) for df in test]) >= (dl_limit - self.length_to_predict)
        print(self.__msg, f"Amount of train/test data is {len(train)}.")
        return train, test

    def __get_model(self, metric, app, train, test):
        length_of_shortest_time_series = min([len(df) for df in train] + [len(df) for df in test])
        model = self.__class_to_test(
            length_of_shortest_time_series=length_of_shortest_time_series,
            metric=metric,
            app=app
        )
        return model

    """
    *******************************************************************************************************************
        Model assessment
    *******************************************************************************************************************
    """

    @staticmethod
    def __calculate_mase(y_pred, y_true, y_train):
        """
        @param y_pred: prediction of our model
        @param y_true: true results
        @return: returns the MASE of the prediction
        """
        mase = MeanAbsoluteScaledError()
        result = mase(y_true=y_true, y_pred=y_pred, y_train=y_train)
        return result

    @staticmethod
    def __calculate_mape(y_true, y_pred):
        mape = MeanAbsolutePercentageError()
        result = mape(y_true=y_true, y_pred=y_pred)
        return result

    @staticmethod
    def __calculate_mse(y_true, y_pred):
        mse = MeanSquaredError()
        result = mse(y_true=y_true, y_pred=y_pred)
        assert result == (np.square(y_true - y_pred)).mean()
        return result

    @staticmethod
    def __get_mse_precision_recall_f1_mase_and_mape(y_true, y_pred, y_train):
        """
        @param y_true: true values
        @param y_pred: prediction values
        @return: the mse, precision_recall_f1 and MASE of the results
        """
        assert len(y_true) == len(y_pred)
        mse_here = TestBench.__calculate_mse(y_pred=y_pred, y_true=y_true)

        actual_positives = [y_true[i + 1] >= y_true[i] for i in range(len(y_true) - 1)]
        predicted_positives = [y_pred[i + 1] >= y_pred[i] for i in range(len(y_pred) - 1)]
        assert len(actual_positives) == len(predicted_positives)
        true_positive = sum([
            1 if (og == predicted and predicted) else 0
            for og, predicted in zip(actual_positives, predicted_positives)
        ])
        false_positive = sum([
            1 if (og != predicted and predicted) else 0
            for og, predicted in zip(actual_positives, predicted_positives)
        ])
        false_negative = sum([
            1 if (og != predicted and not predicted) else 0
            for og, predicted in zip(actual_positives, predicted_positives)
        ])
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive != 0) else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative != 0) else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall != 0) else 0

        mase = TestBench.__calculate_mase(y_pred=y_pred, y_true=y_true, y_train=y_train)
        mape = TestBench.__calculate_mape(y_pred=y_pred, y_true=y_true)

        return mse_here, precision, recall, f1, mase, mape

    def __give_one_test_to_model(self, test_sample, model, should_print):
        """

        @param test_sample:  test sample
        @param model: the model we're training
        @param should_print: true if we want to plot
        @return: mse, precision, recall, f1, mase of the test sample
        """
        assert self.length_to_predict < len(test_sample)
        how_much_to_predict = self.length_to_predict
        how_much_to_give = len(test_sample) - how_much_to_predict
        returned_ts_as_np_array = model.predict(
            ts_as_df_start=test_sample[: how_much_to_give],
            how_much_to_predict=how_much_to_predict
        )
        # make sure the output is in the right format
        assert isinstance(returned_ts_as_np_array, np.ndarray)
        assert len(returned_ts_as_np_array) == how_much_to_predict
        assert returned_ts_as_np_array.shape == (how_much_to_predict,)
        assert returned_ts_as_np_array.dtype == np.float64
        # plot if needed
        if should_print:
            plot_result(
                original=test_sample,
                prediction_as_np_array=returned_ts_as_np_array,
            )
        out_should_be = test_sample["sample"].to_numpy()
        mse_here, precision, recall, f1, mase, mape = self.__get_mse_precision_recall_f1_mase_and_mape(
            y_true=out_should_be[how_much_to_give:], y_pred=returned_ts_as_np_array,
            y_train=out_should_be[:how_much_to_give]
        )
        return mse_here, precision, recall, f1, mase, mape

    def __print_report(self, metric, app, mse, precision, recall, f1, training_time, mase, mape, as_table=False):
        """
        prints the following parameters
        @param metric:
        @param app:
        @param mse:
        @param precision:
        @param recall:
        @param f1:
        @param training_time:
        @param mase:
        @param mape:
        @param as_table: whether to print as a table or not.
        """
        if as_table:
            print(self.__msg,
                  f"| {metric} | {app} | {round(training_time)} seconds   | {round(mse, 5)} | {round(precision, 5)} | {round(recall, 5)} | {round(f1, 5)}  | {round(mase, 5)} | {round(mape, 5)} |")
        else:
            print(self.__msg, f"***********************************************************************")
            print(self.__msg, f"REPORT for                              metric='{metric}', app='{app}':")
            print(self.__msg, f"Training time in seconds is             {training_time}")
            print(self.__msg, f"Average mse over the test set is        {mse}")
            print(self.__msg, f"Average precision over the test set is  {precision}")
            print(self.__msg, f"Average recall over the test set is     {recall}")
            print(self.__msg, f"Average F1 over the test set is         {f1}")
            print(self.__msg, f"Average MASE over the test set is       {mase}")
            print(self.__msg, f"Average MAPE over the test set is       {mape}")
            print(self.__msg, f"***********************************************************************")

    def __test_model(self, test, model):
        """
        predicts according to the samples given in test
        @param test: a list of test samples
        @param model: the model we're training
        @return: mse, precision, recall, f1, training_time, mase of the results
        """
        total_mse = 0
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_mase = 0
        total_mape = 0
        for i, test_sample in enumerate(test):
            mse_here, precision, recall, f1, mase, mape = self.__give_one_test_to_model(
                test_sample=test_sample, model=model, should_print=(i < 10)
            )
            total_mse += mse_here
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_mase += mase
            total_mape += mape
        mse = total_mse / len(test)
        precision = total_precision / len(test)
        recall = total_recall / len(test)
        f1 = total_f1 / len(test)
        mase = total_mase / len(test)
        mape = total_mape / len(test)
        return mse, precision, recall, f1, mase, mape

    def __do_one_test(self, dictionary):
        metric, app = dictionary["metric"], dictionary["app"]
        print(self.__msg, f"Fetching data for metric='{metric}', app='{app}'.")
        train, test = self.__get_data(dictionary=dictionary)
        print(self.__msg, "Making an instance of the class we want to test.")
        model = self.__get_model(metric=metric, app=app, train=train, test=test)
        print(self.__msg, "Starting training loop.")
        training_start_time = time.time()
        model.learn_from_data_set(training_data_set=train)
        training_stop_time = time.time()
        training_time = training_stop_time - training_start_time
        print(self.__msg, f"Training took {training_time} seconds.")
        print(self.__msg, "Starting testing loop")
        mse, precision, recall, f1, mase, mape = self.__test_model(test=test, model=model)
        self.__print_report(
            metric=metric, app=app, mse=mse, precision=precision, recall=recall, f1=f1,
            training_time=training_time, mase=mase, mape=mape
        )
        print(self.__msg, f"Done with metric='{metric}', app='{app}'")
        return mse, precision, recall, f1, training_time, mase, mape

    def print_device_information(self):
        print(self.__msg, "This test was run on:")
        import torch
        if torch.cuda.is_available():
            print(self.__msg, "GPU")
            print(self.__msg, "Device name:", torch.cuda.get_device_name(0))
            os.system("nvidia-smi")
        else:
            print(self.__msg, "CPU")

    def print_table_of_results(self, full_report):
        # plot results
        h = f"| metric | app | training time | mse | precision | recall | F1 | MASE | MAPE |"
        print(self.__msg, h)
        for dictionary, metrics in zip(self.__tests_to_perform, full_report):
            (mse, precision, recall, f1, training_time, mase, mape) = metrics
            app, metric = dictionary["app"], dictionary["metric"]
            self.__print_report(
                metric=metric, app=app, mse=mse, precision=precision, recall=recall, f1=f1,
                training_time=training_time, mase=mase, mape=mape, as_table=True
            )

    """
    *******************************************************************************************************************
        API functions
    *******************************************************************************************************************
    """

    def run_training_and_tests(self):
        print(self.__msg, "Powering on test bench")
        full_report = []
        for dictionary in self.__tests_to_perform:
            app = dictionary["app"]
            metric = dictionary["metric"]
            print(self.__msg, f"testing metric='{metric}', app='{app}'.")
            mse, precision, recall, f1, training_time, mase, mape = self.__do_one_test(dictionary=dictionary)
            full_report += [(mse, precision, recall, f1, training_time, mase, mape)]
        assert len(full_report) == len(self.__tests_to_perform)
        self.print_table_of_results(full_report=full_report)
        # self.print_device_information()
        print(self.__msg, "Powering off test bench")


"""
***********************************************************************************************************************
    main function
***********************************************************************************************************************
"""


class DumbPredictor:
    """
    A dumb predictor is used to test the framework- it takes the first samples and sets them as predictions.
    """

    def __init__(self, length_of_shortest_time_series, metric, app):
        print("Constructor called.")
        self.print_once = True

    def learn_from_data_set(self, training_data_set):
        print("Training started.")
        print("What does a dataframe to learn on look like?")
        print(training_data_set[0])
        print("Training ending.")

    def predict(self, ts_as_df_start, how_much_to_predict):
        # if self.print_once:
        #     self.print_once = False
        #     print("What does a dataframe to predict look like?")
        #     display(ts_as_df_start)
        ts_as_np = ts_as_df_start["sample"].to_numpy()
        res = np.resize(ts_as_np, how_much_to_predict)
        # these checks will also be done by the testbench
        assert isinstance(res, np.ndarray)
        assert len(res) == how_much_to_predict
        assert res.shape == (how_much_to_predict,)
        assert res.dtype == np.float64
        return res


def main(test_to_perform):
    tb = TestBench(
        class_to_test=DumbPredictor,
        path_to_data="../data/",
        tests_to_perform=test_to_perform
    )
    tb.run_training_and_tests()


"""
***********************************************************************************************************************
    run main function
***********************************************************************************************************************
"""

if __name__ == "__main__":
    test_to_perform = (
        # Container CPU
        {"metric": "container_cpu", "app": "kube-rbac-proxy", "prediction length": 16, "sub sample rate": 30,
         "data length limit": 30},
        {"metric": "container_cpu", "app": "dns", "prediction length": 16, "sub sample rate": 30,
         "data length limit": 30},
        {"metric": "container_cpu", "app": "collector", "prediction length": 16, "sub sample rate": 30,
         "data length limit": 30},
        # Container Memory
        {"metric": "container_mem", "app": "nmstate-handler", "prediction length": 16, "sub sample rate": 30,
         "data length limit": 30},
        {"metric": "container_mem", "app": "coredns", "prediction length": 16, "sub sample rate": 30,
         "data length limit": 30},
        {"metric": "container_mem", "app": "keepalived", "prediction length": 16, "sub sample rate": 30,
         "data length limit": 30},
        # Node Memory
        {"metric": "node_mem", "app": "moc/smaug", "prediction length": 16, "sub sample rate": 30,
         "data length limit": 30},
        {"metric": "node_mem", "app": "emea/balrog", "prediction length": 16, "sub sample rate": 30,
         "data length limit": 30}
    )
    main(test_to_perform)
