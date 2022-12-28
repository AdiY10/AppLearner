"""
***********************************************************************************************************************
    imports
***********************************************************************************************************************
"""
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
from torch.utils.data import DataLoader
from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import SMAPE, MultivariateNormalDistributionLoss, NormalDistributionLoss

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(2, 'C:\\Users\\Owner\\AppLearner\\src')
from framework__data_set import get_data_set
# from framework__test_bench_deepar import TestBench
from framework__deepar import DeepARTester
import os
import warnings


warnings.filterwarnings("ignore")

class TestBenchDeepAR:
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
        self.__class_to_test = class_to_test #?????
        self.__path_to_data = path_to_data
        for dictionary in tests_to_perform:
            assert "metric" in dictionary
            assert "app" in dictionary
            assert "prediction length" in dictionary
            assert "sub sample rate" in dictionary
            assert "data length limit" in dictionary
            assert dictionary["data length limit"] > dictionary["prediction length"]
            # as long as we won't be able to work with variable length in our model
            # we'll need to transform our samples to fix length
            # assert "fixed length of samples" in dictionary
        self.__tests_to_perform = tests_to_perform
        self.__msg = "[DeepAR TEST BENCH]"
        # mutable variable
        self.length_to_predict = None

    """
    *******************************************************************************************************************
        Data related functions
    *******************************************************************************************************************
    """
    def __get_data(self, dictionary):
        """
        @param dictionary: a specified test (keys are the definitions of the tests: the metrics, app name and more)
        @return: dataset of the chosen application, after preproccesing before using pytorch forecasting.
        """
        metric = dictionary["metric"]
        app = dictionary["app"]
        ss_rate = dictionary["sub sample rate"]
        dl_limit = dictionary["data length limit"]
        self.length_to_predict = dictionary["prediction length"]
        fix_len = dictionary["fixed length of samples"]
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
        print(self.__msg, f"Transforming the data to be fix length.") ###################### for now
        dataset = self.__preprocces(dataset,fix_len)
        print(self.__msg, "Generating DataFrame from data")
        dataset.rename(columns = {'sample':'value'}, inplace = True)
        dataset = dataset.astype(dict(series=str))
        #

        # print(self.__msg, "Splitting data into train and test.")
        # train, test = dataset.split_to_train_and_test(length_to_predict=self.length_to_predict)
        # assert len(train) == len(test)
        # assert min([len(df) for df in train] + [len(df) for df in test]) >= (dl_limit - self.length_to_predict)
        # print(self.__msg, f"Amount of train/test data is {len(train)}.")
        # return train, test
        return dataset

    def __get_model(self,training):
        model = self.__class_to_test(training = training
        )
        return model        

    # def __get_model(self, metric, app, train, test):
    #     length_of_shortest_time_series = min([len(df) for df in train] + [len(df) for df in test])
    #     model = self.__class_to_test(

    #         metric=metric,
    #         app=app
    #     )
    #     return model    
    
    def __to_dataloaders(self,data,max_encoder_length,max_prediction_length):
        # max_encoder_length = max_encoder_length
        # max_prediction_length = max_prediction_length

        validation_cutoff = data["time_idx"].max() - max_prediction_length
        training_cutoff = data["time_idx"].max() - max_prediction_length*2


        context_length = max_encoder_length
        prediction_length = max_prediction_length

        training = TimeSeriesDataSet(
            data[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="value",
            group_ids=["series"],
            static_categoricals=[
                "series"
            ],  # as we plan to forecast correlations, it is important to use series characteristics (e.g. a series identifier)
            time_varying_unknown_reals=["value"],
            max_encoder_length=max_encoder_length, 
            max_prediction_length=max_prediction_length,
        )       
        validation = TimeSeriesDataSet.from_dataset(training, data[lambda x: x.time_idx<=validation_cutoff], min_prediction_idx=training_cutoff + 1)
        test = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=validation_cutoff + 1)
        batch_size = 128
        # synchronize samples in each batch over time - only necessary for DeepVAR, not for DeepAR
        # train_dataloader = DataLoader(training,batch_size=batch_size, batch_sampler=None,collate_fn=PadSequence())
        # validation_dataloader = DataLoader(validation,batch_size=batch_size, batch_sampler=None,collate_fn=PadSequence())
        train_dataloader = training.to_dataloader(
            train=True, batch_size=batch_size, num_workers=0, batch_sampler= None, #?
        )
        val_dataloader = validation.to_dataloader(
            train=False, batch_size=batch_size, num_workers=0, batch_sampler= None #?
        )
        test_dataloader = test.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0, batch_sampler="synchronized" #?
        )   
        return train_dataloader ,val_dataloader, test_dataloader ,training ,validation, test
    
    def __preprocces(self, df, fix_len):
        x = []
        for i in range(len(df)):
            x.append(df[i])
            x[i]["time_idx"] = range(x[i].shape[0])
        dataset = [x[i].iloc[:fix_len,:] for i in range(len(x)) if max(x[i]["time_idx"])>fix_len]
        for i in range(len(dataset)):
             dataset[i]["series"] = str(i)
        print(i)
        return pd.concat(dataset,ignore_index=True)


    @staticmethod
    def __calculate_mase(actuals, predictions):
        """
        @param predictions: prediction of our model
        @param actuals: true results
        @return: returns the MASE of the prediction
        """
        # Calculate the MAE of the forecast
        mae = torch.abs(predictions - actuals).float().mean()
        # Calculate the MAE of the naive forecast (forecasted values are just the previous actual value)
        tsr = actuals[:, :-1]
        zero = torch.zeros((tsr.shape[0], 1))
        naive = torch.cat([zero, tsr],dim=1)

        mae_naive = torch.abs(naive - actuals).float().mean()
        # Calculate the MASE
        mase = mae / mae_naive
        return mase

    @staticmethod
    def __calculate_mape(actuals, predictions):
        # Calculate the absolute percentage error for each element
        error = torch.abs((actuals - predictions) / actuals)
        # Calculate the mean absolute percentage error
        mape = error.mean() 
        return mape
    
    @staticmethod
    def __calculate_mse(actuals, predictions):
        # Calculate the squared error for each element
        error = (actuals - predictions) ** 2
        # Calculate the mean squared error
        mse = error.mean()
        return mse

    @staticmethod
    def __get_mse_precision_recall_f1_mase_and_mape(actuals, predictions):
        """
        @param actuals: true values
        @param predictions: prediction values
        @return: the mse, precision_recall_f1 and MASE of the results
        """
        
        assert len(actuals) == len(actuals)
        mse_here = TestBenchDeepAR.__calculate_mse(predictions=predictions, actuals=actuals)

        # Calculate the actual and predicted positives using element-wise comparison
        actual_positives = actuals[:-1] >= actuals[1:]
        predicted_positives = predictions[:-1] >= predictions[1:]

        # Calculate the number of true positives
        true_positive = (actual_positives == predicted_positives).sum()

        # Calculate the number of false positives
        false_positive = ((actual_positives != predicted_positives) & predicted_positives).sum()

        # Calculate the number of false negatives
        false_negative = ((actual_positives != predicted_positives) & ~predicted_positives).sum()
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive != 0) else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative != 0) else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall != 0) else 0

        mase = TestBenchDeepAR.__calculate_mase(predictions=predictions, actuals=actuals)
        mape = TestBenchDeepAR.__calculate_mape(predictions=predictions, actuals=actuals)

        return mse_here, precision, recall, f1, mase, mape

    def __print_report(self, metric, app, mse, precision, recall, f1,training_time, mase, mape, as_table=False):
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
            print(self.__msg, f"MSE over the test set is        {mse}")
            print(self.__msg, f"Precision over the test set is  {precision}")
            print(self.__msg, f"Recall over the test set is     {recall}")
            print(self.__msg, f"F1 over the test set is         {f1}")
            print(self.__msg, f"MASE over the test set is       {mase}")
            print(self.__msg, f"MAPE over the test set is       {mape}")
            print(self.__msg, f"***********************************************************************")
###########################################################################################################


    def __do_one_test(self, dictionary):
        metric, app, max_prediction_length, plot  = dictionary["metric"], dictionary["app"], dictionary["prediction length"], dictionary["plot"]
        max_encoder_length = max_prediction_length*3 #TODO
        print(self.__msg, f"Fetching data for metric='{metric}', app='{app}'.")
        try:
            dataset = pd.read_pickle("dataset_{}.pkl".format(app))
        except:
            dataset = self.__get_data(dictionary=dictionary)
            dataset.to_pickle(".\\dataset_{}.pkl".format(app))
        train_dataloder, val_dataloader ,test_dataloader, training ,validation, test = self.__to_dataloaders(dataset,max_encoder_length,max_prediction_length) #TODO insert as param
        print(self.__msg, "Making an instance of the class we want to test.")
        model = self.__get_model(training)
        print(self.__msg, "Starting training loop.")
        training_start_time = time.time()
        model.learn_from_data_set(train_dataloder, val_dataloader)
        training_stop_time = time.time()
        training_time = training_stop_time - training_start_time
        print(self.__msg, f"Training took {training_time} seconds.")
        print(self.__msg, "Starting testing loop")
        raw_predictions, x = model.predictions(train_dataloder, test_dataloader)
        predictions = raw_predictions[0].mean(dim=2)
        
        actuals, predictions = model.get_actuals(test_dataloader)
        mse, precision, recall, f1, mase, mape = self.__get_mse_precision_recall_f1_mase_and_mape(actuals, predictions)
        if(plot):
            model.plot_predictions(raw_predictions, x, test)
        self.__print_report(
            metric=metric, app=app, mse=mse, precision=precision, recall=recall, f1=f1,
            training_time=training_time, mase=mase, mape=mape
        )
        print(self.__msg, f"Done with metric='{metric}', app='{app}'")
        return mse, precision, recall, f1, training_time, mase, mape   

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
        # self.print_table_of_results(full_report=full_report)
        # self.print_device_information()
        print(self.__msg, "Powering off test bench")

    """
    *******************************************************************************************************************
        main function
    *******************************************************************************************************************
    """
def main(test_to_perform):
    tb = TestBenchDeepAR(
        class_to_test=DeepARTester,
        path_to_data="C:\\Users\\Owner\\AppLearner\\data\\OperatreFirst_PrometheusData_AppLearner\\",
        tests_to_perform=test_to_perform
    )
    tb.run_training_and_tests()


"""
***********************************************************************************************************************
    run main function
***********************************************************************************************************************
"""

if __name__ == "__main__":
    test_to_perform = [
        # Container CPU
        # {"metric": "container_cpu", "app": "kube-rbac-proxy", "prediction length": 20, "sub sample rate": 5,
        #  "data length limit": 30, "fixed length of samples": 100, "plot": True}
        # {"metric": "container_cpu", "app": "dns", "prediction length": 20, "sub sample rate": 30,
        #  "data length limit": 30},
        # {"metric": "container_cpu", "app": "collector", "prediction length": 20, "sub sample rate": 30,
        #  "data length limit": 30},
        # # Container Memory
        # {"metric": "container_mem", "app": "nmstate-handler", "prediction length": 20, "sub sample rate": 30,
        #  "data length limit": 30},
        # {"metric": "container_mem", "app": "coredns", "prediction length": 20, "sub sample rate": 30,
        #  "data length limit": 30},
        # {"metric": "container_mem", "app": "keepalived", "prediction length": 20, "sub sample rate": 30,
        #  "data length limit": 30},
        # # Node Memory
        # {"metric": "node_mem", "app": "moc/smaug", "prediction length": 20, "sub sample rate": 30,
        #  "data length limit": 30},
        # {"metric": "node_mem", "app": "emea/balrog", "prediction length": 20, "sub sample rate": 30,
        #  "data length limit": 30}
    ]
    main(test_to_perform)
