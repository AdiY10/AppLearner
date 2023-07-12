# imports
import sys
import os
# setting path
sys.path.append(os.path.abspath('..'))
import framework__data_set as ds
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from darts import TimeSeries
from darts import concatenate
from darts.models import TCNModel
from darts.utils.likelihood_models import GaussianLikelihood
from darts.metrics import mae, rmse
from darts.dataprocessing.transformers import Scaler
from darts.utils.missing_values import extract_subseries


def main():
    # get data
    cpu_dataset = ds.get_data_set(
        metric="container_cpu",
        application_name="collector",
        path_to_data="../data/OperatreFirst_PrometheusData_AppLearner/"
    )

    mem_dataset = ds.get_data_set(
        metric="container_mem",
        application_name="collector",
        path_to_data="../data/OperatreFirst_PrometheusData_AppLearner/"
    )
    # sort data by time and build merged dataframe
    cpu_dataset.sort_by_time()
    mem_dataset.sort_by_time()
    # scale data to range (-1,1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    transformer = Scaler(scaler)
    # join dataframes
    merged_df = pd.merge(cpu_dataset.get_marged(), mem_dataset.get_marged(), on='time', how='inner')
    cpu_merge_series = TimeSeries.from_dataframe(merged_df, time_col='time', value_cols='sample_x',
                                                 fill_missing_dates=True)
    mem_merge_series = TimeSeries.from_dataframe(merged_df, time_col='time', value_cols='sample_y',
                                                 fill_missing_dates=True)
    # scale data
    cpu_merge_series = transformer.fit_transform(cpu_merge_series)
    mem_merge_series = transformer.fit_transform(mem_merge_series)

    # extract subseries from data
    cpu_series_lst = extract_subseries(cpu_merge_series, min_gap_size=1, mode='all')
    mem_series_lst = extract_subseries(mem_merge_series, min_gap_size=1, mode='all')

    # concatenate cpu and mem series
    multivariate_series_lst = []
    for cpu_series, mem_series in zip(cpu_series_lst, mem_series_lst):
        if min(len(cpu_series), len(mem_series)) <= 122:
            continue
        multivariate_series = concatenate([cpu_series, mem_series], axis=1)
        multivariate_series_lst.append(multivariate_series)

    # get intervals for train and test data
    number = int("12") % int(len(multivariate_series_lst) * 0.2)
    test_series_lst, train_series_lst = multivariate_series_lst[
                                        int(len(multivariate_series_lst) * 0.8) + number:], multivariate_series_lst[
                                                                                            :int(
                                                                                                len(multivariate_series_lst) * 0.8) + number]

    # The output length must be strictly smaller than the input length
    model = TCNModel(
        dropout=0.26,
        batch_size=16,
        n_epochs=500,
        num_filters=5,
        num_layers=3,
        optimizer_kwargs={"lr": 0.0003},
        random_state=0,
        input_chunk_length=61,
        output_chunk_length=60,
        kernel_size=5,
        weight_norm=True,
        dilation_base=4,
        likelihood=GaussianLikelihood(),
        pl_trainer_kwargs={"accelerator": "gpu", "devices": 8}
    )
    # fit model
    model.fit(series=train_series_lst, past_covariates=None, verbose=True)
    # save model
    model.save("Multivariate_TCN_model_after_HP_tuning.pt")
    # evaluate model
    model.to_cpu()
    backtest_en = model.historical_forecasts(
        series=test_series_lst,
        past_covariates=None,
        forecast_horizon=60,
        stride=3,
        retrain=False,
        verbose=False,
    )
    for orig, backtest in zip(test_series_lst, backtest_en):
        if len(orig) < 500:
            continue
        maes = mae(orig, backtest, n_jobs=1, verbose=False)
        rmses = rmse(orig, backtest, n_jobs=1, verbose=False)
        print("Number of samples:", len(orig))
        print("MAE:", maes)
        print("RMSE:", rmses)
        plt.figure(figsize=(10, 6))
        orig_untransformed = transformer.inverse_transform(orig)
        backtest_untransformed = transformer.inverse_transform(backtest)
        # Apply the filter to the time series
        orig_untransformed.plot(label="actual")
        backtest_untransformed.plot(label="backtest", linewidth=1, color='red')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    # freeze_support() so we could train with multi gpu
    torch.multiprocessing.freeze_support()
    main()
