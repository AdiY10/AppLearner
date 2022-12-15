import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytorch__driver_for_test_bench as pytorch__driver_for_test_bench

"""
creating CNN for time series prediction.
for now, we gonna set the Forecast Horizon to 1, for simplicity.
"""


class CNNPredictor(nn.Module):
    def __init__(self, input_size, output_size, length_of_shortest_time_series):
        super(CNNPredictor, self).__init__()
        self.__length_of_shortest_time_series = length_of_shortest_time_series
        # seq_length = input_size.shape[2]
        self.__seq_model = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=1, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            # the next line depends on the length of the minimal time series we need to change the 4
            # the 4 is because length of thr shortest time series is 23 and
            # 23->21->10->8->4 (2 layers of conv1d and 2 layers of pooling operation)
            nn.Linear(in_features=4, out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=output_size)
        )

    def forward(self, x):
        print(self.__length_of_shortest_time_series)
        print(x.shape)
        # use only the last "length_of_shortest_time_series" values of the time series
        x = x[:, :, -1*self.__length_of_shortest_time_series:]
        out = self.__seq_model(x)
        return out

    def flatten_parameters(self):
        pass
        # self.__seq_model[0].flatten_parameters()


class PytorchCNNTester:
    def __init__(self, length_of_shortest_time_series, metric, app):
        # prepare parameters
        self.__msg = "[PytorchCNNTester]"
        self.__model_input_length = length_of_shortest_time_series // 2
        self.__model = CNNPredictor(
            input_size=1,
            output_size=1,
            length_of_shortest_time_series=self.__model_input_length
        ).to(pytorch__driver_for_test_bench.get_device())
        # Some Hyper-parameters
        self.__optimizer = optim.Adam(self.__model.parameters(), lr=0.01)
        self.__best_model = self.__model
        self.__criterion = nn.MSELoss()
        # prints
        print(self.__msg, f"model = {self.__model}")
        print(self.__msg, f"optimizer = {self.__optimizer}")
        print(self.__msg, f"criterion = {self.__criterion}")


    def learn_from_data_set(self, training_data_set):
        self.__best_model = pytorch__driver_for_test_bench.train_neural_network(
            training_data_set=training_data_set,
            model=self.__model,
            num_epochs=10,
            model_input_length=self.__model_input_length,
            batch_size=64,
            criterion=self.__criterion,
            optimizer=self.__optimizer
        )

    def predict(self, ts_as_df_start, how_much_to_predict):
        # ignore if CNN ?
        # self.__best_model.flatten_parameters()
        return pytorch__driver_for_test_bench.predict(
            ts_as_df_start=ts_as_df_start, how_much_to_predict=how_much_to_predict, best_model=self.__best_model
        )


"""
***********************************************************************************************************************
    main function
***********************************************************************************************************************
"""


def main(test_to_perform):
    import framework__test_bench as framework__test_bench
    tb = framework__test_bench.TestBench(
        class_to_test=PytorchCNNTester,
        path_to_data="../data/",
        tests_to_perform=test_to_perform
    )
    tb.run_training_and_tests()


if __name__ == "__main__":
    test_to_perform = (
        # Container CPU
        {"metric": "container_cpu", "app": "collector", "prediction length": 5, "sub sample rate": 5,
         "data length limit": 50},
        {"metric": "container_cpu", "app": "dns", "prediction length": 16, "sub sample rate": 30,
         "data length limit": 30}
    )
    main(test_to_perform)
