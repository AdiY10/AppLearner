import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytorch__driver_for_test_bench as pytorch__driver_for_test_bench

"""
creating CNN for time series prediction.
for now, we gonna set the Forecast Horizon to 1, for simplicity.
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size')
parser.add_argument('--save_num', type=int, default=7, metavar='N', help='number on the file to save')
parser.add_argument('--integers', metavar='N', type=int, nargs='+',help='lr decay')
parser.add_argument('--kernel', metavar='N', type=int, nargs='+',help='kernel sizes')
parser.add_argument('--filter_num', type=int, default=16, metavar='N', help='number of filters')
parser.add_argument('--pooling_size', type=int, default=1, metavar='N', help='number of filters')
parser.add_argument('--lr', type=int, default=0.001, metavar='N', help='learning rate')

args = parser.parse_args()
print(args.kernel)

class CNNPredictor(nn.Module):
    def __init__(self, input_size, output_size, length_of_shortest_time_series, pooling_size, kernel_size, num_of_filters):
        super(CNNPredictor, self).__init__()
        self.__length_of_shortest_time_series = length_of_shortest_time_series
        self.pooling_size = pooling_size
        self.kernel_size = kernel_size
        self.num_of_filters = num_of_filters

        # calculate the first fully connected layer size
        fully_connected_features = length_of_shortest_time_series
        for i in range(len(kernel_size)):
            fully_connected_features = np.floor((fully_connected_features-kernel_size[i]+1)/pooling_size)
        fully_connected_features = num_of_filters * int(fully_connected_features)

        # build the model
        for i in range(len(kernel_size)):
            if i == 0:
                self.__seq_model = nn.Sequential(
                    nn.Conv1d(
                        in_channels=input_size,
                        out_channels=num_of_filters,
                        kernel_size=kernel_size[i],
                        stride=1,
                        padding=0
                    ),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=pooling_size),
                )
            else:
                self.__seq_model.add_module(
                    'conv' + str(i),
                    module=nn.Conv1d(
                        in_channels=num_of_filters,
                        out_channels=num_of_filters,
                        kernel_size=kernel_size[i],
                        stride=1,
                        padding=0
                    )

                )
                self.__seq_model.add_module(
                    'relu' + str(i),
                    module=nn.ReLU()
                )
                self.__seq_model.add_module(
                    'pool' + str(i),
                    module=nn.MaxPool1d(kernel_size=pooling_size)
                )

        self.__seq_model.add_module('flatten', nn.Flatten())
        self.__seq_model.add_module('linear1', module=nn.Linear(in_features=fully_connected_features, out_features=20))
        self.__seq_model.add_module('relu_after_linear1', module=nn.ReLU())
        self.__seq_model.add_module('linear2', module=nn.Linear(in_features=20, out_features=output_size))

        # self.__seq_model = nn.Sequential(
        #     nn.Conv1d(in_channels=input_size, out_channels=num_of_filters, kernel_size=kernel_size, stride=1, padding=0),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=pooling_size),
        #     nn.Conv1d(in_channels=num_of_filters, out_channels=num_of_filters, kernel_size=kernel_size, stride=1, padding=0),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=pooling_size),
        #     nn.Flatten(),
            # the next line depends on the length of the minimal time series we need to change the 4
            # the 4 is because length of thr shortest time series is 23 and
            # 23->21->10->8->4 (2 layers of conv1d and 2 layers of pooling operation)
        #



    def forward(self, x):
        # use only the last "length_of_shortest_time_series" values of the time series
        x = x[:, :, -1*self.__length_of_shortest_time_series:]
        out = self.__seq_model(x)
        return out

    def flatten_parameters(self):
        pass


class PytorchCNNTester:
    def __init__(self, length_of_shortest_time_series, metric, app, model_name = "CNN"):
        # prepare parameters
        self.model_name = model_name
        self.__msg = "[PytorchCNNTester]"
        self.__model_input_length = length_of_shortest_time_series // 2
        self.__model = CNNPredictor(
            input_size=1,
            output_size=1,
            length_of_shortest_time_series=self.__model_input_length,
            pooling_size=args.pooling_size,
            kernel_size=args.kernel,
            num_of_filters=args.filter_num
        ).to(pytorch__driver_for_test_bench.get_device())
        # Some Hyper-parameters
        self.__optimizer = optim.Adam(self.__model.parameters(), lr=0.001)
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
            optimizer=self.__optimizer,
            model_name=self.model_name
        )

    def predict(self, ts_as_df_start, how_much_to_predict):
        return pytorch__driver_for_test_bench.predict(
            ts_as_df_start=ts_as_df_start, how_much_to_predict=how_much_to_predict, best_model=self.__best_model,
            model_name="CNN"
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
