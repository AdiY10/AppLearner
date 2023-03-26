"""
***********************************************************************************************************************
    imports
***********************************************************************************************************************
"""

import our_src.training_utils as training_utils
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from SCINet.models.SCINet import SCINet
"""
***********************************************************************************************************************
    LSTMPredictor class
***********************************************************************************************************************
"""
        
"""
***********************************************************************************************************************
    Testable class
***********************************************************************************************************************
"""

class PytorchSCITester:
    def __init__(self, length_of_shortest_time_series, metric, app):
        # prepare parameters
        self.__msg = "[PytorchLSTMTester]"
        self.__model_input_length = 40
        self.__model_output_length = 20
        assert self.__model_output_length + self.__model_input_length <= length_of_shortest_time_series
        self.__model = SCINet(
                output_len= self.__model_output_length,
                input_len= self.__model_input_length,
                input_dim=1,
                hid_size = 256,
                num_stacks= 1,
                num_levels= 2,
                num_decoder_layer= 1,
                concat_len = 0,
                groups = 1,
                kernel = 5,
                dropout = 0.5,
                single_step_output_One = 0,
                positionalE = True,
                modified = True,
                RIN=False
        ).to(training_utils.device)
        
        self.__optimizer = optim.Adam(self.__model.parameters(), lr=0.0005)
        self.__scheduler = optim.lr_scheduler.ExponentialLR(self.__optimizer, gamma=0.9)
        self.__best_model = self.__model
        self.__criterion = nn.MSELoss()
        # print
        print(self.__msg, f"model = {self.__model}")
        print(self.__msg, f"optimizer = {self.__optimizer}")
        print(self.__msg, f"optimizer = {self.__scheduler}")
        print(self.__msg, f"criterion = {self.__criterion}")

    """
    *******************************************************************************************************************
        API functions
    *******************************************************************************************************************
    """

    def learn_from_data_set(self, training_data_set):
        best_model =training_utils.train(
        training_data_set=training_data_set,
        model=self.__model,
        num_epochs=15,
        model_input_length=self.__model_input_length,
        model_output_length=self.__model_output_length,
        batch_size=32,
        criterion=self.__criterion,
        optimizer=self.__optimizer
        )
        return best_model


    def predict(self, ts_as_df_start, how_much_to_predict):
        x =ts_as_df_start['sample'].to_numpy()
        with torch.no_grad():
            x = torch.from_numpy(x).to(torch.float32).to(training_utils.device).unsqueeze(0).unsqueeze(-1)
            x = x[:,-self.__model_input_length:,:]
            y = self.__model(x)
            y = y.squeeze(0).squeeze(-1).numpy()
            y.astype(np.float64)
        return y


"""
***********************************************************************************************************************
    main function
***********************************************************************************************************************
"""


def main(test_to_perform):
    import src.framework__test_bench as framework__test_bench
    tb = framework__test_bench.TestBench(
        class_to_test=PytorchLSTMTester,
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
        {"metric": "container_cpu", "app": "kube-rbac-proxy", "prediction length": 20, "sub sample rate": 30,
         "data length limit": 84},
        # {"metric": "container_cpu", "app": "dns", "prediction length": 16, "sub sample rate": 30,
         # "data length limit": 30}
        # {"metric": "container_cpu", "app": "collector", "prediction length": 16, "sub sample rate": 30,
        #  "data length limit": 30},
        # # Container Memory
        # {"metric": "container_mem", "app": "nmstate-handler", "prediction length": 16, "sub sample rate": 30,
        #  "data length limit": 30},
        # {"metric": "container_mem", "app": "coredns", "prediction length": 16, "sub sample rate": 30,
        #  "data length limit": 30},
        # {"metric": "container_mem", "app": "keepalived", "prediction length": 16, "sub sample rate": 30,
        #  "data length limit": 30},
        # # Node Memory
        # {"metric": "node_mem", "app": "moc/smaug", "prediction length": 16, "sub sample rate": 30,
        #  "data length limit": 30},
        # {"metric": "node_mem", "app": "emea/balrog", "prediction length": 16, "sub sample rate": 30,
        #  "data length limit": 30}
    )
    main(test_to_perform)
