import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from src.SCINet import SCINet

import time
import copy
import random
import math
import matplotlib.pyplot as plt





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(training_data_set, model, num_epochs, model_input_length, model_output_length, batch_size, optimizer, criterion,
                         ):
    list_of_batch = __prepare_batches(
        training_data_set=training_data_set,
        model_input_length=model_input_length,
        model_output_length=model_output_length,
        batch_size=batch_size
    )
    epoch_time = 0
    min_sum_of_losses = float('inf')
    best_model = copy.deepcopy(model)
    for e in range(num_epochs):
        epoch_start_time = time.time()
        sum_of_losses = 0
        print_one=True
        for batch_data in list_of_batch:
            train_input, train_target = batch_data
            
            optimizer.zero_grad()
            out = model.forward(x=train_input.clone())
            if print_one:
                out_d = out.detach()
                print_one=False
                i=0
                plt.plot(range(model_input_length+model_output_length), list(train_input[i,:,0].cpu())+list(train_target[i,:].cpu()))
                plt.plot(range(model_input_length,model_input_length+model_output_length), list(out_d[i,:,0].cpu()))
                plt.figure(figsize=(2, 2))
                plt.show()
            # loss = criterion(out, train_target.unsqueeze(-1)) + MAPELoss(out, train_target.unsqueeze(-1))
            loss = criterion(out, train_target.unsqueeze(-1))
            # loss = MAPELoss(out, train_target.unsqueeze(-1))
            # loss_array[true_if_pad] = 0
            # loss = loss_array.sum() / false_if_pad.sum()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            sum_of_losses += loss

        if sum_of_losses < min_sum_of_losses:
            min_sum_of_losses = sum_of_losses
            best_model = copy.deepcopy(model)
            assert not (best_model is model)  # assert different objects
        epoch_stop_time = time.time()
        epoch_time = epoch_stop_time - epoch_start_time
        avg_loss = sum_of_losses / len(list_of_batch)
        print(f"Epoch {e + 1} done. Epoch time was {epoch_time}. Average loss for the batches in epoch is {avg_loss}")
    return best_model



def __prepare_batches(training_data_set, model_input_length, model_output_length, batch_size):
    list_of_np_array = [
        ts_as_df["sample"].to_numpy()
        for ts_as_df in training_data_set
    ]
    list_of_input_output_np_array = [
        (arr[i: model_input_length + i], arr[model_input_length + i: model_input_length + i + model_output_length])
        for arr in list_of_np_array
        for i in range(len(arr) - model_input_length - model_output_length)
    ]
    print(f"number of training samples = {len(list_of_input_output_np_array)}")
    list_of_input_output_np_array_batched = __partition_list_to_batches(
        list_of_something=list_of_input_output_np_array, batch_size=batch_size
    )
    combined = __combine_batches_of_np_array(batches=list_of_input_output_np_array_batched)
    return combined

def __partition_list_to_batches(list_of_something, batch_size):
    random.shuffle(list_of_something)
    num_batches = math.ceil(len(list_of_something) / batch_size)
    result = [
        list_of_something[i * batch_size: (i + 1) * batch_size]
        for i in range(num_batches)
    ]
    return result


def __combine_batches_of_np_array(batches):
    combined_batches = []
    for batch in batches:
        batch_in_as_list_of_np_array = [tup[0] for tup in batch]
        batch_out_as_list_of_np_array = [tup[1] for tup in batch]
        stacked_in = np.stack(batch_in_as_list_of_np_array)
        stacked_out = np.stack(batch_out_as_list_of_np_array)
        stacked_in_tensor = torch.from_numpy(stacked_in).to(torch.float32).to(device)[:, :, None]
        stacked_out_tensor = torch.from_numpy(stacked_out).to(torch.float32).to(device)

        combined_batches += [(stacked_in_tensor, stacked_out_tensor)]
    return combined_batches


def MAPELoss(y, y_hat, mask=None):
    mape = torch.abs(y - y_hat)
    mape = torch.mean(mape)
    return mape

'''
###############
utils above, model below
###############
'''
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
        ).to(device)
        
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
        best_model =train(
        training_data_set=training_data_set,
        model=self.__model,
        num_epochs=20,
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
            
            x = torch.from_numpy(x).to(torch.float32).to(device)
            
            i= x.shape[0] -self.__model_output_length
            parts = []
            while True:
                if i-self.__model_input_length <0:
                    break
                part = x[i-self.__model_input_length: i]
                parts.append(part)
                i -= self.__model_output_length
            parts = torch.stack(parts[::-1]).unsqueeze(-1)
            y = self.__model(parts).unsqueeze(-1)
            y = torch.flatten(y)
        return y.cpu().numpy()
            
#             while i< x.shape[1]: 
#                 iter_num+=1
#                 y = self.__model(x[:,i-self.__model_input_length:i,:])
#                 y = y.squeeze(0).squeeze(-1).numpy()
#                 y.astype(np.float64)
#                 y_lst.append(y_lst)
#                 i+=self.__model_output_length
#         return np.concatenate(y_lst)[:x.shape[1]-self.__model_input_length]




