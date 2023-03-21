import torch 
import torch.nn as nn
import numpy as np
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
                plt.plot(range(model_input_length+model_output_length), list(train_input[i,:,0])+list(train_target[i,:]))
                plt.plot(range(model_input_length,model_input_length+model_output_length), list(out_d[i,:,0]))
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

