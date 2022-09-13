"""
***********************************************************************************************************************
    imports
***********************************************************************************************************************
"""

import numpy as np
import torch
import copy
from torch.autograd import Variable
import random
import math
import src.framework__test_bench as framework__test_bench
import time

"""
***********************************************************************************************************************
    CONSTANTS
***********************************************************************************************************************
"""

__msg = "[PytorchTester]"
__padding = -9999999999

"""
***********************************************************************************************************************
    Helper functions
***********************************************************************************************************************
"""


def __convert_np_array_to_pytorch_tensor(array):
    return torch.from_numpy(array).to(torch.float32)


def __plot_prediction_of_random_sample(training_data_set, best_model):
    print(__msg, f"Plotting prediction for some random sample in the test set.")
    test_sample = random.choice([ts for ts in training_data_set])
    how_much_to_give = len(test_sample) // 2
    how_much_to_predict = len(test_sample) - how_much_to_give
    returned_ts_as_np_array = predict(
        ts_as_df_start=test_sample[: how_much_to_give],
        how_much_to_predict=how_much_to_predict,
        best_model=best_model
    )
    framework__test_bench.plot_result(
        original=test_sample,
        prediction_as_np_array=returned_ts_as_np_array,
    )
    out_should_be = test_sample["sample"].to_numpy()[how_much_to_give:]
    mse_here = (np.square(out_should_be - returned_ts_as_np_array)).mean()
    print(__msg, f"MSE of this prediction is: {mse_here}")


"""
***********************************************************************************************************************
    Helper functions for batch making
***********************************************************************************************************************
"""


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
        stacked_in_tensor = Variable(__convert_np_array_to_pytorch_tensor(stacked_in).to(get_device()))[:, :, None]
        stacked_out_tensor = Variable(__convert_np_array_to_pytorch_tensor(stacked_out).to(get_device()))

        true_if_pad = (stacked_out_tensor == __padding)
        assert true_if_pad.sum() == 0
        false_if_pad = (stacked_out_tensor != __padding)
        predict_length = stacked_out_tensor.size(1)
        combined_batches += [(stacked_in_tensor, stacked_out_tensor)]
    return combined_batches


def __prepare_batches(training_data_set, model_input_length, batch_size):
    list_of_np_array = [
        ts_as_df["sample"].to_numpy()
        for ts_as_df in training_data_set
    ]
    list_of_input_output_np_array = [
        (arr[i: model_input_length + i], arr[model_input_length + i: model_input_length + i + 1])
        for arr in list_of_np_array
        for i in range(len(arr) - model_input_length)
    ]
    print(__msg, f"number of training samples = {len(list_of_input_output_np_array)}")
    list_of_input_output_np_array_batched = __partition_list_to_batches(
        list_of_something=list_of_input_output_np_array, batch_size=batch_size
    )
    combined = __combine_batches_of_np_array(batches=list_of_input_output_np_array_batched)
    return combined


"""
***********************************************************************************************************************
    Helper functions for training
***********************************************************************************************************************
"""


def __do_batch(batch_data, optimizer, model, criterion):
    train_input, train_target = batch_data
    optimizer.zero_grad()
    out = model.forward(x=train_input)
    loss = criterion(out, train_target)
    # loss_array[true_if_pad] = 0
    # loss = loss_array.sum() / false_if_pad.sum()
    loss.backward()
    optimizer.step()
    return loss.item()


def __do_epoch(epoch_num, list_of_batch, training_data_set, optimizer, model, criterion):
    sum_of_losses = 0
    for i, batch_data in enumerate(list_of_batch):
        loss = __do_batch(batch_data=batch_data, optimizer=optimizer, model=model, criterion=criterion)
        # print(__msg, f"loss of batch {i + 1} / {len(list_of_batch)}: {loss}")
        sum_of_losses += loss
    # choose random sample and plot
    # if epoch_num % 5 == 0:
    #     __plot_prediction_of_random_sample(training_data_set=training_data_set, best_model=model)
    return sum_of_losses


"""
*******************************************************************************************************************
    API functions
*******************************************************************************************************************
"""


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_neural_network(training_data_set, model, num_epochs, model_input_length, batch_size, optimizer, criterion,
                         min_training_time_in_seconds=5):
    list_of_batch = __prepare_batches(
        training_data_set=training_data_set,
        model_input_length=model_input_length,
        batch_size=batch_size
    )
    epoch_time = 0
    training_start_time = time.time()
    min_sum_of_losses = float('inf')
    best_model = copy.deepcopy(model)
    for e in range(99999999):
        if (e >= num_epochs) and (time.time() - training_start_time > min_training_time_in_seconds):
            break
        epoch_start_time = time.time()
        sum_of_losses = __do_epoch(
            epoch_num=e, list_of_batch=list_of_batch, training_data_set=training_data_set, optimizer=optimizer,
            model=model, criterion=criterion
        )
        if sum_of_losses < min_sum_of_losses:
            min_sum_of_losses = sum_of_losses
            best_model = copy.deepcopy(model)
            assert not (best_model is model)  # assert different objects
        epoch_stop_time = time.time()
        epoch_time = epoch_stop_time - epoch_start_time
        avg_loss = sum_of_losses / len(list_of_batch)
        print(__msg, f"Epoch {e + 1} done. Epoch time was {epoch_time}. Average loss for the batches in epoch is {avg_loss}")
    return best_model


def predict(ts_as_df_start, how_much_to_predict, best_model):
    with torch.no_grad():
        ts_as_np = ts_as_df_start["sample"].to_numpy()
        ts_as_tensor = __convert_np_array_to_pytorch_tensor(ts_as_np)[None, :, None].to(get_device())
        for _ in range(how_much_to_predict):
            prediction = best_model.forward(ts_as_tensor)
            ts_as_tensor = torch.cat([ts_as_tensor, prediction[None, :]], dim=1)
        prediction_flattened = ts_as_tensor.view(how_much_to_predict + len(ts_as_df_start)).cpu()
        y = prediction_flattened.detach().numpy()[-how_much_to_predict:]
        res = np.float64(y)
        assert isinstance(res, np.ndarray)
        assert len(res) == how_much_to_predict
        assert res.shape == (how_much_to_predict,)
        assert res.dtype == np.float64
        return res