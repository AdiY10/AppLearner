import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

import time
import copy
import random
import math
import matplotlib.pyplot as plt


import math
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import torch
import argparse
import numpy as np



import json
import matplotlib.pyplot as plt
import pandas as pd
import random
from datetime import datetime, timedelta
from os import listdir
from os.path import isfile, join
import numpy as np
from copy import deepcopy


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



###########################################################################################################################################################################

class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, ::2, :]

    def odd(self, x):
        return x[:, 1::2, :]

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.even(x), self.odd(x))


class Interactor(nn.Module):
    def __init__(self, in_planes, splitting=True,
                 kernel = 5, dropout=0.5, groups = 1, hidden_size = 1, INN = True):
        super(Interactor, self).__init__()
        self.modified = INN
        self.kernel_size = kernel
        self.dilation = 1
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.groups = groups
        if self.kernel_size % 2 == 0:
            pad_l = self.dilation * (self.kernel_size - 2) // 2 + 1 #by default: stride==1 
            pad_r = self.dilation * (self.kernel_size) // 2 + 1 #by default: stride==1 

        else:
            pad_l = self.dilation * (self.kernel_size - 1) // 2 + 1 # we fix the kernel size of the second layer as 3.
            pad_r = self.dilation * (self.kernel_size - 1) // 2 + 1
        self.splitting = splitting
        self.split = Splitting()

        modules_P = []
        modules_U = []
        modules_psi = []
        modules_phi = []
        prev_size = 1

        size_hidden = self.hidden_size
        modules_P += [
            nn.ReplicationPad1d((pad_l, pad_r)),

            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups= self.groups),
            nn.Tanh()
        ]
        modules_U += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups= self.groups),
            nn.Tanh()
        ]

        modules_phi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups= self.groups),
            nn.Tanh()
        ]
        modules_psi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups= self.groups),
            nn.Tanh()
        ]
        self.phi = nn.Sequential(*modules_phi)
        self.psi = nn.Sequential(*modules_psi)
        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        if self.modified:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)

            d = x_odd.mul(torch.exp(self.phi(x_even)))
            c = x_even.mul(torch.exp(self.psi(x_odd)))

            x_even_update = c + self.U(d)
            x_odd_update = d - self.P(c)

            return (x_even_update, x_odd_update)

        else:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)

            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)

            return (c, d)


class InteractorLevel(nn.Module):
    def __init__(self, in_planes, kernel, dropout, groups , hidden_size, INN):
        super(InteractorLevel, self).__init__()
        self.level = Interactor(in_planes = in_planes, splitting=True,
                 kernel = kernel, dropout=dropout, groups = groups, hidden_size = hidden_size, INN = INN)

    def forward(self, x):
        (x_even_update, x_odd_update) = self.level(x)
        return (x_even_update, x_odd_update)

class LevelSCINet(nn.Module):
    def __init__(self,in_planes, kernel_size, dropout, groups, hidden_size, INN):
        super(LevelSCINet, self).__init__()
        self.interact = InteractorLevel(in_planes= in_planes, kernel = kernel_size, dropout = dropout, groups =groups , hidden_size = hidden_size, INN = INN)

    def forward(self, x):
        (x_even_update, x_odd_update) = self.interact(x)
        return x_even_update.permute(0, 2, 1), x_odd_update.permute(0, 2, 1) #even: B, T, D odd: B, T, D

class SCINet_Tree(nn.Module):
    def __init__(self, in_planes, current_level, kernel_size, dropout, groups, hidden_size, INN):
        super().__init__()
        self.current_level = current_level


        self.workingblock = LevelSCINet(
            in_planes = in_planes,
            kernel_size = kernel_size,
            dropout = dropout,
            groups= groups,
            hidden_size = hidden_size,
            INN = INN)


        if current_level!=0:
            self.SCINet_Tree_odd=SCINet_Tree(in_planes, current_level-1, kernel_size, dropout, groups, hidden_size, INN)
            self.SCINet_Tree_even=SCINet_Tree(in_planes, current_level-1, kernel_size, dropout, groups, hidden_size, INN)
    
    def zip_up_the_pants(self, even, odd):
        even = even.permute(1, 0, 2)
        odd = odd.permute(1, 0, 2) #L, B, D
        even_len = even.shape[0]
        odd_len = odd.shape[0]
        mlen = min((odd_len, even_len))
        _ = []
        for i in range(mlen):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        if odd_len < even_len: 
            _.append(even[-1].unsqueeze(0))
        return torch.cat(_,0).permute(1,0,2) #B, L, D
        
    def forward(self, x):
        x_even_update, x_odd_update= self.workingblock(x)
        # We recursively reordered these sub-series. You can run the ./utils/recursive_demo.py to emulate this procedure. 
        if self.current_level ==0:
            return self.zip_up_the_pants(x_even_update, x_odd_update)
        else:
            return self.zip_up_the_pants(self.SCINet_Tree_even(x_even_update), self.SCINet_Tree_odd(x_odd_update))

class EncoderTree(nn.Module):
    def __init__(self, in_planes,  num_levels, kernel_size, dropout, groups, hidden_size, INN):
        super().__init__()
        self.levels=num_levels
        self.SCINet_Tree = SCINet_Tree(
            in_planes = in_planes,
            current_level = num_levels-1,
            kernel_size = kernel_size,
            dropout =dropout ,
            groups = groups,
            hidden_size = hidden_size,
            INN = INN)
        
    def forward(self, x):

        x= self.SCINet_Tree(x)

        return x

class SCINet(nn.Module):
    def __init__(self, output_len, input_len, input_dim = 9, hid_size = 1, num_stacks = 1,
                num_levels = 3, num_decoder_layer = 1, concat_len = 0, groups = 1, kernel = 5, dropout = 0.5,
                 single_step_output_One = 0, input_len_seg = 0, positionalE = False, modified = True, RIN=False):
        super(SCINet, self).__init__()

        self.input_dim = input_dim
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_size = hid_size
        self.num_levels = num_levels
        self.groups = groups
        self.modified = modified
        self.kernel_size = kernel
        self.dropout = dropout
        self.single_step_output_One = single_step_output_One
        self.concat_len = concat_len
        self.pe = positionalE
        self.RIN=RIN
        self.num_decoder_layer = num_decoder_layer

        self.blocks1 = EncoderTree(
            in_planes=self.input_dim,
            num_levels = self.num_levels,
            kernel_size = self.kernel_size,
            dropout = self.dropout,
            groups = self.groups,
            hidden_size = self.hidden_size,
            INN =  modified)

        if num_stacks == 2: # we only implement two stacks at most.
            self.blocks2 = EncoderTree(
                in_planes=self.input_dim,
            num_levels = self.num_levels,
            kernel_size = self.kernel_size,
            dropout = self.dropout,
            groups = self.groups,
            hidden_size = self.hidden_size,
            INN =  modified)

        self.stacks = num_stacks

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self.projection1 = nn.Conv1d(self.input_len, self.output_len, kernel_size=1, stride=1, bias=False)
        self.div_projection = nn.ModuleList()
        self.overlap_len = self.input_len//4
        self.div_len = self.input_len//6

        if self.num_decoder_layer > 1:
            self.projection1 = nn.Linear(self.input_len, self.output_len)
            for layer_idx in range(self.num_decoder_layer-1):
                div_projection = nn.ModuleList()
                for i in range(6):
                    lens = min(i*self.div_len+self.overlap_len,self.input_len) - i*self.div_len
                    div_projection.append(nn.Linear(lens, self.div_len))
                self.div_projection.append(div_projection)

        if self.single_step_output_One: # only output the N_th timestep.
            if self.stacks == 2:
                if self.concat_len:
                    self.projection2 = nn.Conv1d(self.concat_len + self.output_len, 1,
                                                kernel_size = 1, bias = False)
                else:
                    self.projection2 = nn.Conv1d(self.input_len + self.output_len, 1,
                                                kernel_size = 1, bias = False)
        else: # output the N timesteps.
            if self.stacks == 2:
                if self.concat_len:
                    self.projection2 = nn.Conv1d(self.concat_len + self.output_len, self.output_len,
                                                kernel_size = 1, bias = False)
                else:
                    self.projection2 = nn.Conv1d(self.input_len + self.output_len, self.output_len,
                                                kernel_size = 1, bias = False)

        # For positional encoding
        self.pe_hidden_size = input_dim
        if self.pe_hidden_size % 2 == 1:
            self.pe_hidden_size += 1
    
        num_timescales = self.pe_hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0

        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                max(num_timescales - 1, 1))
        temp = torch.arange(num_timescales, dtype=torch.float32)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)

        ### RIN Parameters ###
        if self.RIN:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, input_dim))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, input_dim))
    
    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32, device=x.device)  # tensor([0., 1., 2., 3., 4.], device='cuda:0')
        temp1 = position.unsqueeze(1)  # 5 1
        temp2 = self.inv_timescales.unsqueeze(0)  # 1 256
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)  # 5 256
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)  #[T, C]
        signal = F.pad(signal, (0, 0, 0, self.pe_hidden_size % 2))
        signal = signal.view(1, max_length, self.pe_hidden_size)
    
        return signal

    def forward(self, x):
        assert self.input_len % (np.power(2, self.num_levels)) == 0 # evenly divided the input length into two parts. (e.g., 32 -> 16 -> 8 -> 4 for 3 levels)
        if self.pe:
            pe = self.get_position_encoding(x)
            if pe.shape[2] > x.shape[2]:
                x += pe[:, :, :-1]
            else:
                x += self.get_position_encoding(x)

        ### activated when RIN flag is set ###
        if self.RIN:
            print('/// RIN ACTIVATED ///\r',end='')
            means = x.mean(1, keepdim=True).detach()
            #mean
            x = x - means
            #var
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            # affine
            # print(x.shape,self.affine_weight.shape,self.affine_bias.shape)
            x = x * self.affine_weight + self.affine_bias

        # the first stack
        res1 = x
        x = self.blocks1(x)
        x += res1
        if self.num_decoder_layer == 1:
            x = self.projection1(x)
        else:
            x = x.permute(0,2,1)
            for div_projection in self.div_projection:
                output = torch.zeros(x.shape,dtype=x.dtype).cuda()
                for i, div_layer in enumerate(div_projection):
                    div_x = x[:,:,i*self.div_len:min(i*self.div_len+self.overlap_len,self.input_len)]
                    output[:,:,i*self.div_len:(i+1)*self.div_len] = div_layer(div_x)
                x = output
            x = self.projection1(x)
            x = x.permute(0,2,1)

        if self.stacks == 1:
            ### reverse RIN ###
            if self.RIN:
                x = x - self.affine_bias
                x = x / (self.affine_weight + 1e-10)
                x = x * stdev
                x = x + means

            return x

        elif self.stacks == 2:
            MidOutPut = x
            if self.concat_len:
                x = torch.cat((res1[:, -self.concat_len:,:], x), dim=1)
            else:
                x = torch.cat((res1, x), dim=1)

            # the second stack
            res2 = x
            x = self.blocks2(x)
            x += res2
            x = self.projection2(x)
            
            ### Reverse RIN ###
            if self.RIN:
                MidOutPut = MidOutPut - self.affine_bias
                MidOutPut = MidOutPut / (self.affine_weight + 1e-10)
                MidOutPut = MidOutPut * stdev
                MidOutPut = MidOutPut + means

            if self.RIN:
                x = x - self.affine_bias
                x = x / (self.affine_weight + 1e-10)
                x = x * stdev
                x = x + means

            return x, MidOutPut


def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x





"""
***********************************************************************************************************************
    TimeSeriesDataSet Class
***********************************************************************************************************************
"""

class TimeSeriesDataSet:
    """
    Class that houses time series data set.
    """

    def __init__(self, list_of_df):
        self.__list_of_df = list_of_df
        self.__is_data_scaled = False
        self.__mean = None
        self.__std = None

    """
    *******************************************************************************************************************
        Helper functions
    *******************************************************************************************************************
    """

    def __get_mean_and_std(self):
        """
        calculates mean and std of all samples
        @return: mean and std of all samples (type np_array)
        """
        np_array_list = []
        for df in self:
            np_array_list += [df["sample"].to_numpy()]
        flat_np_array = np.concatenate(np_array_list)
        return flat_np_array.mean(), flat_np_array.std()

    """
    *******************************************************************************************************************
        API functions
    *******************************************************************************************************************
    """

    def __getitem__(self, key):
        return self.__list_of_df[key]

    def __len__(self):
        return len(self.__list_of_df)

    def sub_sample_data(self, sub_sample_rate):
        """
        creates sub sampling according to the rate (if for example rate = 5, then every 5 samples, the one with the
        maximal value is chosen to be in the data set).
        @param sub_sample_rate:
        """
        new_list_of_df = []

        for df in self:
            sub_sampled_data = df.groupby(df.index // sub_sample_rate).max()
            assert len(sub_sampled_data) == ((len(df) + sub_sample_rate - 1) // sub_sample_rate)
            new_list_of_df.append(sub_sampled_data)

        self.__list_of_df = new_list_of_df

    def filter_data_that_is_too_short(self, data_length_limit):
        """
        filters the data samples. all data samples that have a length that is lower than data_length_limit will be
        removed from the dataset
        @param data_length_limit: minimal length of sample
        """
        new_list_of_df = []

        for df in self:
            if len(df) > data_length_limit:
                new_list_of_df.append(df)

        self.__list_of_df = new_list_of_df

    def plot_dataset(self, number_of_samples):
        """
        randomly selects samples from the data sets and plots . x-axis is time and y-axis is the value
        @param number_of_samples: number of randomly selected samples
        """
        samples = random.sample(self.__list_of_df, k=number_of_samples)
        for df in samples:
            # plt.close("all")
            ts = df["sample"].copy()
            ts.index = [time for time in df["time"]]
            ts.plot()
            plt.show()

    def unscale_data(self,arr):
        if isinstance(arr,pd.DataFrame):
            df = deepcopy(arr)
            df['sample'] = (df['sample']*self.__std)+self.__mean
            return df
        return (arr*self.__std)+self.__mean

    def scale_data(self):
        """
        rescaling the distribution of values so that the mean of observed values is 0, and the std is 1.
        each sample is standardized (value - mean / std)
        """
        assert not self.__is_data_scaled
        self.__is_data_scaled = True
        self.__mean, self.__std = self.__get_mean_and_std()
        # print(f"self.__mean = {self.__mean}, self.__std = {self.__std}", )
        # print("max_sample = ", max_sample, " min_sample = ", min_sample)
        for df in self:
            standardized_sample_column = (df["sample"] - self.__mean) / self.__std
            # print("sample", df["sample"] , standardized_sample_column)
            df["sample"] = standardized_sample_column

    def split_to_train_and_test(self, length_to_predict):
        """
        not the usual train and test split!!!!
        according to an input, length to predict, we split the entire data set to train set and test set.
        The test set will be the same as the dataset in self. The train set will have the same amount of samples,
        but they will be shorter samples with their "tips" cut off.
        @param length_to_predict: The length to cut off from the train set.
        @return: train data set and test data set with sizes according to the input percentage.
        """
        assert 0 < length_to_predict < min([len(df) for df in self])
        assert isinstance(length_to_predict, int)
        random.shuffle(self.__list_of_df)
        # copy info to test
        test = TimeSeriesDataSet(list_of_df=self.__list_of_df)
        test.__is_data_scaled = self.__is_data_scaled
        test.__mean = self.__mean
        test.__std = self.__std
        # copy info to train
        train = TimeSeriesDataSet(list_of_df=[df[:-length_to_predict] for df in self.__list_of_df])
        train.__is_data_scaled = self.__is_data_scaled
        train.__mean = self.__mean
        train.__std = self.__std
        assert min(len(df) for df in train) == (min(len(df) for df in test) - length_to_predict)
        assert max(len(df) for df in train) == (max(len(df) for df in test) - length_to_predict)
        return train, test
    
    def split_to_train_and_test_SCINet(self, train_test_ratio):
        random.shuffle(self.__list_of_df)
        dfs = self.__list_of_df
        # copy info to train
        train = TimeSeriesDataSet(list_of_df=dfs[:int(train_test_ratio*len(dfs))])
        train.__is_data_scaled = self.__is_data_scaled
        train.__mean = self.__mean
        train.__std = self.__std
        
        test = TimeSeriesDataSet(list_of_df=dfs[int(train_test_ratio*len(dfs)):])
        test.__is_data_scaled = self.__is_data_scaled
        test.__mean = self.__mean
        test.__std = self.__std
        return train, test
    


"""
***********************************************************************************************************************
    get train and test datasets
***********************************************************************************************************************
"""


def __get_names_of_json_files_in_directory(directory_path):
    """
    returns the names of the json files in the directory (specified by the param "directory_path"
    @param directory_path: the name of the directory
    @return: the names of json files in directory
    """
    json_names = [f for f in listdir(directory_path) if (isfile(join(directory_path, f)) and ("json" in f))]
    return json_names


def __get_names_of_relevant_files(metric, path_to_data):
    """
    find the names of files that contain a specified metric in the directory.
    @param metric: specified metric to get : "container_cpu", "container_mem", "node_mem"
    @param path_to_data: the path to the directory
    @return: a list of the files that contain the specified from each json file in the directory specified
    """
    list_of_files = __get_names_of_json_files_in_directory(path_to_data)
    print(list_of_files)
    relevant_files = [file for file in list_of_files if (metric in file)]
    relevant_files.sort()
    return relevant_files


def __get_app_name_from_key(key: str):
    """
    @param key: column from the original data indicating name and other properties
    @return: the name of the app
    """
    app_name = key.split(", ")[0]
    namespace = key.split(", ")[1]
    node = key.split(", ")[2]
    pod = key.split(", ")[3]
    return app_name


def __get_data_as_list_of_df_from_file(data_dict, application_name):
    """
    given data dictionary and an application name, appends all of the data that is associated with the application name
    to create a list and returns it
    @param data_dict: dictionary of data
    @param application_name:
    @return: time series of a specified application name from a data dictionary
    """
    result_list = []
    relevant_keys = [k for k in data_dict.keys() if (application_name == __get_app_name_from_key(key=k))]
    for k in relevant_keys:
        list_of_ts = data_dict[k]
        for time_series in list_of_ts:
            start_time = datetime.strptime(time_series["start"], "%Y-%m-%d %H:%M:%S")
            stop_time = datetime.strptime(time_series["stop"], "%Y-%m-%d %H:%M:%S")
            date_time_range = [start_time + timedelta(minutes=i) for i in range(len(time_series["data"]))]
            assert date_time_range[-1] == stop_time
            time_series_as_df = pd.DataFrame(
                {
                    "sample": time_series["data"],
                    "time": date_time_range
                },
                # index=date_time_range
            )
            result_list.append(time_series_as_df)
    return result_list


def __get_data_as_list_of_df(metric, application_name, path_to_data):
    """
    @param metric: specified metric to get : "container_cpu", "container_mem", "node_mem"
    @param application_name: application name
    @param path_to_data: directory of json files
    @return: a list of the metric data of a specified app from all the json files in the directory (found by
    path_to_data)
    """
    file_names = __get_names_of_relevant_files(metric=metric, path_to_data=path_to_data)
    result_list = []
    for file_name in file_names:
        with open(f'{path_to_data}{file_name}') as json_file:
            data_dict = json.load(json_file)
            result_list += __get_data_as_list_of_df_from_file(
                data_dict=data_dict,
                application_name=application_name
            )

    return result_list


def get_data_set(metric, application_name, path_to_data):
    """
    @param metric: specified metric to get : "container_cpu", "container_mem", "node_mem"
    @param application_name: application name
    @param path_to_data: directory of json files
    @return: a TimeSeriesDataSet according to an app with a specified metric.
    """
    # constants
    __supported_metrics = ["container_cpu", "container_mem", "node_mem"]

    # checks if metric is one of the supported metrics
    assert metric in __supported_metrics

    list_of_df = __get_data_as_list_of_df(
        metric=metric,
        application_name=application_name,
        path_to_data=path_to_data
    )

    ds = TimeSeriesDataSet(list_of_df=list_of_df)
    return ds


def get_amount_of_data_per_application(metric, path_to_data):
    """
    @param metric: specified metric to get : "container_cpu", "container_mem", "node_mem"
    @param path_to_data: directory of json files
    @return: a sorted list of amount of data per each app that has a specific metric in a specified file
    """
    __supported_metrics = ["container_cpu", "container_mem", "node_mem"]
    assert metric in __supported_metrics
    file_names = __get_names_of_relevant_files(metric=metric, path_to_data=path_to_data)
    application_names_histogram = {}
    for file_name in file_names:
        with open(f'{path_to_data}{file_name}') as json_file:
            data_dict = json.load(json_file)
            for k in data_dict.keys():
                app_name = __get_app_name_from_key(key=k)
                # count number of time series samples
                amount_of_data = 0
                for ts in data_dict[k]:
                    amount_of_data += len(ts["data"])
                # add count to running count
                if app_name in application_names_histogram:
                    application_names_histogram[app_name] += amount_of_data
                else:
                    application_names_histogram[app_name] = amount_of_data
    result = sorted(application_names_histogram.items(), key=lambda item: - item[1])
    return result


"""
***********************************************************************************************************************
    main function
***********************************************************************************************************************
"""


def main():
    print("Start.")
    length_to_predict = 4
    test = 0
    if test == 0:
        print("Getting DataSet.")
        dataset = get_data_set(
            metric="container_mem",
            application_name="bridge-marker",
            path_to_data="../data/"
        )
        print("Plotting.")
        dataset.plot_dataset(number_of_samples=3)
        print("Subsampling.")
        dataset.sub_sample_data(sub_sample_rate=60)
        print("Plotting.")
        dataset.plot_dataset(number_of_samples=3)
        print("Normalizing.")
        dataset.scale_data()
        print("Plotting.")
        dataset.plot_dataset(number_of_samples=3)
        print("Filtering time series that are too short.")
        dataset.filter_data_that_is_too_short(data_length_limit=2 * length_to_predict)
        print("Splitting.")
        train, test = dataset.split_to_train_and_test(length_to_predict=length_to_predict)
        print("Plotting.")
        train.plot_dataset(number_of_samples=10)
        test.plot_dataset(number_of_samples=10)
    else:
        hist = get_amount_of_data_per_application(
            metric="container_mem",
            path_to_data="../data/"
        )
        print(hist)


"""
***********************************************************************************************************************
    run main function
***********************************************************************************************************************
"""

if __name__ == "__main__":
    main()