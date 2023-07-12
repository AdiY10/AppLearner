# Author "Jesus Camacho Villanueva <jcamacho@redhat.com>"

# imports
import matplotlib.pyplot as plt
import sys, getopt, subprocess
import src.framework__data_set as ds
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from matplotlib import pylab
from datetime import datetime, timedelta
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential, model_from_json
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None
tf.random.set_seed(0)

class LSTM_Jesus:

    # dataset
    def __init__(self, metric, application_name, interval, path="../data/"):

        self.interval=interval

        # read dataset
        self.dataset = ds.get_data_set(
            metric=metric,
            application_name=application_name,
            path_to_data=path
        )


    # train the model
    def training(self, n_lookback, n_forecast, path):

        scaler_train = MinMaxScaler()
        for i in range(0, int(len(dataset))):
            dataset[i]['sample'] = scaler_train.fit_transform(dataset[i][['sample']])

        # generate the input and output sequences
        n_lookback = n_lookback  # length of input sequences (lookback period)
        n_forecast = n_forecast  # length of output sequences (forecast period)

        # fit the model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
        model.add(LSTM(units=50))
        model.add(Dense(n_forecast))

        # model compilation
        model.compile(loss='mean_squared_error', optimizer='adam')

        upto = int(len(dataset) * 0.67)

        X = []
        Y = []

        # store X->Y from every interval recorder separately (no concatenation between intervals)
        for i in range(0, upto):

            data_train = dataset[i]

            y = data_train['sample'].fillna(method='ffill')
            y = y.values.reshape(-1, 1)

            if(n_lookback >= len(y) - n_forecast + 1):
                continue

            for j in range(n_lookback, len(y) - n_forecast + 1):
                X.append(y[j - n_lookback: j])
                Y.append(y[j: j + n_forecast])

        X = np.array(X)
        Y = np.array(Y)

        # train model
        model.fit(X, Y, epochs=100, batch_size=100, verbose=2)

        # save trained model
        # serialize model to JSON
        model_json = model.to_json()
        with open(path+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(path+".h5")
        print("Saved model to disk")

    
    # model prediction
    def predict_values(self, metric, app, n_lookback, horizon, dataset_test):

        # input and output sequences
        n_lookback = int(n_lookback)  # length of input sequences (lookback period)
        n_forecast = int(horizon)  # length of output sequences (forecast period)

        # trained model file to load
        file = "../TrainedModels/lookback_"+str(n_lookback)+"/app_"+metric+"/"+app+"/"+metric+"_"+app+"_"+str(n_lookback)+"_"+str(horizon)

        # load json and create model
        json_file = open(file+".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(file+".h5")
        print("Loaded model " + file + " from disk")

        # scale data
        scaler_test = MinMaxScaler()
        dataset_test['sample'] = scaler_test.fit_transform(dataset_test[['sample']])

        # samples
        y = dataset_test['sample'].fillna(method='ffill')
        y = y.values.reshape(-1, 1)
        test_len = n_lookback
        data_test = dataset_test.head(test_len)
        len_data = len(dataset_test['sample'].to_numpy())

        # predict forecasts
        X_ = y[test_len-n_lookback:test_len]
        X_ = X_.reshape(1, n_lookback, 1)
        Y_ = model.predict(X_).reshape(-1, 1)
        Y3_ = []
        Y3_ = np.array(Y3_)
        Y3_ = Y_
        # cut first prediction when longer than test set
        if(len(Y3_) > len_data):
            Y3_ = Y_[0:len_data - n_lookback]

        # append forecasts
        for i in range(int((len(dataset_test)-len(data_test))/n_forecast) - 0):
            X2_ = y[test_len-n_lookback+(n_forecast*(i+1)):test_len+(n_forecast*(i+1))]
            X2_ = X2_.reshape(1, n_lookback, 1)
            Y2_ = model.predict(X2_).reshape(-1, 1)
            # cut last append
            if(i == int((len(dataset_test)-len(data_test))/n_forecast) - 1):
                until = len_data - len(Y3_) - n_lookback 
                Y3_ = np.concatenate((Y3_, Y2_[0:until]))
            else:
                Y3_ = np.concatenate((Y3_, Y2_))

        forecast_len = len(Y3_)

        # For calculating RMSE
        Y4_ = Y3_.flatten()
        Y5_ = dataset_test['sample'].to_numpy()

        # mean squared error
        rmse = np.sqrt(np.square(np.subtract(Y4_,Y5_[test_len:test_len+forecast_len])).mean())

        # MAE
        Y4_ = Y3_.flatten()
        Y5_ = dataset_test['sample'].to_numpy()
        mae = np.mean(np.abs(np.subtract(Y5_[test_len:test_len+forecast_len], Y4_)))

        # Recover from normalized data
        Y3_ = scaler_test.inverse_transform(Y3_)
        dataset_test['sample'] = scaler_test.inverse_transform(dataset_test[['sample']])

        return Y3_ , rmse, mae, test_len, forecast_len, len_data


    # print a given prediction
    def predict(self, metric, app, n_lookback, interval):

        # get interval
        number = int(self.interval) % int(len(self.dataset) * 0.33)

        # forecast interval selected (testing part)
        dataset_test = self.dataset[int(len(self.dataset) * 0.67) + number]

        # dataset_test must be greater than lookback
        if(len(dataset_test) <= int(n_lookback)):
            print("Interval not long enough")
            quit()

        # samples
        original_as_series = dataset_test['sample'].copy()
        x_axis = [time for time in dataset_test["time"]]
        original_as_series.index = x_axis
        ax = original_as_series.plot(color="blue", label="Sample", linewidth=1)

        # lists
        horizons = ["12", "24", "48", "96", "192", "384", "768", "1536"]
        colors = ["red", "limegreen", "cyan", "maroon", "orange", "lightgray", "yellow", "black"]
        rmse = []
        mae = []

        # iterate over the different forecast (horizon) values
        for i in range(8):
            # predict horizons
            Y3_ , rmse_, mae_, test_len, forecast_len, len_data = self.predict_values(metric, app, n_lookback, horizons[i], dataset_test)
            # plot results
            df_future = pd.DataFrame(columns=['Forecast'])
            df_future['Forecast'] = Y3_.flatten()
            predicted_as_series = df_future['Forecast']
            predicted_as_series.index = x_axis[test_len:test_len+forecast_len]
            predicted_as_series.plot(ax=ax, color=colors[i], label="Horizon: " + horizons[i], linewidth=1)
            rmse.append(rmse_)
            mae.append(mae_)

        # title
        plt.suptitle("Metric: " + metric + ", App: " + app + ", Lookback: " + str(n_lookback) + ", Horizon: 12-1536\n12-12 RMSE: " + str("{:.5f}".format(rmse[0])) + " - MAE: " + str("{:.5f}".format(mae[0])) + " | 12-192 RMSE: " + str("{:.5f}".format(rmse[4])) + " - MAE: " + str("{:.5f}".format(mae[4])) + "\n12-24 RMSE: " + str("{:.5f}".format(rmse[1])) + " - MAE: " + str("{:.5f}".format(mae[1])) + " | 12-384 RMSE: " + str("{:.5f}".format(rmse[5])) + " - MAE: " + str("{:.5f}".format(mae[5])) + "\n12-48 RMSE: " + str("{:.5f}".format(rmse[2])) + " - MAE: " + str("{:.5f}".format(mae[2])) + " | 12-768 RMSE: " + str("{:.5f}".format(rmse[6])) + " - MAE: " + str("{:.5f}".format(mae[6])) + "\n12-96 RMSE: " + str("{:.5f}".format(rmse[3])) + " - MAE: " + str("{:.5f}".format(mae[3])) + " | 12-1536 RMSE: " + str("{:.5f}".format(rmse[7])) + " - MAE: " + str("{:.5f}".format(mae[7])) + "\n Interval: " + str(interval) + ". Samples in this interval: " + str(len_data), fontsize=6)

        # plot features
        plt.legend(ncol=2, loc="upper left", fontsize=6)
        plt.xlabel("Minutes")
        plt.ylabel("Values")
        #plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.subplots_adjust(bottom=0.1)
        plt.show()


    # print prediction errors for a given interval
    def error(self, path, interval):

        # get interval
        number = int(interval) % int(len(dataset_) * 0.33)

        # forecast interval selected (testing part)
        dataset_test = dataset_[int(len(dataset_) * 0.67) + number]

        # dataset_test must be greater than lookback
        if(len(dataset_test) <= int(n_lookback)):
            print("Interval not long enough")
            quit()

        # lists
        horizons = ["12", "24", "48", "96", "192", "384", "768", "1536"]
        colors = ["red", "limegreen", "cyan", "maroon", "orange", "lightgray", "yellow", "black"]
        rmse = []
        mae = []

        # iterate over the different forecast (horizon) values
        for i in range(8):
            # predict horizons
            Y3_ , rmse_, mae_, test_len, forecast_len, len_data = self.predict_values(n_lookback, horizons[i], dataset_test)
            # append results
            rmse.append(rmse_)
            mae.append(mae_)

        # print errors
        file2 = path;
        with open(file2+".error", 'w') as sys.stdout:
            print("forecast interval #samples RMSE MAE")
            print("-----------------------------------")
            print("12 "+str(number)+" "+str(len(dataset_test))+" "+str("{:.5f}".format(rmse[0]))+" "+str("{:.5f}".format(mae[0])))
            print("24 "+str(number)+" "+str(len(dataset_test))+" "+str("{:.5f}".format(rmse[1]))+" "+str("{:.5f}".format(mae[1])))
            print("48 "+str(number)+" "+str(len(dataset_test))+" "+str("{:.5f}".format(rmse[2]))+" "+str("{:.5f}".format(mae[2])))
            print("96 "+str(number)+" "+str(len(dataset_test))+" "+str("{:.5f}".format(rmse[3]))+" "+str("{:.5f}".format(mae[3])))
            print("192 "+str(number)+" "+str(len(dataset_test))+" "+str("{:.5f}".format(rmse[4]))+" "+str("{:.5f}".format(mae[4])))
            print("384 "+str(number)+" "+str(len(dataset_test))+" "+str("{:.5f}".format(rmse[5]))+" "+str("{:.5f}".format(mae[5])))
            print("768 "+str(number)+" "+str(len(dataset_test))+" "+str("{:.5f}".format(rmse[6]))+" "+str("{:.5f}".format(mae[6])))
            print("1536 "+str(number)+" "+str(len(dataset_test))+" "+str("{:.5f}".format(rmse[7])+" "+str("{:.5f}".format(mae[7]))))


    # obtain a model
    def get_model(self, path):

        # trained models files to load
        file = path+"_"+trained_model;
        # load json and create model
        json_file = open(file+".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # load weights into new model
        model.load_weights(file+".h5")
        print("Loaded model " + file + " from disk")

        return model


    # obtain error for all intervals
    def error_all_intervals(self, path):

        # create file to write errors
        error_file = path+".error"
        try:
            os.remove(error_file)
        except OSError:
            pass
        file = open(error_file, 'w')
        file.write("forecast interval #samples RMSE MAE\n")
        file.write("-----------------------------------\n")
        file.close()

        # lists
        horizons = ["12", "24", "48", "96", "192", "384", "768", "1536"]
        colors = ["red", "limegreen", "cyan", "maroon", "orange", "lightgray", "yellow", "black"]

        # get models

        model_0 = self.get_model(horizons[0])
        model_1 = self.get_model(horizons[1])
        model_2 = self.get_model(horizons[2])
        model_3 = self.get_model(horizons[3])
        model_4 = self.get_model(horizons[4])
        model_5 = self.get_model(horizons[5])
        model_6 = self.get_model(horizons[6])
        model_7 = self.get_model(horizons[7])

        # obtain RMSE and MAE for all intervals
        for n in range(int(len(dataset_) * 0.33)):

            # get interval
            number = n % int(len(dataset_) * 0.33)

            # forecast interval selected (testing part)
            dataset_test = dataset_[int(len(dataset_) * 0.67) + number]

            print("Interval: "+str(number+1)+"/"+str(int(len(dataset_) * 0.33))+", #samples: "+(str(len(dataset_test))))

            # dataset_test must be euqls or greater than lookback
            if(len(dataset_test) <= int(n_lookback)):
                print("Interval not long enough")
                continue

            # iterate over the different forecast (horizon) values
            for i in range(8):

                # predict horizons & append errors
                file = open(error_file, 'a')

                if i == 0:
                    Y3_ , rmse_, mae_, test_len, forecast_len, len_data = self.predict_values(n_lookback, horizons[i], dataset_test, model_0)
                    file.write("12 "+str(number)+" "+str(test_len)+" "+str("{:.5f}".format(rmse))+" "+str("{:.5f}".format(mae))+"\n")
                elif i == 1:
                    Y3_ , rmse_, mae_, test_len, forecast_len, len_data = self.predict_values(n_lookback, horizons[i], dataset_test, model_1)
                    file.write("24 "+str(number)+" "+str(test_len)+" "+str("{:.5f}".format(rmse))+" "+str("{:.5f}".format(mae))+"\n")
                elif i == 2:
                    Y3_ , rmse_, mae_, test_len, forecast_len, len_data = self.predict_values(n_lookback, horizons[i], dataset_test, model_2)
                    file.write("48 "+str(number)+" "+str(test_len)+" "+str("{:.5f}".format(rmse))+" "+str("{:.5f}".format(mae))+"\n")
                elif i == 3:
                    Y3_ , rmse_, mae_, test_len, forecast_len, len_data = self.predict_values(n_lookback, horizons[i], dataset_test, model_3)
                    file.write("96 "+str(number)+" "+str(test_len)+" "+str("{:.5f}".format(rmse))+" "+str("{:.5f}".format(mae))+"\n")
                elif i == 4:
                    Y3_ , rmse_, mae_, test_len, forecast_len, len_data = self.predict_values(n_lookback, horizons[i], dataset_test, model_4)
                    file.write("192 "+str(number)+" "+str(test_len)+" "+str("{:.5f}".format(rmse))+" "+str("{:.5f}".format(mae))+"\n")
                elif i == 5:
                    Y3_ , rmse_, mae_, test_len, forecast_len, len_data = self.predict_values(n_lookback, horizons[i], dataset_test, model_5)
                    file.write("384 "+str(number)+" "+str(test_len)+" "+str("{:.5f}".format(rmse))+" "+str("{:.5f}".format(mae))+"\n")
                elif i == 6:
                    Y3_ , rmse_, mae_, test_len, forecast_len, len_data = self.predict_values(n_lookback, horizons[i], dataset_test, model_6)
                    file.write("768 "+str(number)+" "+str(test_len)+" "+str("{:.5f}".format(rmse))+" "+str("{:.5f}".format(mae))+"\n")
                else:
                    Y3_ , rmse_, mae_, test_len, forecast_len, len_data = self.predict_values(n_lookback, horizons[i], dataset_test, model_7)
                    file.write("1536 "+str(number)+" "+str(test_len)+" "+str("{:.5f}".format(rmse)+" "+str("{:.5f}".format(mae)))+"\n")
                file.close()


    # obtain quintiles
    def get_quantiles(self, path):

        # initialize some variables and arrays
        lookback = None
        quantiles = 12
        forecasts = 9
        Matrix_rmse = [[0 for x in range(quantiles)] for y in range(forecasts)] 
        Matrix_mae = [[0 for x in range(quantiles)] for y in range(forecasts)]

        # load deciles file
        file1 = open(path, 'r')
        Lines = file1.readlines()

        count = 0
        # get lookback
        for line in Lines:
            count += 1
            if(count > 2 and line.strip()):
                lookback = line.split(" ")[0]
                break

        # write RMSE and MAE statistics in the arrays
        count = 0
        new_quantile = 0
        new_forecast = 0
        for line in Lines:
            count += 1
            if(count < 3):
                continue
            if(line.strip()):
                forecast = line.split(" ")[1]
                quantile = line.split(" ")[2]
                total = line.split(" ")[3]
                rmse = line.split(" ")[4]
                mae = line.split(" ")[5]
                new_quantile = new_quantile + 1
                Matrix_rmse[new_forecast][new_quantile] = rmse
                Matrix_mae[new_forecast][new_quantile] = mae
            else:
                new_forecast = new_forecast + 1
                new_quantile = 0

        file1.close()

        return lookback, Matrix_rmse, Matrix_mae

    # print quantiles
    def quantiles(self, path):

        # fined lookback, and errors from the given file
        lookback, Matrix_rmse, Matrix_mae = self.get_quantiles(path)

        # obatin the name of the App
        if(lookback == '12'):
            file_name = path.split('/')[-1].split('_12.deciles', 1)[0]
        else:
            file_name = path.split('/')[-1].split('_24.deciles', 1)[0]

        # quantiles (here deciles)
        x = ['0.0', '<0.5', '<1.0', '<1.5', '<2.0', '<2.5', '<3.0', '<3.5', '<4.0', '<4.5', '<5.0', '<1']

        # map strings results into integers and print RMSE figure
        matrix_rmse = [list(map(int, x)) for x in Matrix_rmse]

        plt.plot(x, matrix_rmse[0], label="Forecast 12")
        plt.plot(x, matrix_rmse[1], label="Forecast 24")
        plt.plot(x, matrix_rmse[2], label="Forecast 48")
        plt.plot(x, matrix_rmse[3], label="Forecast 96")
        plt.plot(x, matrix_rmse[4], label="Forecast 192")
        plt.plot(x, matrix_rmse[5], label="Forecast 384")
        plt.plot(x, matrix_rmse[6], label="Forecast 768")
        plt.plot(x, matrix_rmse[7], label="Forecast 1536")

        plt.suptitle("App: " + file_name + "\nRMSE (lookback=" + lookback + ")", fontsize=10)
        plt.legend(loc='best')
        plt.xlabel("Error percentage")
        plt.ylabel("Samples by error percentage")
        plt.show()

        # map strings results into integers and print MAE figure
        matrix_mae = [list(map(int, x)) for x in Matrix_mae]

        plt.plot(x, matrix_mae[0], label="Forecast 12")
        plt.plot(x, matrix_mae[1], label="Forecast 24")
        plt.plot(x, matrix_mae[2], label="Forecast 48")
        plt.plot(x, matrix_mae[3], label="Forecast 96")
        plt.plot(x, matrix_mae[4], label="Forecast 192")
        plt.plot(x, matrix_mae[5], label="Forecast 384")
        plt.plot(x, matrix_mae[6], label="Forecast 768")
        plt.plot(x, matrix_mae[7], label="Forecast 1536")

        plt.suptitle("App: " + file_name + "\nMAE (lookback=" + lookback + ")", fontsize=10)
        plt.legend(loc='best')
        plt.xlabel("Error percentage")
        plt.ylabel("Samples by error percentage")
        plt.show()


    # function to get range count
    def get_range_count(self, file, forecast, lower, upper):
        cmd = f"cat {file} | grep '^{forecast}' | awk '{{ if ($4 >= {lower} && $4 < {upper}) print $4 }}' | wc -l"
        result = subprocess.check_output(cmd, shell=True).decode().strip()
        return int(result)


    # print quintiles
    def obtain_quantiles(self, path):

        # get file information
        arg = path
        lines_in_file = subprocess.check_output(["wc", arg]).decode().split()
        total_with_head = int(lines_in_file[0])
        total = total_with_head - 2
        total_by_forecast = total // 8
        file = arg[:-6]
        output_file = file + ".quintiles"
        lookback = file[-2:]

        # get quintiles
        range_12_1 = self.get_range_count(arg, "12", 0, 0.05)
        range_12_2 = self.get_range_count(arg, "12", 0.05, 0.10)
        range_12_3 = self.get_range_count(arg, "12", 0.10, 0.15)
        range_12_4 = self.get_range_count(arg, "12", 0.15, 0.20)
        range_12_5 = self.get_range_count(arg, "12", 0.20, 1.00)

        range_24_1 = self.get_range_count(arg, "24", 0, 0.05)
        range_24_2 = self.get_range_count(arg, "24", 0.05, 0.10)
        range_24_3 = self.get_range_count(arg, "24", 0.10, 0.15)
        range_24_4 = self.get_range_count(arg, "24", 0.15, 0.20)
        range_24_5 = self.get_range_count(arg, "24", 0.20, 1.00)

        range_48_1 = self.get_range_count(arg, "48", 0, 0.05)
        range_48_2 = self.get_range_count(arg, "48", 0.05, 0.10)
        range_48_3 = self.get_range_count(arg, "48", 0.10, 0.15)
        range_48_4 = self.get_range_count(arg, "48", 0.15, 0.20)
        range_48_5 = self.get_range_count(arg, "48", 0.20, 1.00)

        range_96_1 = self.get_range_count(arg, "96", 0, 0.05)
        range_96_2 = self.get_range_count(arg, "96", 0.05, 0.10)
        range_96_3 = self.get_range_count(arg, "96", 0.10, 0.15)
        range_96_4 = self.get_range_count(arg, "96", 0.15, 0.20)
        range_96_5 = self.get_range_count(arg, "96", 0.20, 1.00)

        range_192_1 = self.get_range_count(arg, "192", 0, 0.05)
        range_192_2 = self.get_range_count(arg, "192", 0.05, 0.10)
        range_192_3 = self.get_range_count(arg, "192", 0.10, 0.15)
        range_192_4 = self.get_range_count(arg, "192", 0.15, 0.20)
        range_192_5 = self.get_range_count(arg, "192", 0.20, 1.00)

        range_384_1 = self.get_range_count(arg, "384", 0, 0.05)
        range_384_2 = self.get_range_count(arg, "384", 0.05, 0.10)
        range_384_3 = self.get_range_count(arg, "384", 0.10, 0.15)
        range_384_4 = self.get_range_count(arg, "384", 0.15, 0.20)
        range_384_5 = self.get_range_count(arg, "384", 0.20, 1.00)

        range_768_1 = self.get_range_count(arg, "768", 0, 0.05)
        range_768_2 = self.get_range_count(arg, "768", 0.05, 0.10)
        range_768_3 = self.get_range_count(arg, "768", 0.10, 0.15)
        range_768_4 = self.get_range_count(arg, "768", 0.15, 0.20)
        range_768_5 = self.get_range_count(arg, "768", 0.20, 1.00)

        range_1536_1 = self.get_range_count(arg, "1536", 0, 0.05)
        range_1536_2 = self.get_range_count(arg, "1536", 0.05, 0.10)
        range_1536_3 = self.get_range_count(arg, "1536", 0.10, 0.15)
        range_1536_4 = self.get_range_count(arg, "1536", 0.15, 0.20)
        range_1536_5 = self.get_range_count(arg, "1536", 0.20, 1.00)

        # print quintiles
        with open(output_file, "w") as file:
            file.write("lookback forecast quantile total matches\n")
            file.write("----------------------------------------\n")
            file.write(f"{lookback} 12 0.00-0.05 {total_by_forecast} {range_12_1}\n")
            file.write(f"{lookback} 12 0.05-0.10 {total_by_forecast} {range_12_2}\n")
            file.write(f"{lookback} 12 0.10-0.15 {total_by_forecast} {range_12_3}\n")
            file.write(f"{lookback} 12 0.15-0.20 {total_by_forecast} {range_12_4}\n")
            file.write(f"{lookback} 12 0.20-1.00 {total_by_forecast} {range_12_5}\n")
            file.write("\n")
            file.write(f"{lookback} 24 0.00-0.05 {total_by_forecast} {range_24_1}\n")
            file.write(f"{lookback} 24 0.05-0.10 {total_by_forecast} {range_24_2}\n")
            file.write(f"{lookback} 24 0.10-0.15 {total_by_forecast} {range_24_3}\n")
            file.write(f"{lookback} 24 0.15-0.20 {total_by_forecast} {range_24_4}\n")
            file.write(f"{lookback} 24 0.20-1.00 {total_by_forecast} {range_24_5}\n")
            file.write("\n")
            file.write(f"{lookback} 48 0.00-0.05 {total_by_forecast} {range_48_1}\n")
            file.write(f"{lookback} 48 0.05-0.10 {total_by_forecast} {range_48_2}\n")
            file.write(f"{lookback} 48 0.10-0.15 {total_by_forecast} {range_48_3}\n")
            file.write(f"{lookback} 48 0.15-0.20 {total_by_forecast} {range_48_4}\n")
            file.write(f"{lookback} 48 0.20-1.00 {total_by_forecast} {range_48_5}\n")
            file.write("\n")
            file.write(f"{lookback} 96 0.00-0.05 {total_by_forecast} {range_96_1}\n")
            file.write(f"{lookback} 96 0.05-0.10 {total_by_forecast} {range_96_2}\n")
            file.write(f"{lookback} 96 0.10-0.15 {total_by_forecast} {range_96_3}\n")
            file.write(f"{lookback} 96 0.15-0.20 {total_by_forecast} {range_96_4}\n")
            file.write(f"{lookback} 96 0.20-1.00 {total_by_forecast} {range_96_5}\n")
            file.write("\n")
            file.write(f"{lookback} 192 0.00-0.05 {total_by_forecast} {range_192_1}\n")
            file.write(f"{lookback} 192 0.05-0.10 {total_by_forecast} {range_192_2}\n")
            file.write(f"{lookback} 192 0.10-0.15 {total_by_forecast} {range_192_3}\n")
            file.write(f"{lookback} 192 0.15-0.20 {total_by_forecast} {range_192_4}\n")
            file.write(f"{lookback} 192 0.20-1.00 {total_by_forecast} {range_192_5}\n")
            file.write("\n")
            file.write(f"{lookback} 384 0.00-0.05 {total_by_forecast} {range_384_1}\n")
            file.write(f"{lookback} 384 0.05-0.10 {total_by_forecast} {range_384_2}\n")
            file.write(f"{lookback} 384 0.10-0.15 {total_by_forecast} {range_384_3}\n")
            file.write(f"{lookback} 384 0.15-0.20 {total_by_forecast} {range_384_4}\n")
            file.write(f"{lookback} 384 0.20-1.00 {total_by_forecast} {range_384_5}\n")
            file.write("\n")
            file.write(f"{lookback} 768 0.00-0.05 {total_by_forecast} {range_768_1}\n")
            file.write(f"{lookback} 768 0.05-0.10 {total_by_forecast} {range_768_2}\n")
            file.write(f"{lookback} 768 0.10-0.15 {total_by_forecast} {range_768_3}\n")
            file.write(f"{lookback} 768 0.15-0.20 {total_by_forecast} {range_768_4}\n")
            file.write(f"{lookback} 768 0.20-1.00 {total_by_forecast} {range_768_5}\n")
            file.write("\n")
            file.write(f"{lookback} 1536 0.00-0.05 {total_by_forecast} {range_1536_1}\n")
            file.write(f"{lookback} 1536 0.05-0.10 {total_by_forecast} {range_1536_2}\n")
            file.write(f"{lookback} 1536 0.10-0.15 {total_by_forecast} {range_1536_3}\n")
            file.write(f"{lookback} 1536 0.15-0.20 {total_by_forecast} {range_1536_4}\n")
            file.write(f"{lookback} 1536 0.20-1.00 {total_by_forecast} {range_1536_5}\n")


    # function to get range count rmse
    def get_range_count_rmse(self, file, forecast, lower, upper):
        cmd = f"cat {file} | grep '^{forecast}' | awk '{{ if ($4 >= {lower} && $4 < {upper}) print $4 }}' | wc -l"
        result = subprocess.check_output(cmd, shell=True).decode().strip()
        return int(result)


    # function to get range count from mae
    def get_range_count_mae(self, file, forecast, lower, upper):
        cmd = f"cat {file} | grep '^{forecast}' | awk '{{ if ($5 >= {lower} && $5 < {upper}) print $5 }}' | wc -l"
        result = subprocess.check_output(cmd, shell=True).decode().strip()
        return int(result)


    # obtain deciles for RMSE and MAE
    def obtain_quantiles_rmse_mae(self, path):

        # get file information
        arg = path
        lines_in_file = subprocess.check_output(["wc", arg]).decode().split()
        total_with_head = int(lines_in_file[0])
        total = total_with_head - 2
        total_by_forecast = total // 8
        file = arg[:-6]
        output_file = file + ".deciles"
        lookback = file[-2:]

        # get deciles from rmse
        range_12_rmse_1  = self.get_range_count_rmse(arg, "12", 0, 0.05)
        range_12_rmse_2  = self.get_range_count_rmse(arg, "12", 0.05, 0.10)
        range_12_rmse_3  = self.get_range_count_rmse(arg, "12", 0.10, 0.15)
        range_12_rmse_4  = self.get_range_count_rmse(arg, "12", 0.15, 0.20)
        range_12_rmse_5  = self.get_range_count_rmse(arg, "12", 0.20, 0.25)
        range_12_rmse_6  = self.get_range_count_rmse(arg, "12", 0.25, 0.30)
        range_12_rmse_7  = self.get_range_count_rmse(arg, "12", 0.30, 0.35)
        range_12_rmse_8  = self.get_range_count_rmse(arg, "12", 0.35, 0.40)
        range_12_rmse_9  = self.get_range_count_rmse(arg, "12", 0.40, 0.45)
        range_12_rmse_10 = self.get_range_count_rmse(arg, "12", 0.45, 0.50)
        range_12_rmse_11 = self.get_range_count_rmse(arg, "12", 0.50, 1.00)

        range_24_rmse_1  = self.get_range_count_rmse(arg, "24", 0, 0.05)
        range_24_rmse_2  = self.get_range_count_rmse(arg, "24", 0.05, 0.10)
        range_24_rmse_3  = self.get_range_count_rmse(arg, "24", 0.10, 0.15)
        range_24_rmse_4  = self.get_range_count_rmse(arg, "24", 0.15, 0.20)
        range_24_rmse_5  = self.get_range_count_rmse(arg, "24", 0.20, 0.25)
        range_24_rmse_6  = self.get_range_count_rmse(arg, "24", 0.25, 0.30)
        range_24_rmse_7  = self.get_range_count_rmse(arg, "24", 0.30, 0.35)
        range_24_rmse_8  = self.get_range_count_rmse(arg, "24", 0.35, 0.40)
        range_24_rmse_9  = self.get_range_count_rmse(arg, "24", 0.40, 0.45)
        range_24_rmse_10 = self.get_range_count_rmse(arg, "24", 0.45, 0.50)
        range_24_rmse_11 = self.get_range_count_rmse(arg, "24", 0.50, 1.00)

        range_48_rmse_1  = self.get_range_count_rmse(arg, "48", 0, 0.05)
        range_48_rmse_2  = self.get_range_count_rmse(arg, "48", 0.05, 0.10)
        range_48_rmse_3  = self.get_range_count_rmse(arg, "48", 0.10, 0.15)
        range_48_rmse_4  = self.get_range_count_rmse(arg, "48", 0.15, 0.20)
        range_48_rmse_5  = self.get_range_count_rmse(arg, "48", 0.20, 0.25)
        range_48_rmse_6  = self.get_range_count_rmse(arg, "48", 0.25, 0.30)
        range_48_rmse_7  = self.get_range_count_rmse(arg, "48", 0.30, 0.35)
        range_48_rmse_8  = self.get_range_count_rmse(arg, "48", 0.35, 0.40)
        range_48_rmse_9  = self.get_range_count_rmse(arg, "48", 0.40, 0.45)
        range_48_rmse_10 = self.get_range_count_rmse(arg, "48", 0.45, 0.50)
        range_48_rmse_11 = self.get_range_count_rmse(arg, "48", 0.50, 1.00)

        range_96_rmse_1  = self.get_range_count_rmse(arg, "96", 0, 0.05)
        range_96_rmse_2  = self.get_range_count_rmse(arg, "96", 0.05, 0.10)
        range_96_rmse_3  = self.get_range_count_rmse(arg, "96", 0.10, 0.15)
        range_96_rmse_4  = self.get_range_count_rmse(arg, "96", 0.15, 0.20)
        range_96_rmse_5  = self.get_range_count_rmse(arg, "96", 0.20, 0.25)
        range_96_rmse_6  = self.get_range_count_rmse(arg, "96", 0.25, 0.30)
        range_96_rmse_7  = self.get_range_count_rmse(arg, "96", 0.30, 0.35)
        range_96_rmse_8  = self.get_range_count_rmse(arg, "96", 0.35, 0.40)
        range_96_rmse_9  = self.get_range_count_rmse(arg, "96", 0.40, 0.45)
        range_96_rmse_10 = self.get_range_count_rmse(arg, "96", 0.45, 0.50)
        range_96_rmse_11 = self.get_range_count_rmse(arg, "96", 0.50, 1.00)

        range_192_rmse_1  = self.get_range_count_rmse(arg, "192", 0, 0.05)
        range_192_rmse_2  = self.get_range_count_rmse(arg, "192", 0.05, 0.10)
        range_192_rmse_3  = self.get_range_count_rmse(arg, "192", 0.10, 0.15)
        range_192_rmse_4  = self.get_range_count_rmse(arg, "192", 0.15, 0.20)
        range_192_rmse_5  = self.get_range_count_rmse(arg, "192", 0.20, 0.25)
        range_192_rmse_6  = self.get_range_count_rmse(arg, "192", 0.25, 0.30)
        range_192_rmse_7  = self.get_range_count_rmse(arg, "192", 0.30, 0.35)
        range_192_rmse_8  = self.get_range_count_rmse(arg, "192", 0.35, 0.40)
        range_192_rmse_9  = self.get_range_count_rmse(arg, "192", 0.40, 0.45)
        range_192_rmse_10 = self.get_range_count_rmse(arg, "192", 0.45, 0.50)
        range_192_rmse_11 = self.get_range_count_rmse(arg, "192", 0.50, 1.00)

        range_384_rmse_1  = self.get_range_count_rmse(arg, "384", 0, 0.05)
        range_384_rmse_2  = self.get_range_count_rmse(arg, "384", 0.05, 0.10)
        range_384_rmse_3  = self.get_range_count_rmse(arg, "384", 0.10, 0.15)
        range_384_rmse_4  = self.get_range_count_rmse(arg, "384", 0.15, 0.20)
        range_384_rmse_5  = self.get_range_count_rmse(arg, "384", 0.20, 0.25)
        range_384_rmse_6  = self.get_range_count_rmse(arg, "384", 0.25, 0.30)
        range_384_rmse_7  = self.get_range_count_rmse(arg, "384", 0.30, 0.35)
        range_384_rmse_8  = self.get_range_count_rmse(arg, "384", 0.35, 0.40)
        range_384_rmse_9  = self.get_range_count_rmse(arg, "384", 0.40, 0.45)
        range_384_rmse_10 = self.get_range_count_rmse(arg, "384", 0.45, 0.50)
        range_384_rmse_11 = self.get_range_count_rmse(arg, "384", 0.50, 1.00)

        range_768_rmse_1  = self.get_range_count_rmse(arg, "768", 0, 0.05)
        range_768_rmse_2  = self.get_range_count_rmse(arg, "768", 0.05, 0.10)
        range_768_rmse_3  = self.get_range_count_rmse(arg, "768", 0.10, 0.15)
        range_768_rmse_4  = self.get_range_count_rmse(arg, "768", 0.15, 0.20)
        range_768_rmse_5  = self.get_range_count_rmse(arg, "768", 0.20, 0.25)
        range_768_rmse_6  = self.get_range_count_rmse(arg, "768", 0.25, 0.30)
        range_768_rmse_7  = self.get_range_count_rmse(arg, "768", 0.30, 0.35)
        range_768_rmse_8  = self.get_range_count_rmse(arg, "768", 0.35, 0.40)
        range_768_rmse_9  = self.get_range_count_rmse(arg, "768", 0.40, 0.45)
        range_768_rmse_10 = self.get_range_count_rmse(arg, "768", 0.45, 0.50)
        range_768_rmse_11 = self.get_range_count_rmse(arg, "768", 0.50, 1.00)

        range_1536_rmse_1  = self.get_range_count_rmse(arg, "1536", 0, 0.05)
        range_1536_rmse_2  = self.get_range_count_rmse(arg, "1536", 0.05, 0.10)
        range_1536_rmse_3  = self.get_range_count_rmse(arg, "1536", 0.10, 0.15)
        range_1536_rmse_4  = self.get_range_count_rmse(arg, "1536", 0.15, 0.20)
        range_1536_rmse_5  = self.get_range_count_rmse(arg, "1536", 0.20, 0.25)
        range_1536_rmse_6  = self.get_range_count_rmse(arg, "1536", 0.25, 0.30)
        range_1536_rmse_7  = self.get_range_count_rmse(arg, "1536", 0.30, 0.35)
        range_1536_rmse_8  = self.get_range_count_rmse(arg, "1536", 0.35, 0.40)
        range_1536_rmse_9  = self.get_range_count_rmse(arg, "1536", 0.40, 0.45)
        range_1536_rmse_10 = self.get_range_count_rmse(arg, "1536", 0.45, 0.50)
        range_1536_rmse_11 = self.get_range_count_rmse(arg, "1536", 0.50, 1.00)


        # get deciles from mae
        range_12_mae_1  = self.get_range_count_mae(arg, "12", 0, 0.05)
        range_12_mae_2  = self.get_range_count_mae(arg, "12", 0.05, 0.10)
        range_12_mae_3  = self.get_range_count_mae(arg, "12", 0.10, 0.15)
        range_12_mae_4  = self.get_range_count_mae(arg, "12", 0.15, 0.20)
        range_12_mae_5  = self.get_range_count_mae(arg, "12", 0.20, 0.25)
        range_12_mae_6  = self.get_range_count_mae(arg, "12", 0.25, 0.30)
        range_12_mae_7  = self.get_range_count_mae(arg, "12", 0.30, 0.35)
        range_12_mae_8  = self.get_range_count_mae(arg, "12", 0.35, 0.40)
        range_12_mae_9  = self.get_range_count_mae(arg, "12", 0.40, 0.45)
        range_12_mae_10 = self.get_range_count_mae(arg, "12", 0.45, 0.50)
        range_12_mae_11 = self.get_range_count_mae(arg, "12", 0.50, 1.00)

        range_24_mae_1  = self.get_range_count_mae(arg, "24", 0, 0.05)
        range_24_mae_2  = self.get_range_count_mae(arg, "24", 0.05, 0.10)
        range_24_mae_3  = self.get_range_count_mae(arg, "24", 0.10, 0.15)
        range_24_mae_4  = self.get_range_count_mae(arg, "24", 0.15, 0.20)
        range_24_mae_5  = self.get_range_count_mae(arg, "24", 0.20, 0.25)
        range_24_mae_6  = self.get_range_count_mae(arg, "24", 0.25, 0.30)
        range_24_mae_7  = self.get_range_count_mae(arg, "24", 0.30, 0.35)
        range_24_mae_8  = self.get_range_count_mae(arg, "24", 0.35, 0.40)
        range_24_mae_9  = self.get_range_count_mae(arg, "24", 0.40, 0.45)
        range_24_mae_10 = self.get_range_count_mae(arg, "24", 0.45, 0.50)
        range_24_mae_11 = self.get_range_count_mae(arg, "24", 0.50, 1.00)

        range_48_mae_1  = self.get_range_count_mae(arg, "48", 0, 0.05)
        range_48_mae_2  = self.get_range_count_mae(arg, "48", 0.05, 0.10)
        range_48_mae_3  = self.get_range_count_mae(arg, "48", 0.10, 0.15)
        range_48_mae_4  = self.get_range_count_mae(arg, "48", 0.15, 0.20)
        range_48_mae_5  = self.get_range_count_mae(arg, "48", 0.20, 0.25)
        range_48_mae_6  = self.get_range_count_mae(arg, "48", 0.25, 0.30)
        range_48_mae_7  = self.get_range_count_mae(arg, "48", 0.30, 0.35)
        range_48_mae_8  = self.get_range_count_mae(arg, "48", 0.35, 0.40)
        range_48_mae_9  = self.get_range_count_mae(arg, "48", 0.40, 0.45)
        range_48_mae_10 = self.get_range_count_mae(arg, "48", 0.45, 0.50)
        range_48_mae_11 = self.get_range_count_mae(arg, "48", 0.50, 1.00)

        range_96_mae_1  = self.get_range_count_mae(arg, "96", 0, 0.05)
        range_96_mae_2  = self.get_range_count_mae(arg, "96", 0.05, 0.10)
        range_96_mae_3  = self.get_range_count_mae(arg, "96", 0.10, 0.15)
        range_96_mae_4  = self.get_range_count_mae(arg, "96", 0.15, 0.20)
        range_96_mae_5  = self.get_range_count_mae(arg, "96", 0.20, 0.25)
        range_96_mae_6  = self.get_range_count_mae(arg, "96", 0.25, 0.30)
        range_96_mae_7  = self.get_range_count_mae(arg, "96", 0.30, 0.35)
        range_96_mae_8  = self.get_range_count_mae(arg, "96", 0.35, 0.40)
        range_96_mae_9  = self.get_range_count_mae(arg, "96", 0.40, 0.45)
        range_96_mae_10 = self.get_range_count_mae(arg, "96", 0.45, 0.50)
        range_96_mae_11 = self.get_range_count_mae(arg, "96", 0.50, 1.00)

        range_192_mae_1  = self.get_range_count_mae(arg, "192", 0, 0.05)
        range_192_mae_2  = self.get_range_count_mae(arg, "192", 0.05, 0.10)
        range_192_mae_3  = self.get_range_count_mae(arg, "192", 0.10, 0.15)
        range_192_mae_4  = self.get_range_count_mae(arg, "192", 0.15, 0.20)
        range_192_mae_5  = self.get_range_count_mae(arg, "192", 0.20, 0.25)
        range_192_mae_6  = self.get_range_count_mae(arg, "192", 0.25, 0.30)
        range_192_mae_7  = self.get_range_count_mae(arg, "192", 0.30, 0.35)
        range_192_mae_8  = self.get_range_count_mae(arg, "192", 0.35, 0.40)
        range_192_mae_9  = self.get_range_count_mae(arg, "192", 0.40, 0.45)
        range_192_mae_10 = self.get_range_count_mae(arg, "192", 0.45, 0.50)
        range_192_mae_11 = self.get_range_count_mae(arg, "192", 0.50, 1.00)

        range_384_mae_1  = self.get_range_count_mae(arg, "384", 0, 0.05)
        range_384_mae_2  = self.get_range_count_mae(arg, "384", 0.05, 0.10)
        range_384_mae_3  = self.get_range_count_mae(arg, "384", 0.10, 0.15)
        range_384_mae_4  = self.get_range_count_mae(arg, "384", 0.15, 0.20)
        range_384_mae_5  = self.get_range_count_mae(arg, "384", 0.20, 0.25)
        range_384_mae_6  = self.get_range_count_mae(arg, "384", 0.25, 0.30)
        range_384_mae_7  = self.get_range_count_mae(arg, "384", 0.30, 0.35)
        range_384_mae_8  = self.get_range_count_mae(arg, "384", 0.35, 0.40)
        range_384_mae_9  = self.get_range_count_mae(arg, "384", 0.40, 0.45)
        range_384_mae_10 = self.get_range_count_mae(arg, "384", 0.45, 0.50)
        range_384_mae_11 = self.get_range_count_mae(arg, "384", 0.50, 1.00)

        range_768_mae_1  = self.get_range_count_mae(arg, "768", 0, 0.05)
        range_768_mae_2  = self.get_range_count_mae(arg, "768", 0.05, 0.10)
        range_768_mae_3  = self.get_range_count_mae(arg, "768", 0.10, 0.15)
        range_768_mae_4  = self.get_range_count_mae(arg, "768", 0.15, 0.20)
        range_768_mae_5  = self.get_range_count_mae(arg, "768", 0.20, 0.25)
        range_768_mae_6  = self.get_range_count_mae(arg, "768", 0.25, 0.30)
        range_768_mae_7  = self.get_range_count_mae(arg, "768", 0.30, 0.35)
        range_768_mae_8  = self.get_range_count_mae(arg, "768", 0.35, 0.40)
        range_768_mae_9  = self.get_range_count_mae(arg, "768", 0.40, 0.45)
        range_768_mae_10 = self.get_range_count_mae(arg, "768", 0.45, 0.50)
        range_768_mae_11 = self.get_range_count_mae(arg, "768", 0.50, 1.00)

        range_1536_mae_1  = self.get_range_count_mae(arg, "1536", 0, 0.05)
        range_1536_mae_2  = self.get_range_count_mae(arg, "1536", 0.05, 0.10)
        range_1536_mae_3  = self.get_range_count_mae(arg, "1536", 0.10, 0.15)
        range_1536_mae_4  = self.get_range_count_mae(arg, "1536", 0.15, 0.20)
        range_1536_mae_5  = self.get_range_count_mae(arg, "1536", 0.20, 0.25)
        range_1536_mae_6  = self.get_range_count_mae(arg, "1536", 0.25, 0.30)
        range_1536_mae_7  = self.get_range_count_mae(arg, "1536", 0.30, 0.35)
        range_1536_mae_8  = self.get_range_count_mae(arg, "1536", 0.35, 0.40)
        range_1536_mae_9  = self.get_range_count_mae(arg, "1536", 0.40, 0.45)
        range_1536_mae_10 = self.get_range_count_mae(arg, "1536", 0.45, 0.50)
        range_1536_mae_11 = self.get_range_count_mae(arg, "1536", 0.50, 1.00)

        # print deciles
        with open(output_file, "w") as file:
            file.write("lookback forecast quantile total RMSE MAE\n")
            file.write("-----------------------------------------\n")
            file.write(f"{lookback} 12 0.00-0.05 {total_by_forecast} {range_12_rmse_1} {range_12_mae_1}\n")
            file.write(f"{lookback} 12 0.05-0.10 {total_by_forecast} {range_12_rmse_2} {range_12_mae_2}\n")
            file.write(f"{lookback} 12 0.10-0.15 {total_by_forecast} {range_12_rmse_3} {range_12_mae_3}\n")
            file.write(f"{lookback} 12 0.15-0.20 {total_by_forecast} {range_12_rmse_4} {range_12_mae_4}\n")
            file.write(f"{lookback} 12 0.20-0.25 {total_by_forecast} {range_12_rmse_5} {range_12_mae_5}\n")
            file.write(f"{lookback} 12 0.25-0.30 {total_by_forecast} {range_12_rmse_6} {range_12_mae_6}\n")
            file.write(f"{lookback} 12 0.30-0.35 {total_by_forecast} {range_12_rmse_7} {range_12_mae_7}\n")
            file.write(f"{lookback} 12 0.35-0.40 {total_by_forecast} {range_12_rmse_8} {range_12_mae_8}\n")
            file.write(f"{lookback} 12 0.40-0.45 {total_by_forecast} {range_12_rmse_9} {range_12_mae_9}\n")
            file.write(f"{lookback} 12 0.45-0.50 {total_by_forecast} {range_12_rmse_10} {range_12_mae_10}\n")
            file.write(f"{lookback} 12 0.50-1.00 {total_by_forecast} {range_12_rmse_11} {range_12_mae_11}\n")
            file.write("\n")

            file.write(f"{lookback} 24 0.00-0.05 {total_by_forecast} {range_24_rmse_1} {range_24_mae_1}\n")
            file.write(f"{lookback} 24 0.05-0.10 {total_by_forecast} {range_24_rmse_2} {range_24_mae_2}\n")
            file.write(f"{lookback} 24 0.10-0.15 {total_by_forecast} {range_24_rmse_3} {range_24_mae_3}\n")
            file.write(f"{lookback} 24 0.15-0.20 {total_by_forecast} {range_24_rmse_4} {range_24_mae_4}\n")
            file.write(f"{lookback} 24 0.20-0.25 {total_by_forecast} {range_24_rmse_5} {range_24_mae_5}\n")
            file.write(f"{lookback} 24 0.25-0.30 {total_by_forecast} {range_24_rmse_6} {range_24_mae_6}\n")
            file.write(f"{lookback} 24 0.30-0.35 {total_by_forecast} {range_24_rmse_7} {range_24_mae_7}\n")
            file.write(f"{lookback} 24 0.35-0.40 {total_by_forecast} {range_24_rmse_8} {range_24_mae_8}\n")
            file.write(f"{lookback} 24 0.40-0.45 {total_by_forecast} {range_24_rmse_9} {range_24_mae_9}\n")
            file.write(f"{lookback} 24 0.45-0.50 {total_by_forecast} {range_24_rmse_10} {range_24_mae_10}\n")
            file.write(f"{lookback} 24 0.50-1.00 {total_by_forecast} {range_24_rmse_11} {range_24_mae_11}\n")
            file.write("\n")

            file.write(f"{lookback} 48 0.00-0.05 {total_by_forecast} {range_48_rmse_1} {range_48_mae_1}\n")
            file.write(f"{lookback} 48 0.05-0.10 {total_by_forecast} {range_48_rmse_2} {range_48_mae_2}\n")
            file.write(f"{lookback} 48 0.10-0.15 {total_by_forecast} {range_48_rmse_3} {range_48_mae_3}\n")
            file.write(f"{lookback} 48 0.15-0.20 {total_by_forecast} {range_48_rmse_4} {range_48_mae_4}\n")
            file.write(f"{lookback} 48 0.20-0.25 {total_by_forecast} {range_48_rmse_5} {range_48_mae_5}\n")
            file.write(f"{lookback} 48 0.25-0.30 {total_by_forecast} {range_48_rmse_6} {range_48_mae_6}\n")
            file.write(f"{lookback} 48 0.30-0.35 {total_by_forecast} {range_48_rmse_7} {range_48_mae_7}\n")
            file.write(f"{lookback} 48 0.35-0.40 {total_by_forecast} {range_48_rmse_8} {range_48_mae_8}\n")
            file.write(f"{lookback} 48 0.40-0.45 {total_by_forecast} {range_48_rmse_9} {range_48_mae_9}\n")
            file.write(f"{lookback} 48 0.45-0.50 {total_by_forecast} {range_48_rmse_10} {range_48_mae_10}\n")
            file.write(f"{lookback} 48 0.50-1.00 {total_by_forecast} {range_48_rmse_11} {range_48_mae_11}\n")
            file.write("\n")

            file.write(f"{lookback} 96 0.00-0.05 {total_by_forecast} {range_96_rmse_1} {range_96_mae_1}\n")
            file.write(f"{lookback} 96 0.05-0.10 {total_by_forecast} {range_96_rmse_2} {range_96_mae_2}\n")
            file.write(f"{lookback} 96 0.10-0.15 {total_by_forecast} {range_96_rmse_3} {range_96_mae_3}\n")
            file.write(f"{lookback} 96 0.15-0.20 {total_by_forecast} {range_96_rmse_4} {range_96_mae_4}\n")
            file.write(f"{lookback} 96 0.20-0.25 {total_by_forecast} {range_96_rmse_5} {range_96_mae_5}\n")
            file.write(f"{lookback} 96 0.25-0.30 {total_by_forecast} {range_96_rmse_6} {range_96_mae_6}\n")
            file.write(f"{lookback} 96 0.30-0.35 {total_by_forecast} {range_96_rmse_7} {range_96_mae_7}\n")
            file.write(f"{lookback} 96 0.35-0.40 {total_by_forecast} {range_96_rmse_8} {range_96_mae_8}\n")
            file.write(f"{lookback} 96 0.40-0.45 {total_by_forecast} {range_96_rmse_9} {range_96_mae_9}\n")
            file.write(f"{lookback} 96 0.45-0.50 {total_by_forecast} {range_96_rmse_10} {range_96_mae_10}\n")
            file.write(f"{lookback} 96 0.50-1.00 {total_by_forecast} {range_96_rmse_11} {range_96_mae_11}\n")
            file.write("\n")

            file.write(f"{lookback} 192 0.00-0.05 {total_by_forecast} {range_192_rmse_1} {range_192_mae_1}\n")
            file.write(f"{lookback} 192 0.05-0.10 {total_by_forecast} {range_192_rmse_2} {range_192_mae_2}\n")
            file.write(f"{lookback} 192 0.10-0.15 {total_by_forecast} {range_192_rmse_3} {range_192_mae_3}\n")
            file.write(f"{lookback} 192 0.15-0.20 {total_by_forecast} {range_192_rmse_4} {range_192_mae_4}\n")
            file.write(f"{lookback} 192 0.20-0.25 {total_by_forecast} {range_192_rmse_5} {range_192_mae_5}\n")
            file.write(f"{lookback} 192 0.25-0.30 {total_by_forecast} {range_192_rmse_6} {range_192_mae_6}\n")
            file.write(f"{lookback} 192 0.30-0.35 {total_by_forecast} {range_192_rmse_7} {range_192_mae_7}\n")
            file.write(f"{lookback} 192 0.35-0.40 {total_by_forecast} {range_192_rmse_8} {range_192_mae_8}\n")
            file.write(f"{lookback} 192 0.40-0.45 {total_by_forecast} {range_192_rmse_9} {range_192_mae_9}\n")
            file.write(f"{lookback} 192 0.45-0.50 {total_by_forecast} {range_192_rmse_10} {range_192_mae_10}\n")
            file.write(f"{lookback} 192 0.50-1.00 {total_by_forecast} {range_192_rmse_11} {range_192_mae_11}\n")
            file.write("\n")

            file.write(f"{lookback} 384 0.00-0.05 {total_by_forecast} {range_384_rmse_1} {range_384_mae_1}\n")
            file.write(f"{lookback} 384 0.05-0.10 {total_by_forecast} {range_384_rmse_2} {range_384_mae_2}\n")
            file.write(f"{lookback} 384 0.10-0.15 {total_by_forecast} {range_384_rmse_3} {range_384_mae_3}\n")
            file.write(f"{lookback} 384 0.15-0.20 {total_by_forecast} {range_384_rmse_4} {range_384_mae_4}\n")
            file.write(f"{lookback} 384 0.20-0.25 {total_by_forecast} {range_384_rmse_5} {range_384_mae_5}\n")
            file.write(f"{lookback} 384 0.25-0.30 {total_by_forecast} {range_384_rmse_6} {range_384_mae_6}\n")
            file.write(f"{lookback} 384 0.30-0.35 {total_by_forecast} {range_384_rmse_7} {range_384_mae_7}\n")
            file.write(f"{lookback} 384 0.35-0.40 {total_by_forecast} {range_384_rmse_8} {range_384_mae_8}\n")
            file.write(f"{lookback} 384 0.40-0.45 {total_by_forecast} {range_384_rmse_9} {range_384_mae_9}\n")
            file.write(f"{lookback} 384 0.45-0.50 {total_by_forecast} {range_384_rmse_10} {range_384_mae_10}\n")
            file.write(f"{lookback} 384 0.50-1.00 {total_by_forecast} {range_384_rmse_11} {range_384_mae_11}\n")
            file.write("\n")

            file.write(f"{lookback} 768 0.00-0.05 {total_by_forecast} {range_768_rmse_1} {range_768_mae_1}\n")
            file.write(f"{lookback} 768 0.05-0.10 {total_by_forecast} {range_768_rmse_2} {range_768_mae_2}\n")
            file.write(f"{lookback} 768 0.10-0.15 {total_by_forecast} {range_768_rmse_3} {range_768_mae_3}\n")
            file.write(f"{lookback} 768 0.15-0.20 {total_by_forecast} {range_768_rmse_4} {range_768_mae_4}\n")
            file.write(f"{lookback} 768 0.20-0.25 {total_by_forecast} {range_768_rmse_5} {range_768_mae_5}\n")
            file.write(f"{lookback} 768 0.25-0.30 {total_by_forecast} {range_768_rmse_6} {range_768_mae_6}\n")
            file.write(f"{lookback} 768 0.30-0.35 {total_by_forecast} {range_768_rmse_7} {range_768_mae_7}\n")
            file.write(f"{lookback} 768 0.35-0.40 {total_by_forecast} {range_768_rmse_8} {range_768_mae_8}\n")
            file.write(f"{lookback} 768 0.40-0.45 {total_by_forecast} {range_768_rmse_9} {range_768_mae_9}\n")
            file.write(f"{lookback} 768 0.45-0.50 {total_by_forecast} {range_768_rmse_10} {range_768_mae_10}\n")
            file.write(f"{lookback} 768 0.50-1.00 {total_by_forecast} {range_768_rmse_11} {range_768_mae_11}\n")
            file.write("\n")

            file.write(f"{lookback} 1536 0.00-0.05 {total_by_forecast} {range_1536_rmse_1} {range_1536_mae_1}\n")
            file.write(f"{lookback} 1536 0.05-0.10 {total_by_forecast} {range_1536_rmse_2} {range_1536_mae_2}\n")
            file.write(f"{lookback} 1536 0.10-0.15 {total_by_forecast} {range_1536_rmse_3} {range_1536_mae_3}\n")
            file.write(f"{lookback} 1536 0.15-0.20 {total_by_forecast} {range_1536_rmse_4} {range_1536_mae_4}\n")
            file.write(f"{lookback} 1536 0.20-0.25 {total_by_forecast} {range_1536_rmse_5} {range_1536_mae_5}\n")
            file.write(f"{lookback} 1536 0.25-0.30 {total_by_forecast} {range_1536_rmse_6} {range_1536_mae_6}\n")
            file.write(f"{lookback} 1536 0.30-0.35 {total_by_forecast} {range_1536_rmse_7} {range_1536_mae_7}\n")
            file.write(f"{lookback} 1536 0.35-0.40 {total_by_forecast} {range_1536_rmse_8} {range_1536_mae_8}\n")
            file.write(f"{lookback} 1536 0.40-0.45 {total_by_forecast} {range_1536_rmse_9} {range_1536_mae_9}\n")
            file.write(f"{lookback} 1536 0.45-0.50 {total_by_forecast} {range_1536_rmse_10} {range_1536_mae_10}\n")
            file.write(f"{lookback} 1536 0.50-1.00 {total_by_forecast} {range_1536_rmse_11} {range_1536_mae_11}\n")
        

