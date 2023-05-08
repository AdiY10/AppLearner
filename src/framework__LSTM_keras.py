import matplotlib.pyplot as plt
import src.framework__data_set as ds
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None
tf.random.set_seed(0)

class LSTM:
    def __init__(self, metric="container_cpu", app="collector", path_to_data="../data/"):
        # prepare parameters
        self.app_to_test = app
        self.file = "collector"
        # read dataset
        self.dataset = ds.get_data_set(
            metric=metric,
            application_name=self.app_to_test,
            path_to_data=path_to_data
        )
    def train_model(self):
        # Normalize data
        scaler = MinMaxScaler()
        for i in range(len(self.dataset)):
            self.dataset[i]['sample'] = scaler.fit_transform(self.dataset[i][['sample']])

        # generate input and output sequences
        n_lookback = 12
        n_forecast = 12

        # build LSTM model
        model = Sequential()
        model.add(LSTM(units=64, return_sequences=True, input_shape=(n_lookback, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=n_forecast))

        # model compilation
        model.compile(loss='mse', optimizer='adam')

        # Train model for each sequence in dataset
        for df in self.dataset:
            # Create input and output sequences
            X, y = [], []
            for i in range(len(df) - n_lookback - n_forecast + 1):
                X.append(df.iloc[i:i + n_lookback, 0].values.reshape(-1, 1))
                y.append(df.iloc[i + n_lookback:i + n_lookback + n_forecast, 0].values.reshape(-1, 1))
            X = np.array(X)
            y = np.array(y)

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # train the model
            history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)

        # evaluate the model
        score = model.evaluate(X_test, y_test, verbose=0)
        print(f'Test loss: {score}')

    def save_model(self, model):
        # save trained model
        model.save('collector_12_12.h5')

        # serialize model to JSON
        model_json = model.to_json()
        with open("collector_12_12.json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights("collector_12_12.h5")
        print("Model saved to disk")

    def predict_values(self, sys3, sys4, dataset_test, file):
        # input and output sequences
        n_lookback = int(sys3)  # length of input sequences (lookback period)
        n_forecast = int(sys4)  # length of output sequences (forecast period)

        # trained model file to load
        # file = sys.argv[2]+"_"+sys3+"_"+sys4
        # file = "check2"
        # load json and create model
        json_file = open(file + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(file + ".h5")
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
        X_ = y[test_len - n_lookback:test_len]
        # if X_ < n_lookback:
        #     break
        X_ = X_.reshape(1, n_lookback, 1)
        Y_ = model.predict(X_).reshape(-1, 1)
        Y3_ = []
        Y3_ = np.array(Y3_)
        Y3_ = Y_
        # cut first prediction when longer than test set
        if len(Y3_) > len_data:
            Y3_ = Y_[0:len_data - n_lookback]

        # append forecasts
        for i in range(int((len(dataset_test) - len(data_test)) / n_forecast) - 0):
            X2_ = y[test_len - n_lookback + (n_forecast * (i + 1)):test_len + (n_forecast * (i + 1))]
            X2_ = X2_.reshape(1, n_lookback, 1)
            Y2_ = model.predict(X2_).reshape(-1, 1)
            # cut last append
            if i == int((len(dataset_test) - len(data_test)) / n_forecast) - 1:
                until = len_data - len(Y3_) - n_lookback
                Y3_ = np.concatenate((Y3_, Y2_[0:until]))
            else:
                Y3_ = np.concatenate((Y3_, Y2_))

        forecast_len = len(Y3_)

        # For calculating RMSE
        Y4_ = Y3_.flatten()
        Y5_ = dataset_test['sample'].to_numpy()

        # mean squared error
        rmse = np.sqrt(np.square(np.subtract(Y4_, Y5_[test_len:test_len + forecast_len])).mean())

        # MAE
        Y4_ = Y3_.flatten()
        Y5_ = dataset_test['sample'].to_numpy()
        mae = np.mean(np.abs(np.subtract(Y5_[test_len:test_len + forecast_len], Y4_)))

        # Recover from normalized data
        Y3_ = scaler_test.inverse_transform(Y3_)
        dataset_test['sample'] = scaler_test.inverse_transform(dataset_test[['sample']])

        return Y3_, rmse, mae, test_len, forecast_len, len_data

    def run_LSTM(self):
        # # get interval
        number = int("12") % int(len(self.dataset) * 0.2)
        #
        # # forecast interval selected (testing part)
        dataset_to_test = self.dataset[int(len(self.dataset) * 0.8) + number:]
        # dataset_to_test = self.dataset

        # samples
        counter = 1  # count number of plots
        for dataset_test in dataset_to_test:
            original_as_series = dataset_test['sample'].copy()
            x_axis = [time for time in dataset_test["time"]]
            original_as_series.index = x_axis
            ax = original_as_series.plot(color="blue", label="Sample", linewidth=1)

            # lists
            horizons = ["12"]
            colors = ["red"]
            rmse = []
            mae = []

            Y3_, rmse_, mae_, test_len, forecast_len, len_data = self.predict_values("12", horizons[0], dataset_test,
                                                                                     self.file)
            if rmse_ < 0.08:
                # plot results
                df_future = pd.DataFrame(columns=['Forecast'])
                df_future['Forecast'] = Y3_.flatten()
                predicted_as_series = df_future['Forecast']
                predicted_as_series.index = x_axis[test_len:test_len + forecast_len]
                predicted_as_series.plot(ax=ax, color=colors[0], label="Horizon: " + horizons[0], linewidth=1)
                rmse.append(rmse_)
                mae.append(mae_)
                # title
                plt.suptitle(
                    "Metric: " + "container_cpu" + ", App: " + self.app_to_test + ", Lookback: " + "12" + ", Horizon: 12\nRMSE: " + str(
                        "{:.5f}".format(rmse[0])) + " - MAE: " + str("{:.5f}".format(mae[0])))
                # plot features
                plt.legend(ncol=2, loc="lower left")
                plt.xlabel("Time (Hours)")
                plt.ylabel("Values")
                plt.subplots_adjust(top=0.85)
                plt.subplots_adjust(bottom=0.1)
                plt.show()
                if counter >= 10:
                    break
                counter += 1
