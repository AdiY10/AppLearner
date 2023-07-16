import matplotlib.pyplot as plt
import src.framework__data_set as ds
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split

tf.random.set_seed(0)


class LSTM:
    def __init__(self, metric="container_cpu", app="collector", path_to_data="../data/"):
        # Set parameters
        self.app_to_test = app
        self.file = app

        # Extract dataset according to parametes
        self.dataset = ds.get_data_set(
            metric=metric,
            application_name=self.app_to_test,
            path_to_data=path_to_data
        )
        # In case of Using GPU
        self.use_cuda = True

    def train_model(self):
        # Set model parameters
        dropout_rate = 0.2
        lookback = 12
        horizon = 12
        Train_size = int(
            len(self.dataset) * 0.7)  # Data is serialized, no need to split data. But test set will be the 30% left

        # Initializing X and Y lists
        X = []
        Y = []

        # Normalize data
        scaler_train = MinMaxScaler()
        for i in range(0, int(len(self.dataset))):
            self.dataset[i]['sample'] = scaler_train.fit_transform(self.dataset[i][['sample']])

        # Defining the sequential model
        model = Sequential()
        model.add(LSTM(units=64, return_sequences=True, input_shape=(n_lookback, 1)))
        model.add(Dropout(dropout_rate))  # Adding dropout for regularization
        model.add(LSTM(units=64, return_sequences=False))
        model.add(Dropout(dropout_rate))  # Adding dropout for regularization
        model.add(LSTM(units=64))
        model.add(Dense(units=horizon))

        # In case of using GPU
        if self.use_cuda:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

        model.compile(loss='mse', optimizer='adam')

        for i in range(0, Train_size):

            data_train = self.dataset[i]

            # fill N/A values
            y = data_train['sample'].fillna(method='ffill')
            y = y.values.reshape(-1, 1)

            if lookback >= (len(y) - horizon + 1):
                continue

            # Creating X and Y lists, with a sliding window
            for j in range(lookback, len(y) - horizon + 1):
                X.append(y[j - lookback: j])
                Y.append(y[j: j + horizon])

        X = np.array(X)
        Y = np.array(Y)

        # Training the model
        model.fit(X, Y, epochs=100, batch_size=128, verbose=2)

    def save_model(self, model):
        # Save trained model
        model.save('collector_12min.h5')

        # Serialize model to JSON
        model_json = model.to_json()
        with open("collector_12min.json", "w") as json_file:
            json_file.write(model_json)

        # Serialize weights to HDF5
        model.save_weights("collector_12min.h5")
        print("Model saved to disk")

    def predict_values(self, sys3, sys4, dataset_test, file):
        # Input and output sequences
        lookback = int(sys3)
        forecast = int(sys4)

        # Load json
        json_file = open(file + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # Load weights
        model.load_weights(file + ".h5")
        print("Loaded model " + file + " from disk")

        # Scale data
        scaler_test = MinMaxScaler()
        dataset_test['sample'] = scaler_test.fit_transform(dataset_test[['sample']])

        # fill N/A values
        y = dataset_test['sample'].fillna(method='ffill')
        y = y.values.reshape(-1, 1)
        test_len = lookback
        data_test = dataset_test.head(test_len)
        len_data = len(dataset_test['sample'].to_numpy())

        # Make predictions
        X_ = y[test_len - lookback:test_len]
        X_ = X_.reshape(1, lookback, 1)
        Y_ = model.predict(X_).reshape(-1, 1)
        Y_pred = Y_
        # When longer than test set
        if len(Y_pred) > len_data:
            Y_pred = Y_[0:len_data - lookback]

        # Append forecasts
        for i in range(int((len(dataset_test) - len(data_test)) / forecast) - 0):
            X2_ = y[test_len - lookback + (forecast * (i + 1)):test_len + (forecast * (i + 1))]
            X2_ = X2_.reshape(1, lookback, 1)
            Y2_ = model.predict(X2_).reshape(-1, 1)

            if i == int((len(dataset_test) - len(data_test)) / forecast) - 1:
                until = len_data - len(Y_pred) - lookback
                Y_pred = np.concatenate((Y_pred, Y2_[0:until]))
            else:
                Y_pred = np.concatenate((Y_pred, Y2_))

        forecast_len = len(Y_pred)

        # For calculating RMSE
        Y3_ = Y_pred.flatten()
        Y4_ = dataset_test['sample'].to_numpy()

        # Root mean squared error
        rmse = np.sqrt(np.square(np.subtract(Y3_, Y4_[test_len:test_len + forecast_len])).mean())

        # MAE
        Y3_ = Y_pred.flatten()
        Y4_ = dataset_test['sample'].to_numpy()
        mae = np.mean(np.abs(np.subtract(Y4_[test_len:test_len + forecast_len], Y3_)))

        # Unnormalize data to original scale
        Y_pred = scaler_test.inverse_transform(Y_pred)
        dataset_test['sample'] = scaler_test.inverse_transform(dataset_test[['sample']])

        return Y_pred, rmse, mae, test_len, forecast_len, len_data

    def run_LSTM(self):
        # Get intervals
        number = int("12") % int(len(self.dataset) * 0.2)
        # Forecast interval selected (testing part)
        dataset_to_test = self.dataset[int(len(self.dataset) * 0.8) + number:]
        # dataset_to_test = self.dataset

        counter = 1  # Count number of plots
        for dataset_test in dataset_to_test:
            original_as_series = dataset_test['sample'].copy()
            x_axis = [time for time in dataset_test["time"]]
            original_as_series.index = x_axis
            ax = original_as_series.plot(color="blue", label="Sample", linewidth=1)

            horizons = ["12"]  # Horizons to present on the graph
            colors = ["red"]  # present each horizon with different color
            rmse = []
            mae = []

            # Inference
            Y3_, rmse_, mae_, test_len, forecast_len, len_data = self.predict_values("12", horizons[0], dataset_test,
                                                                                     self.file)
            # Present predictions with plots
            df_future = pd.DataFrame(columns=['Forecast'])
            df_future['Forecast'] = Y3_.flatten()
            predicted_as_series = df_future['Forecast']
            predicted_as_series.index = x_axis[test_len:test_len + forecast_len]
            predicted_as_series.plot(ax=ax, color=colors[0], label="Horizon: " + horizons[0], linewidth=1)
            rmse.append(rmse_)
            mae.append(mae_)

            # Title
            plt.suptitle(
                "Metric: " + "container_cpu" + ", App: " + self.app_to_test + ", Lookback: " + "12" + ", Horizon: 12\nRMSE: " + str(
                    "{:.5f}".format(rmse[0])) + " - MAE: " + str("{:.5f}".format(mae[0])))

            # Plot features
            plt.legend(ncol=2, loc="lower left")
            plt.xlabel("Time (Hours)")
            plt.ylabel("Values")
            plt.subplots_adjust(top=0.85)
            plt.subplots_adjust(bottom=0.1)
            plt.show()

            # Number of plots to present
            if counter >= 10:
                break
            counter += 1
