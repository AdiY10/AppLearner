import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
from torch.utils.data import DataLoader
from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import SMAPE, MultivariateNormalDistributionLoss, NormalDistributionLoss






class DeepARTester:
    def __init__(self,training):
        # prepare parameters
        self.__msg = "[DeepARTester]"
        self.__model = DeepAR.from_dataset(
            training,
            learning_rate=0.1,
            log_interval=10,
            log_val_interval=1,
            hidden_size=30,
            rnn_layers=2,
            loss=NormalDistributionLoss(),
        )
        self.__best_model = None
        # self.__model_input_length = length_of_shortest_time_series // 2
        # self.__model = LSTMPredictor(
        #     input_size=1,
        #     output_size=1,
        # ).to(pytorch__driver_for_test_bench.get_device())
        # self.__optimizer = optim.Adam(self.__model.parameters(), lr=0.01)
        # self.__best_model = self.__model
        # self.__criterion = nn.MSELoss()
        # print
        # print(self.__msg, f"model = {self.__model}")
        # print(self.__msg, f"optimizer = {self.__optimizer}")
        # print(self.__msg, f"criterion = {self.__criterion}")

    # def __trainer(self):
    #     early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    #     trainer = pl.Trainer(
    #         max_epochs=20,
    #         gpus=0,
    #         enable_model_summary=True,
    #         gradient_clip_val=0.1,
    #         callbacks=[early_stop_callback],
    #         limit_train_batches=50,
    #         enable_checkpointing=True,
    #     )
    #     return trainer
    

    """
    *******************************************************************************************************************
        API functions
    *******************************************************************************************************************
    """  
    def learn_from_data_set(self, train_dataloader,validation_dataloader):
        # self.__best_model = pytorch__driver_for_test_bench.train_neural_network(
        #     training_data_set=training_data_set,
        #     model=self.__model,
        #     num_epochs=30,
        #     model_input_length=self.__model_input_length,
        #     batch_size=64,
        #     criterion=self.__criterion,
        #     optimizer=self.__optimizer
        # )
        # net = self.__model.from_dataset(train_dataloader)
        early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        trainer = pl.Trainer(
            max_epochs=10,
            gpus=0,
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback],
            limit_train_batches=50,
            enable_checkpointing=True,
        )
        trainer.fit(
            self.__model,
            train_dataloaders=train_dataloader,
            val_dataloaders=validation_dataloader
        )
        best_model_path = trainer.checkpoint_callback.best_model_path
        best_model = DeepAR.load_from_checkpoint(best_model_path)
        self.__best_model = best_model
        return best_model #TODO maybe delete

    def get_actuals_and_predictions(self, val_dataloader):
        actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
        predictions = self.__best_model.predict(val_dataloader)
        return actuals, predictions

    def predictions(self,train_dataloader,val_dataloader):
        raw_predictions, x = self.__best_model.predict(val_dataloader, mode="raw", return_x=True, n_samples=100)
        return raw_predictions, x
    
    def plot_predictions(self, raw_predictions, x, validation):
            series = validation.x_to_index(x)["series"]
            for idx in range(20):  # plot 10 examples
                self.__best_model.plot_prediction(x, raw_predictions, idx=idx)
                plt.suptitle(f"Series: {series.iloc[idx]}") 
                plt.show()
    
