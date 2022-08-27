from abc import ABC
from typing import Optional, List
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .BatchProcessing import BatchProcessing


class ClinicalTrialsDataModule(pl.LightningDataModule, ABC):
    def __init__(
        self,
        model_name: str,
        fields: List[str],
        path_to_run: str,
        path_to_qrels: str,
        train_batch_size: Optional[int] = None,
        eval_batch_size: Optional[int] = 16,
        n_train_samples: int = 1024,
        n_val_samples: Optional[int] = None,
        n_test_samples: Optional[int] = None,
        mode: str = "train"
    ):
        """
        :param train_batch_size: number of examples used on each training step
        :param eval_batch_size: number of examples used on each validation, testing and prediction step
        :param n_train_samples: for restricting the number of samples used per epoch
        :param n_val_samples: for restricting the validation to the top n results of each run
        :param n_test_samples: for restricting the evaluation to the top n results of each run
        :param mode: values["train", "predict_w_labels", "pred_w_no_labels"] required to
        define which kind of process the data module is used for.
        """

        super().__init__()

        batch_processing = BatchProcessing(
            fields=fields,
            train_batch_size=train_batch_size,
            n_val_samples=n_val_samples,
            n_test_samples=n_test_samples,
            mode=mode,
            tokenizer_name=model_name,
            path_to_run=path_to_run,
            path_to_qrels=path_to_qrels
        )

        if mode in ["train"]:

            self.n_training_steps = n_train_samples // train_batch_size

            self.train_pool_batch_size = int(
                batch_processing.data_train.__len__() // self.n_training_steps
            )

            self.data_train = batch_processing.data_train
            self.data_val = batch_processing.data_val
            self.data_test = batch_processing.data_test

            self.eval_batch_size = eval_batch_size

            self.train_batch_processing = batch_processing.build_train_batch
            self.eval_batch_processing = batch_processing.build_batch

        elif mode in ["predict_w_labels", "pred_w_no_labels"]:
            self.data_test = batch_processing.data
            self.pred_batch_size = eval_batch_size
            self.pred_batch_processing = batch_processing.build_batch

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.train_pool_batch_size,
            shuffle=True,
            collate_fn=self.train_batch_processing,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.eval_batch_size,
            collate_fn=self.eval_batch_processing
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.eval_batch_size,
            collate_fn=self.eval_batch_processing,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.pred_batch_size,
            collate_fn=self.pred_batch_processing,
        )
