from abc import ABC
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from BatchProcessing import BatchProcessing


class ClinicalTrialsDataModule(pl.LightningDataModule, ABC):
    def __init__(
        self,
        train_batch_size: Optional[int] = None,
        eval_batch_size: Optional[int] = 16,
        n_train_samples: int = 1024,
        n_val_samples: Optional[int] = None,
        n_test_samples: Optional[int] = None,
        mode: str = "train",
    ):
        super().__init__()
        if mode == "train":
            self.expected_batches = n_train_samples / train_batch_size
            batch_processing = BatchProcessing(
                train_batch_size=train_batch_size,
                n_val_samples=n_val_samples,
                n_test_samples=n_test_samples,
            )

            train_sample_size = int(
                len(batch_processing.train) // self.expected_batches
            )

            self.train_data = list(batch_processing.train)
            self.val_data = list(batch_processing.val)
            self.test_data = list(batch_processing.test)

            self.train_batch_size = train_sample_size
            self.test_batch_size = eval_batch_size

            self.train_batch_processing = batch_processing.build_train_batch
            self.val_batch_processing = batch_processing.build_val_batch
            self.eval_batch_processing = batch_processing.build_test_batch
        elif mode == "prediction":
            # TODO how to handle data for predictions?
            #  it can be input queries not loaded to the db? but they have different fields
            self.pred_data = list(batch_processing.pred)
            self.pred_batch_size = eval_batch_size
            self.pred_batch_processing = batch_processing.build_pred_batch
            self.pred_samples = batch_processing.pred

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.train_batch_size,
            shuffle=False,
            collate_fn=self.train_batch_processing,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.test_batch_size,
            collate_fn=self.val_batch_processing,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.test_batch_size,
            collate_fn=self.eval_batch_processing,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_data,
            batch_size=self.pred_batch_size,
            collate_fn=self.pred_batch_processing,
        )