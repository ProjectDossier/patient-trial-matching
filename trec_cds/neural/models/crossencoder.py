from abc import ABC

import ir_measures
import numpy as np
import pytorch_lightning as pl
from ir_measures import *
from torch import nn, split
from transformers import AdamW, AutoConfig, AutoModel, get_linear_schedule_with_warmup

from trec_cds.neural.utils.evaluator import Evaluator
from trec_cds.neural.utils.loss import PairwiseHingeLoss


class CrossEncoder(pl.LightningModule, ABC):
    def __init__(
        self,
        model_name: str,
        num_labels: int = None,
        n_training_steps: int = None,
        n_warmup_steps: int = None,
        batch_size: int = 16,
        optimization_metric: str = P @ 10,
        mode: str = "train",
        evaluator=None,
    ):
        super().__init__()
        self.n_training_steps = n_training_steps
        self.batch_size = batch_size
        self.n_warmup_steps = n_warmup_steps
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)

        if num_labels is not None:
            self.config.num_labels = num_labels - 1

        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)

        try:
            self.out_size = self.transformer.pooler.dense.out_features
        except KeyError:
            self.out_size = self.transformer.config.dim

        self.dropout = nn.Dropout(0.1)

        self.linear = nn.Linear(self.out_size, num_labels - 1)

        self.criterion = self.criterion = PairwiseHingeLoss()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        if evaluator is None:
            self.evaluator = Evaluator(
                optimization_metric=ir_measures.parse_measure(optimization_metric),
                mode=mode,
            )
        else:
            self.evaluator = evaluator

    def forward(self, input_ids, attention_mask, token_type_ids):

        sequence_output = self.transformer(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        ).last_hidden_state

        linear_output = self.linear(
            self.dropout(sequence_output[:, 0, :].view(-1, self.out_size))
        )

        return linear_output

    def training_step(self, batch, batch_idx):
        model_predictions = self(
            batch["input_ids"], batch["attention_mask"], batch["token_type_ids"]
        )

        model_predictions = self.sigmoid(model_predictions)
        model_predictions_p, model_predictions_n = split(
            model_predictions, model_predictions.size(dim=0) // 2
        )

        loss_value = self.criterion(
            model_predictions_p[:, 0], model_predictions_n[:, 0]
        )
        self.log("train_loss", loss_value, prog_bar=True, logger=True)

        return loss_value

    def eval_batch(self, batch):
        batch, qids, docnos = batch

        preds = self(
            batch["input_ids"], batch["attention_mask"], batch["token_type_ids"]
        )

        return {"qid": qids, "docno": docnos, "prediction": preds}

    def eval_epoch(self, outputs, name, epoch=-1):
        qids, docnos, preds = [], [], []
        if name == "pred":
            outputs = outputs[0]
        for output in outputs:
            qids.extend(output["qid"])
            docnos.extend(output["docno"])
            preds.append(output["prediction"].cpu().detach().numpy())

        preds = np.concatenate(preds, 0)

        eval = self.evaluator(
            qids=qids, docnos=docnos, pred_scores=preds, epoch=epoch, out_f_name=name
        )

        if name != "pred":
            for metric, value in eval.items():
                self.log(str(metric), value, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        return self.eval_batch(batch)

    def validation_epoch_end(self, outputs):
        self.eval_epoch(outputs, "during_training", self.current_epoch)

    def test_step(self, batch, batch_idx):
        return self.eval_batch(batch)

    def test_epoch_end(self, outputs):
        self.eval_epoch(outputs, "dev")

    def predict_step(self, batch, batch_idx):
        return self.eval_batch(batch)

    def on_predict_epoch_end(self, outputs):
        self.eval_epoch(outputs, "pred")

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        optimizer_class = AdamW
        optimizer_params = {"lr": 2e-5}
        linear = ["linear.weight", "linear.bias"]
        params = list(
            map(
                lambda x: x[1],
                list(filter(lambda kv: kv[0] in linear, param_optimizer)),
            )
        )
        base_params = list(
            map(
                lambda x: x[1],
                list(filter(lambda kv: kv[0] not in linear, param_optimizer)),
            )
        )
        optimizer = optimizer_class(
            [{"params": base_params}, {"params": params, "lr": 1e-3}],
            **optimizer_params
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps,
        )

        return dict(
            optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step")
        )
