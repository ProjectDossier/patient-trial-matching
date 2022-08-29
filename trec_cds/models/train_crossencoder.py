import pytorch_lightning as pl
import yaml
from dotmap import DotMap
from trec_cds.data.ClinicalTrialsDataModule import ClinicalTrialsDataModule
from crossencoder import CrossEncoder
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


if __name__ == "__main__":
    with open("../../config/train_config.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)["easy"]
        config = DotMap(config)

    data_module = ClinicalTrialsDataModule(
        model_name=config.MODEL_NAME,
        train_batch_size=config.TRAIN_BATCH_SIZE,
        eval_batch_size=config.EVAL_BATCH_SIZE,
        n_train_samples=config.N_TRAIN_SAMPLES,
        n_val_samples=config.N_VAL_SAMPLES,
        n_test_samples=config.N_TEST_SAMPLES,
        fields=config.FIELDS,
        query_repr=config.QUERY_REPR,
        relevant_labels=config.RELEVANT_LABELS,
        path_to_run=config.PATH_2_RUN,
        path_to_qrels=config.PATH_2_QRELS
    )

    model = CrossEncoder(
        model_name=config.MODEL_NAME,
        num_labels=2,
        n_warmup_steps=config.WARMUP_STEPS,
        n_training_steps=data_module.n_training_steps,
        batch_size=config.TRAIN_BATCH_SIZE,
        optimization_metric=config.TRACK_METRIC
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"../../models/{config.MODEL_ALIAS}/checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor=config.TRACK_METRIC,
        mode="max",
    )

    logger = TensorBoardLogger(
        save_dir=f"../../reports/{config.MODEL_ALIAS}_train_logs",
        name=config.LOGGER_NAME
    )

    early_stopping_callback = EarlyStopping(
        monitor=config.TRACK_METRIC,
        patience=config.PATIENCE,
        mode="max"
    )

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            early_stopping_callback,
            checkpoint_callback
        ],
        max_epochs=config.N_EPOCHS,
        gpus=config.GPUS,
        accumulate_grad_batches=config.ACCUM_ITER,
        check_val_every_n_epoch=config.EVAL_EVERY_N_EPOCH
    )

    trainer.fit(
        model=model,
        datamodule=data_module
    )

    trainer.test(
        dataloaders=data_module.test_dataloader()
    )
