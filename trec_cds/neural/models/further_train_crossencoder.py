import pytorch_lightning as pl
import yaml
from dotmap import DotMap
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from trec_cds.neural.data.ClinicalTrialsDataModule import ClinicalTrialsDataModule
from trec_cds.neural.models.crossencoder import CrossEncoder
from trec_cds.neural.utils.evaluator import Evaluator

if __name__ == "__main__":
    config_name = "difficult"
    with open("../../config/train/config.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)[
            config_name
        ]  # name of the configuration
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
        irrelevant_labels=config.IRRELEVANT_LABELS,
        path_to_run=config.PATH_2_RUN,
        path_to_qrels=config.PATH_2_QRELS,
    )

    evaluator = Evaluator(
        write_csv=True,
        mode="train",
        output_path="../../reports/",
        run_id=config_name,
        re_rank=True,
        config_name=config_name,
        qrels_file=config.PATH_2_QRELS,
    )

    model = CrossEncoder.load_from_checkpoint(
        checkpoint_path=f"../../models/{config.MODEL_ALIAS}/checkpoints/{config.CHECKPOINT}",
        model_name=config.MODEL_NAME,
        num_labels=2,
        n_warmup_steps=config.WARMUP_STEPS,
        n_training_steps=data_module.n_training_steps,
        batch_size=config.TRAIN_BATCH_SIZE,
        optimization_metric=config.TRACK_METRIC,
        evaluator=evaluator,
    )

    logger = TensorBoardLogger(
        save_dir=f"../../reports/{config.MODEL_ALIAS}_pred_logs",
        name=config.LOGGER_NAME,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"../../models/{config.MODEL_ALIAS}/checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor=config.TRACK_METRIC,
        mode="max",
    )

    early_stopping_callback = EarlyStopping(
        monitor=config.TRACK_METRIC, patience=config.PATIENCE, mode="max"
    )

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[early_stopping_callback, checkpoint_callback],
        max_epochs=config.N_EPOCHS,
        gpus=config.GPUS,
        accumulate_grad_batches=config.ACCUM_ITER,
        check_val_every_n_epoch=config.EVAL_EVERY_N_EPOCH,
    )

    trainer.fit(model=model, datamodule=data_module)

    trainer.test(dataloaders=data_module.test_dataloader())
