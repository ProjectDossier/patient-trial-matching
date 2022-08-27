from dotmap import DotMap
import pytorch_lightning as pl
from trec_cds.models.crossencoder import CrossEncoder
from trec_cds.data.ClinicalTrialsDataModule import ClinicalTrialsDataModule
from pytorch_lightning.loggers import TensorBoardLogger
import yaml

if __name__ == "__main__":
    with open("../../config/prediction_config.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)["curriculum_learning"]  # name of the configuration
        config = DotMap(config)

    data_module = ClinicalTrialsDataModule(
        eval_batch_size=config.BATCH_SIZE,
        n_test_samples=config.N_SAMPLES,
        model_name=config.MODEL_NAME,
        mode=config.MODE,
        fields=config.FIELDS,
        query_repr=config.QUERY_REPR,
        path_to_run=config.PATH_2_RUN,
        path_to_qrels=config.PATH_2_QRELS
    )

    model = CrossEncoder.load_from_checkpoint(
        checkpoint_path=f"../../models/{config.MODEL_ALIAS}/checkpoints/{config.CHECKPOINT}",
        model_name=config.MODEL_NAME,
        num_labels=2
    )

    logger = TensorBoardLogger(
        save_dir=f"../../reports/{config.MODEL_ALIAS}_pred_logs",
        name=config.LOGGER_NAME
    )

    trainer = pl.Trainer(
        logger=logger,
        gpus=config.GPUS
    )

    trainer.predict(
        model=model,
        dataloaders=data_module.predict_dataloader()
    )
