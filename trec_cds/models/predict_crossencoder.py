from dotmap import DotMap
import pytorch_lightning as pl
from trec_cds.models.crossencoder import CrossEncoder
from trec_cds.data.ClinicalTrialsDataModule import ClinicalTrialsDataModule
from pytorch_lightning.loggers import TensorBoardLogger
import yaml


with open("../../config/prediction_config.yml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)[experiment]  # name of the configuration
    config = DotMap(config)


data_module = ClinicalTrialsDataModule(
    eval_batch_size=config.TEST_BATCH_SIZE,
    n_test_samples=config.N_TEST_SAMPLES,
    mode="prediction"
)

# MODEL_NAME: crossencoder
model = CrossEncoder.load_from_checkpoint(
    checkpoint_path=f"../../models/{config.MODEL_NAME}/checkpoints/{config.CHECKPOINT}",
    model_name=config.MODEL_NAME,
    num_labels=2,
    pred_samples=data_module.pred_samples
)

# LOGGER_NAME: CT_preds
logger = TensorBoardLogger(
    save_dir=f"../../reports/{config.MODEL_NAME}_logs",
    name=config.LOGGER_NAME
)

trainer = pl.Trainer(
    logger=logger,
    gpus=[1]
)

trainer.predict(
    model=model,
    dataloaders=data_module.predict_dataloader()
)
