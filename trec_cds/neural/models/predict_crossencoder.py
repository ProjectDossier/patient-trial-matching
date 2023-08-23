import pytorch_lightning as pl
import yaml
from dotmap import DotMap
from pytorch_lightning.loggers import TensorBoardLogger

from trec_cds.neural.data.ClinicalTrialsDataModule import ClinicalTrialsDataModule
from trec_cds.neural.models.crossencoder import CrossEncoder
from trec_cds.neural.utils.evaluator import Evaluator

if __name__ == "__main__":
    config_name = "DoSSIER_5_difficult"
    with open("../../config/predict/config.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)[
            config_name
        ]  # name of the configuration
        config = DotMap(config)

    data_module = ClinicalTrialsDataModule(
        eval_batch_size=config.BATCH_SIZE,
        n_test_samples=config.N_SAMPLES,
        model_name=config.MODEL_NAME,
        mode=config.MODE,
        fields=config.FIELDS,
        query_repr=config.QUERY_REPR,
        relevant_labels=config.RELEVANT_LABELS,
        path_to_run=config.PATH_2_RUN,
        path_to_qrels=config.PATH_2_QRELS,
        path_to_trials_jsonl=config.PATH_2_TRIALS,
        dataset_version=config.VERSION,
    )

    evaluator = Evaluator(
        write_csv=True,
        mode="predict",
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
        mode="predict",
        evaluator=evaluator,
    )

    logger = TensorBoardLogger(
        save_dir=f"../../reports/{config.MODEL_ALIAS}_pred_logs",
        name=config.LOGGER_NAME,
    )

    trainer = pl.Trainer(logger=logger, gpus=config.GPUS)

    trainer.predict(model=model, dataloaders=data_module.predict_dataloader())
