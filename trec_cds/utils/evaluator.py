from os.path import exists
import csv
from typing import List, Dict
import ir_measures
from ir_measures import *
import pandas as pd
from sklearn.metrics import f1_score
import numpy as np


class Evaluator:
    """
    Class responsible for performing evaluation of the model.
    """

    def __init__(
        self,
        optimization_metric=P@10,
        write_csv: bool = True,
        output_path: str = "../../reports/",
        run_id: str = None,
    ):

        self.optimization_metric = optimization_metric
        self.output_path = output_path
        self.write_csv = write_csv
        graded_metrics = [nDCG@10, nDCG@5]
        non_graded_metrics = [RR, P@10]
        self.csv_headers = ["epoch"] + [str(i) for i in self.eval.keys()]
        self.pred_samples = pred_samples

        # TODO get qrels here

        columns_mappings = {'qid': 'query_id', 'docno': 'doc_id', 'label': 'relevance'}

        qrels_map = qrels.rename(
            columns=columns_mappings
            ).copy()


        self.evaluator_graded = ir_measures.evaluator(graded_metrics, qrels_map)
        # TODO copy and modify 1-0 2-1
        self.evaluator_non_graded = ir_measures.evaluator(non_graded_metrics, qrels_map)

    def __call__(
        self,
        examples: List[List[str]] = None,
        ids: List = None,
        labels: List[int] = None,
        pred_scores=None,
        epoch: int = -1,
        save_report: bool = True,
        out_f_name: str = "",
        return_report: bool = False
    ) -> Dict[str, float]:

        output_path = self.output_path

        df_scores = pd.DataFrame(
            {"qid": self.qids, "score": self.pred_scores[:, 0], "label": self.labels})
        df_scores.sort_values(by=["qid", "score"], ascending=False, inplace=True)

        res_map = df_scores.rename(columns=columns_mappings).copy()

        eval = evaluator_graded.calc_aggregate(res_map)
        eval = eval.update(evaluator_non_graded.calc_aggregate(res_map))
        optimization_metric = eval[self.optimization_metric]

        if output_path is not None and self.write_csv:
            csv_path = f"{output_path}/{self.csv_file}"
            output_file_exists = exists(csv_path)
            with open(csv_path, "a", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch] + [eval[metric] for metric in self.metrics])

        return optimization_metric, [[str(metric), eval[str(metric)]] for metric in self.metrics]