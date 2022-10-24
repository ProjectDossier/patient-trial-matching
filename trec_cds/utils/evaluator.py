from os.path import exists
import csv
from typing import List, Union, Tuple, Any
import ir_measures
from ir_measures import *
import pandas as pd


class Evaluator:
    """
    Class responsible for performing evaluation of the model.
    """

    def __init__(
        self,
        optimization_metric=P@10,
        write_csv: bool = True,
        mode="train",
        output_path: str = "../../reports/",
        run_id: str = "DoSSIER_5_difficult",
    ):

        self.optimization_metric = optimization_metric
        self.output_path = output_path
        self.write_csv = write_csv
        graded_metrics = [nDCG@10, nDCG@5]
        non_graded_metrics = [RR, P@10]
        self.metrics = graded_metrics + non_graded_metrics
        self.csv_headers = ["epoch"] + [str(i) for i in self.metrics]
        self.run_id = run_id

        self.columns_mappings = {
            'qid': 'query_id',
            'docno': 'doc_id',
            'label': 'relevance',
            'bm25_score': 'score',
            'agg_score': 'score'
        }

        qrels = pd.read_csv(
            "../../data/raw/qrels_Judgment-of-0-is-non-relevant-1-is-excluded-and-2-is-eligible.txt",
            header=None,
            names=[
                "qid",
                "Q0",
                "docno",
                "label",
            ],
            sep=" ",
            converters={"qid": str}
        )

        # TODO: DO BETTER
        #  e.g. same way as the dataloader
        import yaml
        with open(f"../../config/{mode}/config.yml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)[self.run_id]

        self.bm25 = pd.read_csv(
            config["PATH_2_RUN"],
            header=None,
            names=[
                "qid",
                "Q0",
                "docno",
                "rank",
                "bm25_score",
                "run_id"
            ],
            usecols=[
                "qid",
                "docno",
                "bm25_score"
            ],
            converters={"qid": str},
            sep=" "
        )

        qrels_map = qrels.rename(
            columns=self.columns_mappings
            ).copy()

        self.evaluator_graded = ir_measures.evaluator(graded_metrics, qrels_map)

        qrels_map_non_graded = qrels_map.copy()
        qrels_map_non_graded.loc[(qrels_map_non_graded.relevance == 1), "relevance"] = 0
        qrels_map_non_graded.loc[(qrels_map_non_graded.relevance == 2), "relevance"] = 1

        self.evaluator_non_graded = ir_measures.evaluator(non_graded_metrics, qrels_map_non_graded)

    def __call__(
        self,
        examples: List[List[str]] = None,
        qids: List = None,
        docnos: List[int] = None,
        pred_scores=None,
        epoch: int = -1,
        save_report: bool = True,
        out_f_name: str = "",
        return_report: bool = False
    ) -> Tuple[Any, List[List[Union[str, Any]]]]:

        output_path = self.output_path

        df_scores = pd.DataFrame(
            {"qid": qids, "docno": docnos, "score": pred_scores[:, 0]}
        )
        df_scores.qid = df_scores.qid.astype(str)

        df_scores = df_scores.merge(
            self.bm25,
            on=["qid", "docno"],
            how="left"
        )

        # TODO get rid of redundant code

        df_scores["agg_score"] = (df_scores.score * .7) + (df_scores.bm25_score * .3)

        df_bm25_scores = df_scores[["qid", "docno", "bm25_score"]].copy()
        df_agg_socres = df_scores[["qid", "docno", "agg_score"]].copy()
        df_scores = df_scores[["qid", "docno", "score"]]

        df_scores = df_scores.rename(columns=self.columns_mappings)
        df_bm25_scores = df_bm25_scores.rename(columns=self.columns_mappings)
        df_agg_socres = df_agg_socres.rename(columns=self.columns_mappings)

        df_scores.sort_values(by=["query_id", "score"], ascending=False, inplace=True)
        df_bm25_scores.sort_values(by=["query_id", "score"], ascending=False, inplace=True)
        df_agg_socres.sort_values(by=["query_id", "score"], ascending=False, inplace=True)

        df_scores.query_id = df_scores.query_id.astype(str)
        df_bm25_scores.query_id = df_bm25_scores.query_id.astype(str)
        df_agg_socres.query_id = df_agg_socres.query_id.astype(str)

        if out_f_name in ["pred"]:
            df_agg_socres["Q0"] = "Q0"
            df_agg_socres["run_id"] = self.run_id
            df_agg_socres["rank"] = list(range(1, 51)) * 50# TODO make it variable
            df_agg_socres[
                [
                    "query_id",
                    "Q0",
                    "doc_id",
                    "rank",
                    "score",
                    "run_id"
                ]
            ].to_csv(
                f"{self.output_path}/{self.run_id}",
                index=False,
                sep=" ",
                header=False
            )

        eval_summary = []

        eval = self.evaluator_graded.calc_aggregate(df_bm25_scores)
        eval.update(self.evaluator_non_graded.calc_aggregate(df_bm25_scores))
        eval_summary += [eval]

        eval = self.evaluator_graded.calc_aggregate(df_scores)
        eval.update(self.evaluator_non_graded.calc_aggregate(df_scores))
        eval_summary += [eval]

        eval = self.evaluator_graded.calc_aggregate(df_agg_socres)
        eval.update(self.evaluator_non_graded.calc_aggregate(df_agg_socres))
        eval_summary += [eval]

        optimization_metric = eval[ir_measures.parse_measure(self.optimization_metric)]

        if output_path is not None and self.write_csv:
            csv_path = f"{output_path}/report_{out_f_name}.csv"
            output_file_exists = exists(csv_path)
            with open(csv_path, "a", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)
                for eval in eval_summary:
                    writer.writerow([epoch] + [eval[metric] for metric in self.metrics])

        return eval
