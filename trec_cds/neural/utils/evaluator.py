import csv
from os.path import exists
from typing import List, Union, Tuple, Any, Optional

import ir_measures
import pandas as pd
from ir_measures import nDCG, RR, P


def judgements_mapping(qrels, mode):
    if mode == "judgement_correction":
        qrels.relevance = qrels.relevance - 1
    elif mode == "binary_mapping":
        qrels.loc[(qrels.relevance == 1), "relevance"] = 0
        qrels.loc[(qrels.relevance == 2), "relevance"] = 1
    return qrels


def read_run(
    config_file: Optional[str] = None,
    config_name: str = "easy",
    file_name: Optional[str] = None,
    sep: str = " ",
    bm25: bool = False,
):
    if file_name is None:
        import yaml

        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)[config_name]
        file_name = config["PATH_TO_RUN"]
    if bm25:
        score_field = "bm25_score"
    else:
        score_field = "score"

    run = pd.read_csv(
        file_name,
        header=None,
        names=["qid", "Q0", "docno", "rank", score_field, "run_id"],
        usecols=["qid", "docno", score_field],
        converters={"qid": str},
        sep=sep,
    )

    return run


class Evaluator:
    """
    Class responsible for performing evaluation of the model.
    """

    def __init__(
        self,
        optimization_metric=P @ 10,
        write_csv: bool = True,
        mode="train",
        output_path: str = "../../reports/",
        run_id: str = "DoSSIER_5_difficult",
        re_rank: bool = True,
        config_name: Optional[str] = None,
        path_to_base_run: Optional[str] = None,
        qrels_file: str = "../../data/raw/qrels_Judgment-of-0-is-non-relevant-1-is-excluded-and-2-is-eligible.txt",
        skip_Q0: bool = False,
        qrels_sep: str = " ",
    ):

        self.bm25 = None

        self.optimization_metric = optimization_metric
        self.output_path = output_path
        self.write_csv = write_csv
        graded_metrics = [nDCG @ 10, nDCG @ 5]
        non_graded_metrics = [RR, P @ 10]
        self.metrics = graded_metrics + non_graded_metrics
        self.csv_headers = ["epoch"] + [str(i) for i in self.metrics]
        self.run_id = run_id

        self.columns_mappings = {
            "qid": "query_id",
            "docno": "doc_id",
            "label": "relevance",
            "bm25_score": "score",
            "agg_score": "score",
        }

        qrels_fields = [
            "qid",
            "Q0",
            "docno",
            "label",
        ]

        if skip_Q0:
            qrels_fields.remove("Q0")

        qrels = pd.read_csv(
            qrels_file,
            header=None,
            names=qrels_fields,
            sep=qrels_sep,
            converters={"qid": str},
        )

        qrels = qrels.rename(columns=self.columns_mappings)

        if re_rank:
            self.bm25 = read_run(
                config_file=f"../../config/{mode}/config.yml",
                config_name=config_name,
                file_name=path_to_base_run,
                bm25=True,
            )

        qrels_map = qrels.rename(columns=self.columns_mappings).copy()

        self.n_queries = len(qrels_map["query_id"].unique())

        if len(set(qrels_map.relevance.unique()).intersection({1, 2, 3})) == 3:
            qrels_map = judgements_mapping(qrels_map, "judgement_correction")

        self.evaluator_graded = ir_measures.evaluator(graded_metrics, qrels_map)

        qrels_map_non_graded = judgements_mapping(qrels_map, "binary_mapping")

        self.evaluator_non_graded = ir_measures.evaluator(
            non_graded_metrics, qrels_map_non_graded
        )

    def __call__(
        self,
        examples: List[List[str]] = None,
        run_file: str = None,
        qids: List = None,
        docnos: List[int] = None,
        pred_scores=None,
        epoch: int = -1,
        save_report: bool = True,
        out_f_name: str = "",
        return_report: bool = False,
    ) -> Tuple[Any, List[List[Union[str, Any]]]]:

        output_path = self.output_path
        if run_file is not None:
            df_scores = read_run(file_name=run_file, sep=" ")
        else:
            df_scores = pd.DataFrame(
                {"qid": qids, "docno": docnos, "score": pred_scores[:, 0]}
            )
            df_scores.qid = df_scores.qid.astype(str)

        if self.bm25 is not None:
            df_scores = df_scores.merge(self.bm25, on=["qid", "docno"], how="left")

            df_scores["agg_score"] = (df_scores["score"] * 0.7) + (
                df_scores["bm25_score"] * 0.3
            )

            df_bm25_scores = df_scores[["qid", "docno", "bm25_score"]].copy()
            df_bm25_scores = df_bm25_scores.rename(columns=self.columns_mappings)
            df_bm25_scores.sort_values(
                by=["query_id", "score"], ascending=False, inplace=True
            )

            df_agg_socres = df_scores[["qid", "docno", "agg_score"]].copy()
            df_agg_socres = df_agg_socres.rename(columns=self.columns_mappings)
            df_agg_socres.sort_values(
                by=["query_id", "score"], ascending=False, inplace=True
            )

        df_scores = df_scores[["qid", "docno", "score"]]
        df_scores = df_scores.rename(columns=self.columns_mappings)
        if run_file is None:
            df_scores.sort_values(
                by=["query_id", "score"], ascending=False, inplace=True
            )

        if out_f_name in ["pred"]:

            if self.bm25 is not None:
                self.write_run(df_agg_socres, self.n_queries)
            else:
                self.write_run(df_scores, self.n_queries)

        eval_summary = []

        eval = self.evaluator_graded.calc_aggregate(df_scores)
        eval.update(self.evaluator_non_graded.calc_aggregate(df_scores))
        eval_summary += [eval]

        if self.bm25 is not None:
            eval = self.evaluator_graded.calc_aggregate(df_bm25_scores)
            eval.update(self.evaluator_non_graded.calc_aggregate(df_bm25_scores))
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

    def write_run(self, run, n_queries=50):
        run["Q0"] = "Q0"
        run["run_id"] = self.run_id
        run["rank"] = run.groupby("query_id")["score"].rank("dense", ascending=False)
        run[["query_id", "Q0", "doc_id", "rank", "score", "run_id"]].to_csv(
            f"{self.output_path}/{self.run_id}", index=False, sep=" ", header=False
        )


if __name__ == "__main__":
    evaluator = Evaluator(
        write_csv=True,
        output_path="../../reports/",
        qrels_file="2022_qrels_Judgment-of-0-is-non-relevant-2-is-excluded-and-2-is-eligible.txt",
    )

    evaluator(
        run_file="../../reports/DoSSIER_5_difficult",
        save_report=False,
        out_f_name="report_runs",
        return_report=False,
    )
