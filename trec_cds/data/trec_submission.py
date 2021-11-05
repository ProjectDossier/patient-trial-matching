import json
import logging
import os
from typing import Union

logging.basicConfig(level=logging.INFO)


def convert_to_trac_submission(
    result_filename: str,
    run_name: str,
    output_folder: str,
    trim_scores_less_than: Union[None, float] = None,
) -> None:
    """Converts intermediate result json file to a TREC submission format:
    TOPIC_NO Q0 ID RANK SCORE RUN_NAME

    :param result_filename: json file containing retrieval results
        json file format: {TOPIC_ID : { NCT_ID_1 : score , NCT_ID_2 : score , ...}, ...}
    :param run_name: run name should have less than 12 characters
    :param output_folder:
    :param trim_scores_less_than:
    """

    with open(result_filename) as fp:
        results = json.load(fp)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    logging.info("Converting total number of %d topics", len(results))
    with open(f"{output_folder}/{run_name}", "w") as fp:
        for topic_no in results:
            logging.info("working on topic: %s", topic_no)

            sorted_results = {
                k: v
                for k, v in sorted(
                    results[topic_no].items(), key=lambda item: item[1], reverse=True
                )
            }
            if trim_scores_less_than:
                logging.info("Trimming scores less than %f", trim_scores_less_than)
                before_len = len(sorted_results)
                sorted_results = {
                    k: v for k, v in sorted_results.items() if v > trim_scores_less_than
                }
                logging.info(
                    "Removed %f (%f %%) results after trimming. Final len: %d",
                    (before_len - len(sorted_results)),
                    (100 * (before_len - len(sorted_results)) / before_len),
                    len(sorted_results),
                )

            logging.info("normalizing results")
            max_value = max(sorted_results.values())
            sorted_results = {k: v / max_value for k, v in sorted_results.items()}

            for rank, doc in enumerate(sorted_results):
                if rank >= 1000:  # TREC submission allows max top 1000 results
                    break
                score = sorted_results[doc]

                line = f"{topic_no} Q0 {doc} {rank + 1} {score} {run_name}\n"
                fp.write(line)


if __name__ == "__main__":
    convert_to_trac_submission(
        result_filename="data/processed/bm25-baseline-postprocessed-reranked-4000.json",
        run_name="rerank4000",
        output_folder="data/processed/submissions/",
        trim_scores_less_than=0.20,
    )
