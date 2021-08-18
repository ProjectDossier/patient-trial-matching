import json


def postprocessing():
    pass


def convert_to_trac_submission(result_filename: str, run_name: str, output_folder):
    """submission format:
    TOPIC_NO Q0 ID RANK SCORE RUN_NAME
    """

    with open(result_filename) as fp:
        results = json.load(fp)

    with open(f"{output_folder}/{run_name}", "w") as fp:
        for topic_no in results:
            print(topic_no)
            # print(results[topic_no])
            rank = 0

            sorted_results = {
                k: v
                for k, v in sorted(
                    results[topic_no].items(), key=lambda item: item[1], reverse=True
                )
            }
            print(sorted_results)
            max_value = max(sorted_results.values())
            print(max_value)

            sorted_results = {k: v / max_value for k, v in sorted_results.items()}
            print(sorted_results)

            for rank, doc in enumerate(results[topic_no]):
                if rank >= 1000:
                    break
                # print(doc, results[topic_no][doc])
                score = results[topic_no][doc]

                line = f"{topic_no} Q0 {doc} {rank} {score} {run_name}\n"
                fp.write(line)


if __name__ == "__main__":
    convert_to_trac_submission(
        result_filename="data/processed/bm25-baseline-scores.json",
        run_name="baseline",
        output_folder="data/processed/submissions/",
    )
