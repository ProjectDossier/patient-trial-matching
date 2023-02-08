import os
from ranx import Run, Qrels
from ranx import compare

if __name__ == '__main__':
    runs_path = "/newstorage4/wkusa/data/trec_cds/data/processed/jbi/ie_filtered/"
    year = "2021"
    run_files = [
        x
        for x in os.listdir(f"{runs_path}/{year}")
        if not (x.endswith("json") or x.endswith("-results"))
    ]
    sections_base_run = f"/newstorage4/wkusa/data/trec_cds/data/processed/jbi/sections/{year}/bm25p-summary_description_titles_conditions_inclusion-20230202"
    base_run = f"/newstorage4/wkusa/data/trec_cds/data/processed/jbi/ie/{year}/bm25p-an_cpf-20230207"

    runs = [Run.from_file(sections_base_run, kind="trec"), Run.from_file(base_run, kind="trec")]
    for run in run_files:
        if not run.startswith("bm25p-an_cpf-20230207"):
            continue

        if run in ['bm25p-an_cpf-20230207_smoking',
                   'bm25p-an_cpf-20230207_drinking']:
            continue
        run = Run.from_file(f"{runs_path}/{year}/{run}", kind="trec")
        runs.append(run)

    qrels = Qrels.from_file(
        f"/home/wkusa/projects/trec-cds/data/external/qrels{year}.txt", kind="trec"
    )

    cmp1 = compare(qrels=qrels, runs=runs, metrics={"ndcg@10", "ndcg@5"}, max_p=0.05)

    print(cmp1.to_latex())


    qrels = Qrels.from_file(
        f"/home/wkusa/projects/trec-cds/data/external/qrels{year}_binary.txt", kind="trec"
    )

    cmp2 = compare(qrels=qrels, runs=runs, metrics={"precision@10", "mrr"}, max_p=0.05)

    print(cmp2.to_latex())



