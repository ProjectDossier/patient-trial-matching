import spacy
from spacy import displacy
import json
import random
from trec_cds.data.parsers import parse_topics_from_xml

with open("data/raw/label_config.json", "r") as json_file:
    labels = json.load(json_file)

# colors = {label['text'] : f"linear-gradient(90deg, {'#' + ''.join([random.choice('ABCDEF0123456789') for i in range(6)])}," \
#                           f"{'#' + ''.join([random.choice('ABCDEF0123456789') for i in range(6)])})" for label in labels}
colors = {
    label[
        "text"
    ]: f"{'#' + ''.join([random.choice('ABCDEF3456789') for i in range(6)])}"
    for label in labels
}

options = {"ents": [label["text"] for label in labels], "colors": colors}


# colors = {
#     "DISEASE": "linear-gradient(90deg, #aa9cfc, #fc9ce7)",
#     "CHEMICAL": "linear-gradient(90deg, #21fc3c, #ddcce7)",
#     "AGE": "linear-gradient(90deg, #413cec, #4480e7)",
#     "GENDER": "linear-gradient(90deg, #f1dc3c, #9d3c47)",
# }
# options = {"ents": ["DISEASE", "CHEMICAL", "AGE", "GENDER"], "colors": colors}


def load_model(model_dir="models/ner_age_gender/"):
    print("Loading from", model_dir)
    return spacy.load(model_dir)


if __name__ == "__main__":
    nlp = spacy.load("en_ner_bc5cdr_md")
    text = """
    Myeloid derived suppressor cells (MDSC) are immature 
    myeloid cells with immunosuppressive activity. 
    They accumulate in tumor-bearing mice and humans 
    with different types of cancer, including hepatocellular 
    carcinoma (HCC).
    """
    doc = nlp(text)

    print(doc)
    # displacy.serve(doc, style="ent")
    # displacy.serve(doc, style="ent", options=options)

    print(list(doc.sents))

    print(doc.ents)
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)

    import spacy

    age_gender_nlp = load_model()
    # give this component a copy of its own tok2vec
    # age_gender_nlp.replace_listeners("tok2vec", "ner", ["model.tok2vec"])

    # now you can put the drug component before or after the other ner
    # This will print a W113 warning but it's safe to ignore here
    nlp.add_pipe(
        "ner",
        name="ner_age_gender",
        source=age_gender_nlp,
        after="ner",
    )

    topic_file = "data/external/topics2021.xml"
    topics = parse_topics_from_xml(topic_file)

    docs = []
    for topic in topics:
        doc = nlp(topic.text)
        docs.append(doc)
        for ent in doc.ents:
            print(ent.text, ent.start_char, ent.end_char, ent.label_)

    from pathlib import Path

    # displacy.serve(docs, style="ent")
    displacy.serve(docs, style="ent", options=options)

    svg = displacy.render(docs, style="ent", options=options)
    output_path = Path("reports/figures/ner.svg")
    output_path.open("w", encoding="utf-8").write(svg)
