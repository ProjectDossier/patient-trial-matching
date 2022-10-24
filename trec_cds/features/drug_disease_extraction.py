import medspacy


class EntityExtraction:
    def __init__(self):
        self.nlp = medspacy.load("en_ner_bc5cdr_md", enable=["sentencizer", "context"])

    def get_entities(self, text):
        doc = self.nlp(text)

        entities = []
        negated_entities = []
        pmh_entities = []
        fh_entities = []
        for ent in doc.ents:
            if any(
                [
                    ent._.is_negated,
                    ent._.is_uncertain,
                    ent._.is_historical,
                    ent._.is_family,
                    ent._.is_hypothetical,
                ]
            ):
                if ent._.is_negated:
                    negated_entities.append(str(ent))
                elif ent._.is_historical:
                    pmh_entities.append(str(ent))
                elif ent._.is_family:
                    fh_entities.append(str(ent))
                else:
                    entities.append(str(ent))
            else:
                entities.append(str(ent))

        return {
            "positive_entities": entities,
            "negated_entities": negated_entities,
            "pmh_entities": pmh_entities,
            "fh_entities": fh_entities,
        }
