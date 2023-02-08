from typing import List, Dict

import medspacy


class EntityExtraction:
    def __init__(self):
        self.nlp = medspacy.load("en_ner_bc5cdr_md", enable=["sentencizer", "context"])

    def get_entities(self, text:str) -> Dict[str, List[Dict[str, str]]]:
        doc = self.nlp(text)

        cmh_entities : List[Dict[str, str]] = []
        pmh_entities : List[Dict[str, str]] = []
        fh_entities  : List[Dict[str, str]] = []
        for ent in doc.ents:
            if ent._.is_family:
                fh_entities.append({
                    "text": str(ent),
                    "negated": ent._.is_negated,
                })
            elif ent._.is_historical:
                pmh_entities.append({
                    "text": str(ent),
                    "negated": ent._.is_negated,
                })
            else:
                cmh_entities.append({
                    "text": str(ent),
                    "negated": ent._.is_negated,
                })

        return {
            "cmh_entities": cmh_entities,
            "pmh_entities": pmh_entities,
            "fh_entities": fh_entities,
        }
