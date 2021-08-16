from trec_cds.data.utils import Gender


class ClinicalTrial:
    def __init__(
        self,
        org_study_id,
        nct_id,
        summary,
        description,
        criteria,
        gender,
        minimum_age,
        maximum_age,
        healthy_volunteers,
        inclusion,
        exclusion,
    ):
        self.org_study_id : str = org_study_id
        self.nct_id: str = nct_id
        self.summary:str = summary
        self.description:str = description

        self.criteria:str = criteria
        self.inclusion = inclusion
        self.exclusion = exclusion

        self.gender:Gender = gender
        self.minimum_age:int = minimum_age
        self.maximum_age:int = maximum_age
        self.healthy_volunteers:bool = healthy_volunteers


    def get_text(self):
        if self.criteria is None or self.criteria.strip() == "":
            return self.criteria