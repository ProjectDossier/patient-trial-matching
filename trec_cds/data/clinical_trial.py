class ClinicalTrial:
    def __init__(
        self,
        org_study_id,
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
        self.org_study_id = org_study_id
        self.summary = summary
        self.description = description

        self.criteria = criteria
        self.inclusion = inclusion
        self.exclusion = exclusion

        self.gender = gender
        self.minimum_age = minimum_age
        self.maximum_age = maximum_age
        self.healthy_volunteers = healthy_volunteers
