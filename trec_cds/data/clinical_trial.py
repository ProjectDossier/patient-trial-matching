from typing import List
from datetime import datetime
from trec_cds.data.utils import Gender


class ClinicalTrial:
    """ClinicalTrial is a wrapper class that contains most important fields from the ClicnialTrials xml dump file."""

    def __init__(
        self,
        org_study_id: str,
        nct_id: str,
        brief_title: str,
        official_title: str,
        summary: str,
        description: str,
        criteria: str,
        inclusion: List[str],
        exclusion: List[str],
        gender: Gender,
        minimum_age: datetime,
        maximum_age: datetime,
        healthy_volunteers: bool,
    ):
        self.org_study_id: str = org_study_id
        self.nct_id: str = nct_id  # primary id

        self.brief_title: str = brief_title
        self.official_title: str = official_title

        self.summary: str = summary
        self.description: str = description

        self.criteria: str = criteria
        self.inclusion: List[str] = inclusion
        self.exclusion: List[str] = exclusion

        self.gender: Gender = gender
        self.minimum_age: datetime = minimum_age
        self.maximum_age: datetime = maximum_age
        self.healthy_volunteers: bool = healthy_volunteers  # True means accept healthy

    def get_text(self):
        if self.criteria is None or self.criteria.strip() == "":
            return self.criteria
