from typing import List, Union

from trec_cds.data.utils import Gender


class ClinicalTrial:
    """ClinicalTrial is a wrapper class that contains most important fields
    from the ClicnialTrials xml dump file."""

    text_preprocessed: List[str]  # text which was preprocessed and is already tokenized

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
            minimum_age: Union[int, float, None],
            maximum_age: Union[int, float, None],
            healthy_volunteers: bool,
            text: str,
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
        self.minimum_age: Union[int, float, None] = minimum_age
        self.maximum_age: Union[int, float, None] = maximum_age
        self.healthy_volunteers: bool = healthy_volunteers  # True means accept healthy

        self.text: str = text
