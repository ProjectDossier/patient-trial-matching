from dataclasses import dataclass
from typing import List, Union

from trec_cds.data.utils import Gender


@dataclass
class ClinicalTrial:
    """ClinicalTrial is a wrapper class that contains most important fields
    from the ClicnialTrials xml dump file."""

    org_study_id: str
    nct_id: str  # primary id
    brief_title: str
    official_title: str
    summary: str
    description: str
    criteria: str
    inclusion: List[str]
    exclusion: List[str]
    gender: Gender
    minimum_age: Union[int, float, None]
    maximum_age: Union[int, float, None]
    healthy_volunteers: bool  # True means accept healthy
    text: str

    # text which was preprocessed and is already tokenized
    text_preprocessed: Union[None, List[str]] = None
