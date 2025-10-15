"""
MI Knowledge Hub - Agent Package
7 specialized AI agents for content creation and maintenance
"""

__version__ = "0.1.0"

from .base_agent import BaseAgent
from .scholar_agent import ScholarAgent
from .content_agent import ContentAgent
from .tutor_agent import TutorAgent
from .data_agent import DataAgent
from .design_agent import DesignAgent
from .maintenance_agent import MaintenanceAgent
from .academic_reviewer_agent import AcademicReviewerAgent

__all__ = [
    "BaseAgent",
    "ScholarAgent",
    "ContentAgent",
    "TutorAgent",
    "DataAgent",
    "DesignAgent",
    "MaintenanceAgent",
    "AcademicReviewerAgent",
]
