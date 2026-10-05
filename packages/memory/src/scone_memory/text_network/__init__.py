"""Bounded, model-free analysis of explicitly supplied text; no store writes."""
from .build import TextNetworkCancelled, build_text_network
from .models import NetworkLimits, TextDocument, TextNetwork, TextTerm
from .research import build_section_coverage, surrounding_context
from .research_models import (ContextBatch, ContextRequest, ContextWindow, CoverageTerm,
                              ResearchCoverage, ResearchLimits, SectionCoverage,
                              SectionStatistics, SectionTermCount, TextSection)

__all__ = ['TextDocument', 'TextTerm', 'TextNetwork', 'NetworkLimits', 'TextNetworkCancelled', 'build_text_network']
__all__ += ['TextSection', 'ResearchLimits', 'CoverageTerm', 'SectionTermCount', 'SectionStatistics',
            'ResearchCoverage', 'SectionCoverage', 'ContextRequest', 'ContextWindow', 'ContextBatch',
            'build_section_coverage', 'surrounding_context']
