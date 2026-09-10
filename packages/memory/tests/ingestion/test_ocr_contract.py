import pytest
from pydantic import ValidationError

from scone_memory.ocr.types import OcrRegion, OcrResult


def test_regions_require_finite_ordered_normalized_geometry():
    for box in ((0.5, 0.1, 0.2, 0.8), (-.1, 0., 1., 1.), (0., 0., float('nan'), 1.)):
        with pytest.raises(ValidationError):
            OcrRegion(text='word', box=box, score=.9)


def test_score_is_optional_and_never_a_correctness_probability():
    region = OcrRegion(text='word', box=(0., 0., 1., 1.), score=None)
    result = OcrResult(engine='configured-engine', width=10, height=10, regions=(region,))
    assert result.regions[0].score is None
    with pytest.raises(ValidationError):
        OcrRegion(text='word', box=(0., 0., 1., 1.), score=1.1)
