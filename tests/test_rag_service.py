import pytest
from app.services.rag_service import IDK_MSG, _normalize_answer


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("", IDK_MSG),
        ("  ", IDK_MSG),
        ("unknown", IDK_MSG),
        ("n/a", IDK_MSG),
        ("  valid answer  ", "valid answer"),
    ],
)
def test_normalize_answer_variants(raw, expected):
    assert _normalize_answer(raw) == expected
