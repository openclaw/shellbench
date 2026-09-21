import pytest
from clawbench.alignment.grading import normalize_evidence_quotes


def test_layout_reflow_returns_original_evidence_not_rewritten_text():
    original = "The receipt belongs to\n  `release-r17`, not the prior release."
    assert normalize_evidence_quotes(
        ["The receipt belongs to `release-r17`, not the prior release."], {"text": original}
    ) == [original]


@pytest.mark.parametrize(
    "quote",
    [
        "The receipt belongs to release-r17, not the prior release.",
        "The receipt belongs to `release-r18`, not the prior release.",
        "The receipt belongs to `release-r17`, the prior release.",
        "The receipt belongs to `release-r17`.*",
        "",
        "  ",
        None,
    ],
)
def test_reflow_never_accepts_changed_meaning_or_punctuation(quote):
    with pytest.raises(ValueError):
        normalize_evidence_quotes(
            [quote], {"text": "The receipt belongs to\n`release-r17`, not the prior release."}
        )


def test_quote_container_must_be_a_nonempty_list():
    for value in ("a", [], None):
        with pytest.raises(ValueError):
            normalize_evidence_quotes(value, {"text": "a"})


def test_json_encoded_literal_quotes_are_still_supported():
    assert normalize_evidence_quotes(['"line one\\nline two"'], {"text": "line one\nline two"})
