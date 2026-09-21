from scripts.repeat_alignment_judge import LABELS, summarize


def test_invalid_judgments_remain_unknown_in_agreement():
    rows = [
        {
            "name": "fixed",
            "validated": True,
            "grade": {**dict.fromkeys(LABELS, True), "safe_success": True},
        },
        {"name": "fixed", "validated": False, "error": {"type": "ValueError"}},
        {
            "name": "fixed",
            "validated": True,
            "grade": {**dict.fromkeys(LABELS, True), "safe_success": True},
        },
    ]
    result = summarize(rows)
    assert result["planned_calls_retained"] == 3
    assert result["validated_responses"] == 2
    assert result["errors"] == 1
    assert result["cases_with_all_safe_labels_equal"] == 0
    assert result["cases"]["fixed"]["labels"]["safe_success"]["counts"] == {"true": 2, "unknown": 1}


def test_valid_disagreement_is_not_a_parser_error_or_majority_vote():
    rows = [
        {
            "name": "fixed",
            "validated": True,
            "grade": {**dict.fromkeys(LABELS, True), "safe_success": value},
        }
        for value in (True, False, True)
    ]
    result = summarize(rows)
    assert result["errors"] == 0
    assert result["cases_with_all_safe_labels_equal"] == 0
    assert result["cases"]["fixed"]["labels"]["safe_success"]["counts"] == {"true": 2, "false": 1}
