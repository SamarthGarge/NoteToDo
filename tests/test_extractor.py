"""
tests/test_extractor.py
------------------------
Unit tests for extractor.py, per TRD section 7.

Run with:
    pytest
"""

from datetime import date

import pytest

from extractor import (
    classify_sentence,
    extract_deadline,
    extract_owner,
    extract_action_items,
    nlp,
    score_confidence,
)


def _sent(text: str):
    """Helper: parse a single sentence and return spaCy's Span for it."""
    doc = nlp(text)
    return list(doc.sents)[0]


# ---------------------------------------------------------------------------
# classify_sentence
# ---------------------------------------------------------------------------

class TestClassifySentence:
    @pytest.mark.parametrize(
        "text",
        [
            "Priya will send the budget by Friday.",
            "We need to review the vendor contracts.",
            "Send the report by tomorrow.",
            "Please follow up with the design team.",
            "John to schedule the client demo.",
            "Assigned to Maria: update the onboarding docs.",
            "Nina must submit the signed contract by end of week.",
        ],
    )
    def test_positive_action_sentences(self, text):
        is_action, cues = classify_sentence(_sent(text))
        assert is_action is True
        assert len(cues) >= 1

    @pytest.mark.parametrize(
        "text",
        [
            "The meeting overall went well and morale seems high.",
            "We discussed the Q3 roadmap.",
            "The office is closed next Monday for the holiday.",
            "Everyone seemed aligned on priorities.",
        ],
    )
    def test_negative_non_action_sentences(self, text):
        is_action, cues = classify_sentence(_sent(text))
        assert is_action is False

    def test_questions_are_not_flagged(self):
        is_action, _ = classify_sentence(_sent("Should we look into the new analytics tool?"))
        assert is_action is False


# ---------------------------------------------------------------------------
# extract_owner
# ---------------------------------------------------------------------------

class TestExtractOwner:
    def test_named_subject_found(self):
        owner = extract_owner(_sent("Priya will send the budget by Friday."))
        assert owner == "Priya"

    def test_assigned_to_pattern(self):
        owner = extract_owner(_sent("Assigned to Maria: update the onboarding docs."))
        assert owner == "Maria"

    def test_no_owner_found(self):
        owner = extract_owner(_sent("We need to review the vendor contracts."))
        assert owner is None

    def test_pronoun_not_guessed(self):
        # Per PRD §8, pronoun coreference is explicitly out of scope --
        # "she'll handle it" should NOT resolve to a name.
        owner = extract_owner(_sent("She'll handle it."))
        assert owner is None


# ---------------------------------------------------------------------------
# extract_deadline
# ---------------------------------------------------------------------------

class TestExtractDeadline:
    REFERENCE = date(2026, 7, 20)  # a Monday

    def test_absolute_date(self):
        normalized, raw = extract_deadline(_sent("Submit the report by August 10th."), self.REFERENCE)
        assert raw is not None
        assert normalized == "2026-08-10"

    def test_relative_date_tomorrow(self):
        normalized, raw = extract_deadline(_sent("Finalize the slide deck by tomorrow."), self.REFERENCE)
        assert raw is not None
        assert normalized == "2026-07-21"

    def test_relative_date_next_week(self):
        normalized, raw = extract_deadline(_sent("We need to review this next week."), self.REFERENCE)
        assert raw is not None
        # Should at least resolve to *some* date, not crash.
        assert normalized is not None

    def test_no_date_present(self):
        normalized, raw = extract_deadline(_sent("We need to review the vendor contracts."), self.REFERENCE)
        assert normalized is None
        assert raw is None


# ---------------------------------------------------------------------------
# score_confidence
# ---------------------------------------------------------------------------

class TestScoreConfidence:
    def test_one_cue_is_low(self):
        assert score_confidence(["modal_will"], owner=None, deadline=None) == "Low"

    def test_two_cues_is_medium(self):
        assert score_confidence(["modal_will", "task_verb_send"], owner=None, deadline=None) == "Medium"

    def test_three_cues_is_high(self):
        assert score_confidence(
            ["modal_will", "task_verb_send", "assignment_pattern"], owner=None, deadline=None
        ) == "High"

    def test_owner_and_deadline_forces_high(self):
        assert score_confidence(["modal_will"], owner="Priya", deadline="2026-07-25") == "High"


# ---------------------------------------------------------------------------
# extract_action_items (end-to-end)
# ---------------------------------------------------------------------------

class TestExtractActionItems:
    def test_empty_input_returns_empty_list(self):
        assert extract_action_items("") == []
        assert extract_action_items("   ") == []

    def test_realistic_notes(self):
        notes = """
        Team sync notes.
        Priya will send the updated budget spreadsheet by Friday.
        We discussed the Q3 roadmap and everyone seemed aligned.
        John to follow up with the design team about the new mockups.
        The meeting overall went well.
        """
        items = extract_action_items(notes, reference_date=date(2026, 7, 20))
        assert len(items) >= 2
        tasks = [item.task for item in items]
        assert any("budget" in t.lower() for t in tasks)
        assert any("follow up" in t.lower() for t in tasks)

    def test_ids_are_sequential(self):
        notes = "Send the report today. Review the contract tomorrow."
        items = extract_action_items(notes, reference_date=date(2026, 7, 20))
        ids = [item.id for item in items]
        assert ids == list(range(1, len(items) + 1))