"""
extractor.py
------------
Core NLP logic for the Meeting Notes -> Action Items Extractor.

This module is pure Python + spaCy (no Streamlit imports) so it can be
tested in isolation. All rules are "classical" NLP: POS tags, dependency
parsing, named-entity recognition, and regex -- no machine-learning
classifiers or LLM calls, per the project constraints.

Main entry point: extract_action_items(text, reference_date)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Optional

import spacy
from dateutil import parser as dateutil_parser


# ---------------------------------------------------------------------------
# spaCy model loading
# ---------------------------------------------------------------------------
# We load the model once at import time and reuse it everywhere. If the
# model isn't installed yet (e.g. first run on Streamlit Cloud), we try to
# download it automatically instead of crashing the whole app.

_MODEL_NAME = "en_core_web_sm"


def _load_spacy_model():
    """Load the spaCy English model, downloading it if it's missing."""
    try:
        return spacy.load(_MODEL_NAME)
    except OSError:
        from spacy.cli import download as spacy_download

        spacy_download(_MODEL_NAME)
        return spacy.load(_MODEL_NAME)


nlp = _load_spacy_model()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ActionItem:
    """One detected action item, plus everything needed to explain it."""

    id: int
    sentence: str
    task: str
    owner: Optional[str] = None
    deadline: Optional[str] = None       # normalized ISO date, e.g. "2026-07-25"
    deadline_raw: Optional[str] = None   # original phrase, e.g. "by next Friday"
    confidence: str = "Low"              # "High" | "Medium" | "Low"
    matched_cues: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Cue vocabularies (kept simple & editable -- this is the "explainability"
# backbone: every match adds a human-readable string to matched_cues)
# ---------------------------------------------------------------------------

# Modal / obligation verbs and phrases that signal something is expected
# to happen. Checked as lowercase substrings/lemmas.
MODAL_CUES = {
    "will": "modal_will",
    "shall": "modal_shall",
    "should": "modal_should",
    "must": "modal_must",
    "need to": "modal_need_to",
    "needs to": "modal_need_to",
    "have to": "modal_have_to",
    "has to": "modal_have_to",
    "ought to": "modal_ought_to",
}

# Verbs that, on their own, tend to signal a task even without a modal verb.
TASK_VERBS = {
    "follow up", "review", "prepare", "schedule", "finalize", "finalise",
    "send", "share", "update", "draft", "complete", "submit", "contact",
    "email", "call", "organize", "organise", "arrange", "check", "confirm",
    "create", "write", "fix", "test", "publish", "upload", "book", "set up",
    "coordinate", "circulate", "compile", "research", "investigate",
}

# Words that commonly start a sentence but are NOT the subject of an
# imperative verb (used to avoid false-positive imperative detection).
_QUESTION_STARTERS = {"who", "what", "when", "where", "why", "how", "which", "is", "are", "do", "does", "did", "can", "could", "would"}

# Common capitalized words that are NOT names, used to filter the
# fallback proper-noun regex for owner detection.
_COMMON_CAPITALIZED_NON_NAMES = {
    "The", "This", "That", "These", "Those", "We", "I", "Next", "Then",
    "Also", "Please", "Note", "Meeting", "Action", "Item", "Team", "Today",
    "Tomorrow", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday", "January", "February", "March", "April", "May",
    "June", "July", "August", "September", "October", "November",
    "December",
}

# Relative date phrases that spaCy's NER sometimes misses, as a regex
# fallback (case-insensitive).
_RELATIVE_DATE_PATTERN = re.compile(
    r"\b(today|tomorrow|tonight|EOD|EOW|COB|"
    r"next week|this week|next month|"
    r"next monday|next tuesday|next wednesday|next thursday|next friday|"
    r"next saturday|next sunday)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 4.2 Action-item classification
# ---------------------------------------------------------------------------

def classify_sentence(sent) -> tuple[bool, list[str]]:
    """Decide whether a spaCy sentence Span is a candidate action item.

    Returns (is_action_item, matched_cues) where matched_cues is a list of
    short human-readable tags explaining exactly which rule(s) fired --
    this list is what powers the "Why was this flagged?" UI panel.
    """
    text_lower = sent.text.lower()
    cues: list[str] = []

    # Skip pure questions -- they rarely describe a committed task.
    is_question = sent.text.strip().endswith("?")

    # --- Modal / obligation verbs ---------------------------------------
    for phrase, tag in MODAL_CUES.items():
        if _contains_word_or_phrase(text_lower, phrase):
            cues.append(tag)

    # --- Imperative mood: sentence starts with a base-form verb (VB) and
    # has no subject attached to it (e.g. "Send the report by Friday.") ---
    if _is_imperative(sent) and not is_question:
        cues.append("imperative_verb")

    # --- Task-signaling verbs anywhere in the sentence -------------------
    matched_task_verbs = set()
    for token in sent:
        lemma = token.lemma_.lower()
        if lemma in TASK_VERBS and token.pos_ == "VERB":
            matched_task_verbs.add(lemma)
    # Also catch two-word verbs like "follow up" that spaCy tokenizes apart.
    for phrase in TASK_VERBS:
        if " " in phrase and phrase in text_lower:
            matched_task_verbs.add(phrase)
    for verb in matched_task_verbs:
        cues.append(f"task_verb_{verb.replace(' ', '_')}")

    # --- Assignment phrasing: "X will...", "X to...", "assigned to X" ---
    if _has_assignment_pattern(sent):
        cues.append("assignment_pattern")

    is_action_item = len(cues) >= 1 and not is_question
    return is_action_item, cues


def _contains_word_or_phrase(text_lower: str, phrase: str) -> bool:
    """Whole-word/phrase match so 'will' doesn't match inside 'willing'."""
    pattern = r"\b" + re.escape(phrase) + r"\b"
    return re.search(pattern, text_lower) is not None


def _is_imperative(sent) -> bool:
    """True if the sentence looks like a command: starts with a base-form
    verb (tag VB) and that verb has no nominal subject (nsubj) child."""
    tokens = [t for t in sent if not t.is_space and not t.is_punct]
    if not tokens:
        return False

    first = tokens[0]
    if first.text.lower() in _QUESTION_STARTERS:
        return False

    if first.tag_ != "VB":
        return False

    has_subject = any(child.dep_ in ("nsubj", "nsubjpass") for child in first.children)
    return not has_subject


def _has_assignment_pattern(sent) -> bool:
    """Detect phrasing like 'Priya will handle it', 'assigned to Sam',
    'John's task is to...' via dependency parse + light regex."""
    text_lower = sent.text.lower()

    if re.search(r"\bassigned to\s+\w+", text_lower):
        return True
    if re.search(r"\b\w+'s (task|job|responsibility) is\b", text_lower):
        return True

    # Dependency-based: a proper-noun subject (nsubj) whose head verb is
    # governed by a modal ("will", "to") -- e.g. "Priya will review this".
    for token in sent:
        if token.dep_ == "nsubj" and token.pos_ in ("PROPN", "PRON"):
            head = token.head
            if head.pos_ == "VERB":
                has_modal_child = any(c.tag_ == "MD" for c in head.children)
                if has_modal_child:
                    return True
    return False


# ---------------------------------------------------------------------------
# 4.3 Owner extraction
# ---------------------------------------------------------------------------

# Entity labels that clearly aren't a person's name -- used to keep the
# proper-noun fallback from grabbing a date, place, or org by mistake.
_NON_NAME_ENTITY_LABELS = {"DATE", "TIME", "GPE", "ORG", "MONEY", "PERCENT", "QUANTITY", "CARDINAL", "ORDINAL"}


def extract_owner(sent) -> Optional[str]:
    """Best-effort extraction of who owns the task in this sentence.

    Order of preference (per TRD 4.3):
      1. spaCy PERSON entity, preferring one in subject (nsubj) position
      2. Explicit "assigned to X" / "X's task is" phrasing
      3. Fallback: a capitalized proper noun not caught as a non-name entity
      4. None (we deliberately do not guess from pronouns like "she'll")
    """
    person_ents = [ent for ent in sent.ents if ent.label_ == "PERSON"]
    if person_ents:
        for ent in person_ents:
            if ent.root.dep_ in ("nsubj", "nsubjpass", "poss"):
                return ent.text
        return person_ents[0].text

    # Explicit "assigned to X" phrasing -- small models often mis-tag or
    # miss names entirely, so we check this pattern directly.
    assigned_match = re.search(r"\bassigned to\s+([A-Z][a-zA-Z]*)", sent.text)
    if assigned_match:
        return assigned_match.group(1)

    possessive_match = re.search(r"\b([A-Z][a-zA-Z]*)'s (?:task|job|responsibility) is\b", sent.text)
    if possessive_match:
        return possessive_match.group(1)

    # Fallback: capitalized proper noun not part of a clearly-non-name
    # entity (DATE, ORG, GPE, etc). We deliberately do NOT skip the first
    # word of the sentence -- names very often are the subject and start
    # the sentence (e.g. "Priya will send...").
    non_name_spans = {
        (e.start, e.end) for e in sent.ents if e.label_ in _NON_NAME_ENTITY_LABELS
    }

    candidates = []
    for token in sent:
        if token.pos_ != "PROPN":
            continue
        if not token.text.isalpha() or not token.text[0].isupper():
            continue
        if token.text in _COMMON_CAPITALIZED_NON_NAMES:
            continue
        covered = any(start <= token.i < end for start, end in non_name_spans)
        if covered:
            continue
        candidates.append(token)

    if not candidates:
        return None

    # Prefer a candidate in subject position.
    for token in candidates:
        if token.dep_ in ("nsubj", "nsubjpass", "poss"):
            return token.text

    return candidates[0].text


# ---------------------------------------------------------------------------
# 4.4 Deadline extraction
# ---------------------------------------------------------------------------

def extract_deadline(sent, reference_date: date) -> tuple[Optional[str], Optional[str]]:
    """Find a deadline in the sentence.

    Returns (normalized_iso_date_or_None, raw_phrase_or_None). We always
    keep the raw phrase even if normalization fails, so the UI has
    something useful to show.
    """
    date_ents = [ent for ent in sent.ents if ent.label_ == "DATE"]

    raw_phrase = None
    if date_ents:
        raw_phrase = date_ents[0].text
    else:
        match = _RELATIVE_DATE_PATTERN.search(sent.text)
        if match:
            raw_phrase = match.group(0)

    if raw_phrase is None:
        return None, None

    normalized = _normalize_date_phrase(raw_phrase, reference_date)
    return normalized, raw_phrase


_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _normalize_date_phrase(phrase: str, reference_date: date) -> Optional[str]:
    """Turn a raw date phrase into an ISO date string, using
    reference_date to resolve fuzzy/relative phrases. Returns None if the
    phrase can't be parsed (e.g. 'ASAP', 'sometime next month').

    NER-detected DATE spans sometimes include extra words (e.g. spaCy may
    return "this next week" instead of just "next week"), so we search
    for known relative-date patterns *within* the phrase rather than
    requiring an exact match.
    """
    from datetime import timedelta

    lowered = phrase.strip().lower()

    if re.search(r"\b(today|tonight|eod|cob)\b", lowered):
        return reference_date.isoformat()

    if re.search(r"\btomorrow\b", lowered):
        return (reference_date + timedelta(days=1)).isoformat()

    if re.search(r"\b(eow|end of( the)? week|this week)\b", lowered):
        days_until_friday = (4 - reference_date.weekday()) % 7
        return (reference_date + timedelta(days=days_until_friday)).isoformat()

    if re.search(r"\bnext week\b", lowered):
        return (reference_date + timedelta(days=7)).isoformat()

    if re.search(r"\bnext month\b", lowered):
        # Approximate "next month" as +30 days -- good enough for a
        # rule-based demo tool without pulling in a calendar library.
        return (reference_date + timedelta(days=30)).isoformat()

    weekday_match = re.search(r"(next\s+)?(" + "|".join(_WEEKDAYS) + r")\b", lowered)
    if weekday_match:
        is_next = bool(weekday_match.group(1))
        target_weekday = _WEEKDAYS[weekday_match.group(2)]
        days_ahead = (target_weekday - reference_date.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7  # "Friday"/"next Friday" said on a Friday means the coming one
        elif is_next:
            days_ahead += 7  # "next Friday" skips this week's occurrence
        return (reference_date + timedelta(days=days_ahead)).isoformat()

    try:
        default_dt = datetime(reference_date.year, reference_date.month, reference_date.day)
        parsed = dateutil_parser.parse(lowered, fuzzy=True, default=default_dt)
        return parsed.date().isoformat()
    except (ValueError, OverflowError):
        return None


# ---------------------------------------------------------------------------
# 4.2 Confidence scoring
# ---------------------------------------------------------------------------

def score_confidence(matched_cues: list[str], owner: Optional[str], deadline: Optional[str]) -> str:
    """Per TRD: 1 cue -> Low, 2 cues -> Medium, 3+ cues (or owner+deadline
    both present) -> High."""
    num_cues = len(matched_cues)
    if num_cues >= 3 or (owner is not None and deadline is not None):
        return "High"
    if num_cues == 2:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Line-aware preprocessing
# ---------------------------------------------------------------------------
# Real meeting notes mix two styles: bullet lists (where each line IS a
# sentence, even without ending punctuation) and paragraph text that's
# been word-wrapped across lines (where a newline is NOT a sentence
# boundary). spaCy's sentence splitter alone can't tell these apart, so we
# do a line-aware pass first: bullet lines become their own chunk, and
# consecutive plain lines get joined back into one chunk before spaCy
# splits them into actual sentences.

_BULLET_PATTERN = re.compile(r"^\s*([\-\*\u2022]|\d+[\.\)])\s+")


def _split_into_chunks(text: str) -> list[str]:
    """Group raw lines into chunks ready for spaCy sentence splitting."""
    chunks: list[str] = []
    buffer: list[str] = []

    def _flush():
        if buffer:
            chunks.append(" ".join(buffer))
            buffer.clear()

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            _flush()
            continue
        if _BULLET_PATTERN.match(line):
            _flush()
            chunks.append(_BULLET_PATTERN.sub("", stripped))
        else:
            buffer.append(stripped)
    _flush()
    return chunks


# ---------------------------------------------------------------------------
# Task text cleanup
# ---------------------------------------------------------------------------

def _clean_task_text(sentence_text: str) -> str:
    """Light cleanup for the 'task' column: strip whitespace and any
    leading bullet/numbering characters left over from raw notes."""
    text = sentence_text.strip()
    text = re.sub(r"^[\-\*\u2022\d\.\)]+\s*", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_action_items(text: str, reference_date: Optional[date] = None) -> list[ActionItem]:
    """Run the full pipeline over raw meeting notes and return a list of
    ActionItem records for every sentence flagged as a likely task."""
    if reference_date is None:
        reference_date = date.today()

    if not text or not text.strip():
        return []

    chunks = _split_into_chunks(text)
    items: list[ActionItem] = []
    next_id = 1

    for chunk_text in nlp.pipe(chunks):
        for sent in chunk_text.sents:
            if not sent.text.strip():
                continue

            is_action, cues = classify_sentence(sent)
            if not is_action:
                continue

            owner = extract_owner(sent)
            deadline, deadline_raw = extract_deadline(sent, reference_date)
            confidence = score_confidence(cues, owner, deadline)

            items.append(
                ActionItem(
                    id=next_id,
                    sentence=sent.text.strip(),
                    task=_clean_task_text(sent.text),
                    owner=owner,
                    deadline=deadline,
                    deadline_raw=deadline_raw,
                    confidence=confidence,
                    matched_cues=cues,
                )
            )
            next_id += 1

    return items