# Meeting Notes → Action Items Extractor

A lightweight, free, fully offline tool that turns rough meeting notes into
a clean checklist of action items — each with a detected **owner**,
**deadline**, and **confidence score** — using classical rule-based NLP
(no LLMs, no paid APIs, no model training).

Built as a portfolio project. See `PRD.MD` and `TRD.MD` for the full
product/technical spec this implementation follows.

## Features

- Paste notes or upload a `.txt` file
- Two built-in sample note sets (bullet-style and paragraph-style) for a
  quick demo
- Detects action items using modal verbs, imperative mood, task-signaling
  verbs, and assignment phrasing
- Extracts **owner** (via NER + dependency parsing) and **deadline** (via
  NER `DATE` entities + regex fallback for phrases like "next week")
- Editable results table — uncheck or remove false positives before export
- "Why was this flagged?" panel per row for explainability
- Export as `.txt` (checklist) or `.csv`
- Runs 100% locally — nothing leaves your machine

## Project structure

```
meeting-action-extractor/
├── app.py                 # Streamlit UI
├── extractor.py           # Core NLP logic (imperative detection, NER, dates)
├── utils.py                # Formatting, export, sample notes
├── samples/                # Example meeting notes for testing
│   ├── sample_notes_1.txt  # bullet-style notes
│   └── sample_notes_2.txt  # paragraph-style notes
├── tests/
│   └── test_extractor.py
├── requirements.txt
├── setup.sh                 # downloads the spaCy model
└── README.md
```

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the spaCy English model
bash setup.sh
# (or directly: python -m spacy download en_core_web_sm)
```

> Note: `extractor.py` also tries to auto-download the model on first run
> if it's missing, so `setup.sh` is a convenience step, not a hard
> requirement.

## Run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Test

```bash
pytest
```

## How detection works

For every sentence, the tool checks for these cue categories (see
`extractor.py::classify_sentence`):

| Cue | Example |
|---|---|
| Modal/obligation verb | "will", "need to", "should", "must" |
| Imperative mood | "Send the report by Friday." |
| Task-signaling verb | "review", "schedule", "follow up", "finalize" |
| Assignment phrasing | "Priya will...", "assigned to Sam", "Sam's task is..." |

A sentence needs **1+** matched cue to be flagged. Confidence is Low (1
cue), Medium (2 cues), or High (3+ cues, or an owner *and* deadline both
found).

Owners are found via spaCy's `PERSON` named-entity recognition (preferring
the subject of the sentence), with a fallback to capitalized proper nouns
and a couple of explicit regex patterns like "assigned to X". Deadlines
come from spaCy `DATE` entities, normalized with `dateutil`, with regex
fallbacks for phrases like "next week", "EOD", or "next Friday" that NER
sometimes misses.

## Known limitations

This is a **rule-based** tool, not an ML/LLM system, so it inherits the
limitations noted in the PRD (§8):

- Implicit or hedged action items ("maybe we should look into X") may be
  missed or under-confident.
- Pronoun coreference ("she'll handle it") is intentionally out of scope —
  the tool won't guess who "she" is.
- Informal date phrases like "ASAP" or "sometime next month" may not
  resolve to an exact date (the raw phrase is still shown).
- Occasional false positives are expected (e.g. a modal verb inside a
  subordinate clause); use the "Include" checkbox in the results table to
  remove them before exporting.

## Requirements

- Python 3.10+
- streamlit
- spacy (`en_core_web_sm`)
- pandas
- python-dateutil
- pytest (for running tests)