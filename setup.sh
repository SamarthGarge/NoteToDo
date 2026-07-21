#!/usr/bin/env bash
# Downloads the spaCy English model needed by extractor.py.
# Run once after `pip install -r requirements.txt`.
# (extractor.py also has a fallback that auto-downloads this if missing,
# but running it explicitly during deploy is faster/more reliable.)
python -m spacy download en_core_web_sm