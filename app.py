"""
app.py
------
Streamlit UI for the Meeting Notes -> Action Items Extractor.

Run locally with:
    streamlit run app.py

All the actual NLP work happens in extractor.py; this file is just the
UI layer (widgets, session state, export buttons).
"""

from datetime import date

import pandas as pd
import streamlit as st

from extractor import extract_action_items
from utils import (
    SAMPLE_NOTES,
    format_checklist_csv,
    format_checklist_txt,
    items_to_dataframe,
    summary_stats,
)

st.set_page_config(page_title="Meeting Notes -> Action Items", page_icon="✅", layout="wide")


# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------
# We keep the extracted results in session_state so that editing checkboxes
# in the data_editor doesn't re-run the (slightly slower) NLP pipeline.
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "notes_text" not in st.session_state:
    st.session_state.notes_text = ""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    meeting_date = st.date_input(
        "Meeting date",
        value=date.today(),
        help="Used as the reference point for resolving relative dates like 'next Friday' or 'in two weeks'.",
    )

    st.divider()

    st.subheader("Try a sample")
    sample_choice = st.selectbox("Sample notes", ["(none)"] + list(SAMPLE_NOTES.keys()))
    if st.button("Load Sample Notes", use_container_width=True):
        if sample_choice != "(none)":
            st.session_state.notes_text = SAMPLE_NOTES[sample_choice]
            st.session_state.results_df = None  # force re-extraction
        else:
            st.warning("Pick a sample from the dropdown first.")

    st.divider()

    with st.expander("How it works"):
        st.markdown(
            """
This tool uses **rule-based NLP** (spaCy) -- no LLMs, no paid APIs, and
nothing leaves your machine.

For every sentence in your notes, it checks for:
- **Modal/obligation verbs** -- "will", "need to", "should", "must"...
- **Imperative mood** -- sentences that start with a command verb, like
  "Send the report..."
- **Task-signaling verbs** -- "review", "schedule", "follow up"...
- **Assignment phrasing** -- "Priya will...", "assigned to Sam"...

Sentences matching one or more of these get flagged as action items.
Owners are found via named-entity recognition (looking for people's
names), and deadlines via date entities + a few regex fallbacks for
phrases like "next week" or "EOD".

Because it's rule-based, it won't catch every implied task, and it
may occasionally flag something that isn't really actionable -- that's
what the "Include" checkbox and delete-row controls are for.
            """
        )


# ---------------------------------------------------------------------------
# Main area -- Input
# ---------------------------------------------------------------------------
st.title("📝 Meeting Notes → Action Items Extractor")
st.caption("Paste your raw meeting notes below and get a clean, exportable checklist of who needs to do what, by when.")

uploaded_file = st.file_uploader("Upload a .txt file (optional)", type=["txt"])
if uploaded_file is not None:
    try:
        st.session_state.notes_text = uploaded_file.read().decode("utf-8")
        st.session_state.results_df = None
    except UnicodeDecodeError:
        st.error(
            "Couldn't read that file -- it doesn't look like plain UTF-8 text. "
            "Try opening it in a text editor and pasting the content directly instead."
        )

notes_text = st.text_area(
    "Meeting notes",
    value=st.session_state.notes_text,
    height=250,
    placeholder="Paste your meeting notes here...\n\nExample: 'Priya will send the budget by Friday. We need to review vendor contracts next week.'",
)
st.session_state.notes_text = notes_text

extract_clicked = st.button("Extract Action Items", type="primary")


# ---------------------------------------------------------------------------
# Run extraction
# ---------------------------------------------------------------------------
if extract_clicked:
    if not notes_text or not notes_text.strip():
        st.warning("Paste or upload some notes first.")
    else:
        with st.spinner("Analyzing notes..."):
            items = extract_action_items(notes_text, reference_date=meeting_date)
        st.session_state.results_df = items_to_dataframe(items)
        st.session_state.item_lookup = {item.id: item for item in items}


# ---------------------------------------------------------------------------
# Main area -- Results
# ---------------------------------------------------------------------------
if st.session_state.results_df is not None:
    df = st.session_state.results_df

    if df.empty:
        st.info("No action items found in these notes. Try rephrasing, or check the 'How it works' panel for what the tool looks for.")
    else:
        stats = summary_stats(df)
        col1, col2, col3 = st.columns(3)
        col1.metric("Action items found", stats["total"])
        col2.metric("With owner", stats["with_owner"])
        col3.metric("With deadline", stats["with_deadline"])

        st.subheader("Checklist")
        st.caption("Uncheck 'Include' to remove a row before exporting. Edits are kept until you re-run extraction.")

        edited_df = st.data_editor(
            df,
            column_config={
                "Include": st.column_config.CheckboxColumn("✅ Include", default=True),
                "id": None,  # hide internal id column
            },
            hide_index=True,
            use_container_width=True,
            key="results_editor",
        )
        st.session_state.results_df = edited_df

        st.subheader("Why was this flagged?")
        lookup = st.session_state.get("item_lookup", {})
        for _, row in edited_df.iterrows():
            item = lookup.get(row["id"])
            if item is None:
                continue
            with st.expander(f"{row['Task'][:80]}"):
                st.write("**Original sentence:**", item.sentence)
                st.write("**Matched cues:**", ", ".join(item.matched_cues) if item.matched_cues else "none")
                st.write("**Owner detected:**", item.owner or "None")
                if item.deadline:
                    st.write("**Deadline detected:**", f"{item.deadline} (from \"{item.deadline_raw}\")")
                elif item.deadline_raw:
                    st.write("**Deadline detected:**", f"Could not resolve exact date from \"{item.deadline_raw}\"")
                else:
                    st.write("**Deadline detected:**", "None")

        # Export
        st.subheader("Export")
        included_df = edited_df[edited_df["Include"]]

        txt_export = format_checklist_txt(included_df)
        csv_export = format_checklist_csv(included_df)

        exp_col1, exp_col2 = st.columns(2)
        exp_col1.download_button(
            "Download as .txt",
            data=txt_export,
            file_name="action_items.txt",
            mime="text/plain",
            use_container_width=True,
        )
        exp_col2.download_button(
            "Download as .csv",
            data=csv_export,
            file_name="action_items.csv",
            mime="text/csv",
            use_container_width=True,
        )
else:
    st.info("Paste your notes above (or load a sample from the sidebar) and click **Extract Action Items** to get started.")