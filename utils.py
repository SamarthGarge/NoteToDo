"""
utils.py
--------
Small formatting/export helpers shared by app.py. Kept separate from
extractor.py so the NLP logic stays free of any presentation concerns.
"""

from __future__ import annotations

import io
from typing import Iterable

import pandas as pd

from extractor import ActionItem


def items_to_dataframe(items: Iterable[ActionItem]) -> pd.DataFrame:
    """Convert a list of ActionItem objects into the dataframe shown in
    the Streamlit data_editor table."""
    rows = []
    for item in items:
        rows.append(
            {
                "Include": True,
                "Task": item.task,
                "Owner": item.owner or "",
                "Deadline": item.deadline or "",
                "Deadline (raw)": item.deadline_raw or "",
                "Confidence": item.confidence,
                "id": item.id,
            }
        )
    columns = ["Include", "Task", "Owner", "Deadline", "Deadline (raw)", "Confidence", "id"]
    return pd.DataFrame(rows, columns=columns)


def format_checklist_txt(df: pd.DataFrame) -> str:
    """Render the (filtered) results dataframe as a plain-text checklist,
    suitable for pasting into an email or Slack message."""
    lines = ["Action Items", "=" * 40, ""]

    if df.empty:
        lines.append("(No action items)")
        return "\n".join(lines)

    for _, row in df.iterrows():
        checkbox = "[ ]"
        line = f"{checkbox} {row['Task']}"
        details = []
        if row.get("Owner"):
            details.append(f"Owner: {row['Owner']}")
        if row.get("Deadline"):
            details.append(f"Due: {row['Deadline']}")
        elif row.get("Deadline (raw)"):
            details.append(f"Due: {row['Deadline (raw)']} (unresolved)")
        details.append(f"Confidence: {row['Confidence']}")
        if details:
            line += "\n      " + " | ".join(details)
        lines.append(line)

    return "\n".join(lines)


def format_checklist_csv(df: pd.DataFrame) -> str:
    """Render the (filtered) results dataframe as CSV text."""
    export_df = df.drop(columns=["Include", "id"], errors="ignore")
    buffer = io.StringIO()
    export_df.to_csv(buffer, index=False)
    return buffer.getvalue()


def summary_stats(df: pd.DataFrame) -> dict:
    """Compute the summary metrics shown at the top of the results area."""
    total = len(df)
    with_owner = int((df["Owner"] != "").sum()) if total else 0
    with_deadline = int((df["Deadline"] != "").sum()) if total else 0
    return {
        "total": total,
        "with_owner": with_owner,
        "with_deadline": with_deadline,
    }


SAMPLE_NOTES = {
    "Project sync (informal bullets)": """\
Team sync notes - 18 July

- Discussed the Q3 roadmap and everyone seemed aligned on priorities.
- Priya will send the updated budget spreadsheet by Friday.
- We need to review the vendor contracts before next week.
- John to follow up with the design team about the new mockups.
- Should we look into the new analytics tool at some point? Maybe.
- Sam's task is to schedule the client demo for next Tuesday.
- The meeting overall went well and morale seems high.
- Please finalize the slide deck by EOD tomorrow.
- Assigned to Maria: update the onboarding docs.
- Reminder: the office is closed next Monday for the holiday.
""",
    "Client call (paragraph style)": """\
Notes from the client call this afternoon. The client is happy with progress
so far and wants to move to the next phase soon. Alex will draft the revised
proposal and send it to the client by next Wednesday. We also need to
confirm the new pricing with finance before we can share anything.
There was a general discussion about long-term strategy but no firm
decisions were made. Nina must submit the signed contract by end of week.
Someone should double check the timeline slide, it looked outdated.
Follow up with legal about the NDA next week.
""",
}