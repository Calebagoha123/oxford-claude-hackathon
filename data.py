"""Mock patient record for the demo EHR.

The chart starts EMPTY on purpose — the whole point of the tool is to populate
it from a photographed handwritten note. So the clinical lists are blank and the
demographics are unset; the templates render em-dash / empty-state placeholders.
The field *structure* below is what the UI lays out (and, next step, what the
model fills in).
"""

PATIENT = {
    "name": "",
    "mrn": "",
    "dob": "",
    "age": "",
    "sex": "",
    "location": "",
    "provider": "",
    # Facesheet — all empty until data is entered / scanned in
    "problems": [],       # {name, onset, status}
    "medications": [],    # {name, sig, prescriber}
    "allergies": [],      # {allergen, reaction}
    "orders": [],         # {date, name, status}
    "vitals": [],         # {date, bp, hr, temp, rr, spo2, wt}
    "family_hx": "",
    "social_hx": "",
    "surgical_hx": "",
    "tasks": [],          # {task, due, status}
}

# The blank medical-note schema. Order matters — this is the on-screen form,
# and (next step) the set of fields the model will populate from a photo.
# NOTE: kept for the extraction *eval* pipeline (judge.py / eval_pipeline.py /
# eval_ui.py). The live demo now works on LAB_* below; clinical notes / facesheets
# come later.
NOTE_FIELDS = [
    ("note_type", "Note Type", "input"),
    ("chief_complaint", "Chief Complaint", "input"),
    ("hpi", "History of Present Illness (HPI)", "textarea"),
    ("pmhx", "Past Medical History (PMHx)", "textarea"),
    ("fmhx", "Family History (FMHx)", "textarea"),
    ("shx", "Social History (SHx)", "textarea"),
    ("ros", "Review of Systems (ROS)", "textarea"),
    ("pe", "Physical Exam (PE)", "textarea"),
    ("assessment", "Assessment", "textarea"),
    ("plan", "Plan", "textarea"),
]

# ---------------------------------------------------------------- lab reports
# The demo's actual target. A lab report is already structured (a header + a table
# of analytes), which is exactly why we start here: the model output is easy to
# lay next to the photo and cross-check row by row.
#
# LAB_META — the report header. Best-effort; any unknown field stays "".
LAB_META = [
    ("patient_name", "Patient"),
    ("patient_id", "Patient ID"),
    ("age_sex", "Age / Sex"),
    ("collected", "Collected"),
    ("specimen", "Specimen"),
    ("panel", "Panel / Report"),
    ("performing_lab", "Performing Lab"),
]

# LAB_COLUMNS — one row per analyte/measurement. Order is the on-screen table.
LAB_COLUMNS = [
    ("test", "Test"),
    ("value", "Result"),
    ("unit", "Unit"),
    ("reference_range", "Reference Range"),
    ("flag", "Flag"),
]

# Abnormality flags the model may attach to a row. Anything else (incl. "")
# renders as a normal result. "critical" is the panic-value case.
LAB_FLAGS = ("high", "low", "critical", "abnormal")
