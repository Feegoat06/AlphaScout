# InterviewHomework

Course scripts for auditing SAS extracts, merging signals with CRSP-style returns and Fama-French/Carhart factors, and building a monthly return-prediction-style panel.

## Environment

- **Python**: 3.10 or newer recommended.
- **Dependencies**: see [`requirements.txt`](requirements.txt).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If `pandas.read_sas(..., format="sas7bdat")` fails, ensure `pyreadstat` is installed (listed in `requirements.txt`).

## Data files

Place these SAS binaries in the **project root** (same folder as the `.py` files):

| File | Role |
|------|------|
| `signals_raw_plus.sas7bdat` | Signals / candidate predictors |
| `msf.sas7bdat` | Monthly stock-level returns (`RET`) and identifiers |
| `factors_monthly.sas7bdat` | Factor portfolios and risk-free rate (`rf`, `mktrf`, etc.) |

Missing files raise `FileNotFoundError` when a script tries to load them.

## How to run

From the repository root:

```bash
cd InterviewHomework          # adjust path if needed

python part1_script.py              # Part 1: audits, duplicates, merge-key notes
python part2_script.py              # Part 2: selected signals report
python part3_script.py              # Part 3: monthly panel + essential-column preview
python return_prediction_model.py   # optional: helpers from part2_script / part3_script
```

See each script’s module docstring for step-level detail. Part 1 includes monthly snapshots for configured example months (e.g. `1995-01`, `1995-02`). Part 3 prints a slim **essential-columns** preview at the end.

## Scripts (current layout)

| Script | Purpose |
|--------|---------|
| `part1_script.py` | Row counts, date ranges, required fields, duplicate keys, merge guidance |
| `part2_script.py` | `SELECTED_SIGNALS` definitions and narrative |
| `part3_script.py` | `cal_ym`, forward `ret_fwd`, factor merge, `excess_ret`, `prediction_panel_essential()` |
| `return_prediction_model.py` | Example pipeline importing part2/part3 helpers |

## Notes

- Paths are relative to the project root; no external config file.
- For the full problem statement, refer to **`TakeHomeProblemMemo.pdf`** (add it to the submission bundle if it is not already in this folder).
- **AI usage**
  - **ChatGPT** and **Cursor** were used to clarify finance concepts and to structure the workflow.
  - The code is **AI-assisted**; it has been reviewed manually and is intended to match the assignment logic.
  - **Reliability practices:** I tend to use Cursor’s plan mode to align on tasks and priorities before implementation. I always review the code and aim to understand the full pipeline. When something is unclear, I ask the AI to clarify—both the plan and the code—and use back-and-forth discussion to improve understanding and catch mistakes.
