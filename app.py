import gradio as gr
import pandas as pd
import pickle
import numpy as np

MODEL_PATH = "best_model.pkl"
CSV_PATH = "online_shoppers_intention.csv"
TARGET = "Revenue"

THRESHOLD = 0.5   
EPS = 1e-6        


with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


df = pd.read_csv(CSV_PATH)

if "Weekend" in df.columns and df["Weekend"].dtype == "bool":
    df["Weekend"] = df["Weekend"].astype(int)


def get_required_columns(pipeline, df_fallback):
    if not hasattr(pipeline, "named_steps") or "preprocessor" not in pipeline.named_steps:
        return [c for c in df_fallback.columns if c != TARGET]

    pre = pipeline.named_steps["preprocessor"]
    if not hasattr(pre, "transformers_"):
        return [c for c in df_fallback.columns if c != TARGET]

    cols = []
    for name, transformer, colspec in pre.transformers_:
        if name == "remainder":
            continue
        if isinstance(colspec, (list, tuple, np.ndarray)):
            cols.extend(list(colspec))
        elif isinstance(colspec, str):
            cols.append(colspec)

    seen = set()
    ordered = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    if getattr(pre, "remainder", None) == "passthrough":
        base = [c for c in df_fallback.columns if c != TARGET]
        for c in base:
            if c not in seen:
                ordered.append(c)

    if len(ordered) == 0:
        ordered = [c for c in df_fallback.columns if c != TARGET]

    return ordered


REQUIRED_COLS = get_required_columns(model, df)


existing = [c for c in REQUIRED_COLS if c in df.columns]
numeric_cols = df[existing].select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_cols = [c for c in REQUIRED_COLS if c not in numeric_cols]

cat_choices = {}
for c in categorical_cols:
    if c in df.columns:
        cat_choices[c] = sorted(df[c].dropna().astype(str).unique().tolist())
    else:
        cat_choices[c] = []

def get_positive_prob(pipeline, X_row_df):
    probs = pipeline.predict_proba(X_row_df)[0]

    classes = None
    if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
        inner = pipeline.named_steps["model"]
        if hasattr(inner, "classes_"):
            classes = inner.classes_
    if classes is None and hasattr(pipeline, "classes_"):
        classes = pipeline.classes_

    if classes is None:
        return float(probs[1]) if len(probs) > 1 else float(probs[0])

    if 1 in classes:
        pos_label = 1
    elif True in classes:
        pos_label = True
    else:
        pos_label = classes[-1]

    pos_idx = list(classes).index(pos_label)
    return float(probs[pos_idx])

def predict_fn(*inputs):
    row = {c: v for c, v in zip(REQUIRED_COLS, inputs)}

    for c in numeric_cols:
        v = row.get(c)
        if c == "Weekend":
            if isinstance(v, bool):
                row[c] = int(v)
            elif v in [None, ""]:
                row[c] = 0
            else:
                row[c] = int(float(v))
        else:
            row[c] = float(v) if v not in [None, ""] else 0.0

    for c in categorical_cols:
        v = row.get(c)
        if v in [None, ""]:
            row[c] = cat_choices[c][0] if len(cat_choices[c]) > 0 else "unknown"
        else:
            row[c] = str(v)

    X = pd.DataFrame([[row.get(c) for c in REQUIRED_COLS]], columns=REQUIRED_COLS)

    if "TotalPages" in REQUIRED_COLS:
        X["TotalPages"] = (
            X.get("Administrative", 0)
            + X.get("Informational", 0)
            + X.get("ProductRelated", 0)
        )

    if "BounceExitRatio" in REQUIRED_COLS:
        X["BounceExitRatio"] = X.get("BounceRates", 0) / (X.get("ExitRates", 0) + EPS)

    prob = get_positive_prob(model, X)
    return "Purchase" if prob >= THRESHOLD else "No Purchase"

inputs_ui = []
for c in REQUIRED_COLS:
    if c in numeric_cols:
        if c == "Weekend":
            inputs_ui.append(gr.Checkbox(label="Weekend", value=False))
        else:
            inputs_ui.append(gr.Number(label=c, value=0.0))
    else:
        choices = cat_choices.get(c, [])
        if len(choices) > 0:
            inputs_ui.append(gr.Dropdown(label=c, choices=choices, value=choices[0]))
        else:
            inputs_ui.append(gr.Textbox(label=c, value=""))

app = gr.Interface(
    fn=predict_fn,
    inputs=inputs_ui,
    outputs=gr.Textbox(label="Prediction"),
    title="Online Shoppers Purchase Predictor",
    description="Output: Purchase / No Purchase (threshold=0.5). Inputs start at 0.",
    flagging_mode="never"  
)

if __name__ == "__main__":
    app.launch()
