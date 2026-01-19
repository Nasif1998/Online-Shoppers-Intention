import gradio as gr
import pandas as pd
import pickle

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

THRESHOLD = 0.5  

def predict_purchase(
    Administrative, Administrative_Duration,
    Informational, Informational_Duration,
    ProductRelated, ProductRelated_Duration,
    BounceRates, ExitRates, PageValues, SpecialDay,
    Month, OperatingSystems, Browser, Region, TrafficType,
    VisitorType, Weekend
):
    X = pd.DataFrame([{
        "Administrative": Administrative,
        "Administrative_Duration": Administrative_Duration,
        "Informational": Informational,
        "Informational_Duration": Informational_Duration,
        "ProductRelated": ProductRelated,
        "ProductRelated_Duration": ProductRelated_Duration,
        "BounceRates": BounceRates,
        "ExitRates": ExitRates,
        "PageValues": PageValues,
        "SpecialDay": SpecialDay,
        "Month": Month,
        "OperatingSystems": OperatingSystems,
        "Browser": Browser,
        "Region": Region,
        "TrafficType": TrafficType,
        "VisitorType": VisitorType,
        "Weekend": int(Weekend)
    }])


    X["TotalPages"] = X["Administrative"] + X["Informational"] + X["ProductRelated"]
    X["BounceExitRatio"] = X["BounceRates"] / (X["ExitRates"] + 1e-6)

    prob = float(model.predict_proba(X)[0][1])
    return "Purchase" if prob >= THRESHOLD else "No Purchase"


inputs = [
    gr.Number(label="Administrative", value=0),
    gr.Number(label="Administrative_Duration", value=0),
    gr.Number(label="Informational", value=0),
    gr.Number(label="Informational_Duration", value=0),
    gr.Number(label="ProductRelated", value=0),
    gr.Number(label="ProductRelated_Duration", value=0),
    gr.Number(label="BounceRates", value=0),
    gr.Number(label="ExitRates", value=0),
    gr.Number(label="PageValues", value=0),
    gr.Number(label="SpecialDay", value=0),

    gr.Dropdown(["Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","Jan"], label="Month", value="Nov"),

    gr.Number(label="OperatingSystems", value=0),
    gr.Number(label="Browser", value=0),
    gr.Number(label="Region", value=0),
    gr.Number(label="TrafficType", value=0),

    gr.Dropdown(["Returning_Visitor", "New_Visitor", "Other"], label="VisitorType", value="Returning_Visitor"),

    gr.Checkbox(label="Weekend", value=False)
]

app = gr.Interface(
    fn=predict_purchase,
    inputs=inputs,
    outputs="text",
    title="Online Shoppers Purchase Predictor"
)

if __name__ == "__main__":
    app.launch()
