import gradio as gr
import pandas as pd
import pickle
import numpy as np

with open("best_model.pkl", "rb") as f: 
    model = pickle.load(f)

def predict_sales(Platform, Year, Genre, Publisher,
                  NA_Sales, EU_Sales, JP_Sales, Other_Sales):
   
   
    total_sales = NA_Sales + EU_Sales + JP_Sales + Other_Sales
    
    input_data = pd.DataFrame([{
        "Platform": Platform,
        "Year": Year,
        "Genre": Genre,
        "Publisher": Publisher,
        "NA_Sales": NA_Sales,
        "EU_Sales": EU_Sales,
        "JP_Sales": JP_Sales,
        "Other_Sales": Other_Sales,
        "Total_Regional_Sales": total_sales,
        "NA_ratio": NA_Sales / (total_sales + 1e-6),
        "EU_ratio": EU_Sales / (total_sales + 1e-6),
        "JP_ratio": JP_Sales / (total_sales + 1e-6),
        "Other_ratio": Other_Sales / (total_sales + 1e-6)
    }])
    
    # Prediction
    pred = model.predict(input_data)[0]
    return np.clip(pred,0,4)

app = gr.Interface(
    fn=predict_sales,
    inputs=[
        gr.Textbox(label="Platform"),
        gr.Number(label="Year"),
        gr.Textbox(label="Genre"),
        gr.Textbox(label="Publisher"),
        gr.Number(label="NA_Sales"),
        gr.Number(label="EU_Sales"),
        gr.Number(label="JP_Sales"),
        gr.Number(label="Other_Sales"),
    ],
    outputs=gr.Number(label="Predicted Global Sales (millions)"),
    title="Video Game Global Sales Prediction",
    description="Enter game details to predict Global Sales (millions)."
)

app.launch()
