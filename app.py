import gradio as gr
import pandas as pd
import pickle
import numpy as np

# Load trained model

with open("model.pkl", "rb") as file:
    model = pickle.load(file)


# Dropdown options

PLATFORM_OPTIONS = [
    'Wii', 'NES', 'GB', 'DS', 'X360', 'PS3', 'PS2', 'SNES', 'GBA',
    '3DS', 'PS4', 'N64', 'PS5', 'XB', 'PC', '2600', 'PSP', 'XOne',
    'GC', 'WiiU', 'GEN', 'DC', 'PSV', 'SAT', 'SCD', 'WS', 'NG',
    'TG16', '3DO', 'GG', 'PCFX'
]

GENRE_OPTIONS = [
    'Sports', 'Platform', 'Racing', 'Role-Playing', 'Puzzle',
    'Misc', 'Shooter', 'Simulation', 'Action', 'Fighting',
    'Adventure', 'Strategy'
]

PUBLISHER_OPTIONS = [
    'Nintendo', 'Microsoft Game Studios', 'Take-Two Interactive',
    'Sony Computer Entertainment', 'Activision', 'Ubisoft',
    'Bethesda Softworks', 'Electronic Arts', 'Sega',
    'SquareSoft', 'Atari', 'Unknown'
]


# Prediction function

def predict_sales(Platform, Year, Genre, Publisher):

    input_data = pd.DataFrame([{
        "Platform": Platform,
        "Year": Year,
        "Genre": Genre,
        "Publisher": Publisher
    }])

    prediction = model.predict(input_data)[0]
    return round(float(prediction), 2)


# Gradio Interface

app = gr.Interface(
    fn=predict_sales,
    inputs=[
        gr.Dropdown(choices=PLATFORM_OPTIONS, label="Platform"),
        gr.Number(label="Year", value=2010),
        gr.Dropdown(choices=GENRE_OPTIONS, label="Genre"),
        gr.Dropdown(choices=PUBLISHER_OPTIONS, label="Publisher")
    ],
    outputs=gr.Number(label="Predicted Global Sales (millions)"),
    title="Video Game Global Sales Prediction",
    description="Predict global video game sales based on platform, year, genre, and publisher."
)

app.launch()
