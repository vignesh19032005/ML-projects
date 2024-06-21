import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import gradio as gr

# Load the dataset
sonar_data = pd.read_csv("D:/Downloads/ML projects/Copy of sonar data.csv", header=None)

# Prepare the data
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Define the prediction function
def predict_sonar(input_data):
    input_data = [float(i) for i in input_data.split(',')]
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    if prediction[0] == "R":
        return "The object is a Rock"
    else:
        return "The object is a Mine"

# Create the Gradio interface
gr_interface = gr.Interface(
    fn=predict_sonar,
    inputs=gr.Textbox(lines=2, placeholder="Enter the 60 features as comma-separated values..."),
    outputs="text",
    title="Sonar Object Prediction",
    description="Predict whether the object is a Rock or a Mine based on the given sonar readings."
)

# Launch the interface
gr_interface.launch()


