import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
import gradio as gr

# Load the dataset
diabetes_data = pd.read_csv("D:/Downloads/ML projects/diabetes.csv")

# Prepare the data
X = diabetes_data.drop(columns="Outcome", axis=1)
Y = diabetes_data["Outcome"]

# Standardization -- will change all the values in the dataset to values ranged b/w -1 to 1
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the model
classifier = svm.SVC(kernel="linear")
classifier.fit(X_train, Y_train)

# Define the prediction function
def predict_diabetes(input_data):
    input_data = [float(i) for i in input_data.split(',')]
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)
    prediction = classifier.predict(std_data)
    if prediction[0] == 1:
        return "The patient is Diabetic"
    else:
        return "The patient is not Diabetic"

# Create the Gradio interface
gr_interface = gr.Interface(
    fn=predict_diabetes,
    inputs=gr.Textbox(lines=2, placeholder="Enter the features as comma-separated values..."),
    outputs="text",
    title="Diabetes Prediction",
    description="Predict whether the patient is Diabetic or not based on the given readings."
)

# Launch the interface
gr_interface.launch()

'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import gradio as gr

# Load the dataset
diabetes_data = pd.read_csv("D:/Downloads/ML projects/diabetes.csv")

# Prepare the data
X = diabetes_data.drop(columns="Outcome", axis=1)
Y = diabetes_data["Outcome"]

# Standardization
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the model
classifier = RandomForestClassifier(n_estimators=100, random_state=2)
classifier.fit(X_train, Y_train)

# Cross-validation for better evaluation
cv_scores = cross_val_score(classifier, standardized_data, Y, cv=5)
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {np.mean(cv_scores)}')

# Define the prediction function
def predict_diabetes(input_data):
    input_data = [float(i) for i in input_data.split(',')]
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)
    prediction = classifier.predict(std_data)
    if prediction[0] == 1:
        return "The patient is Diabetic"
    else:
        return "The patient is not Diabetic"

# Create the Gradio interface
gr_interface = gr.Interface(
    fn=predict_diabetes,
    inputs=gr.Textbox(lines=2, placeholder="Enter the features as comma-separated values..."),
    outputs="text",
    title="Diabetes Prediction",
    description="Predict whether the patient is Diabetic or not based on the given readings."
)

# Launch the interface
gr_interface.launch()
'''