import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

st.title("🩺 Diabetes Prediction Web App")

# Load your actual CSV from /mnt/data (or a URL if needed)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("diabetes.csv") #try local file
    except FileNotFoundError:
        df= pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", header=None, names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])
        #if local fails, get it from URL
    return df

df = load_data()
st.subheader("📊 Sample Data")
st.write(df.head())

# Features and labels
X = df.drop(columns='Outcome', axis=1)
Y = df['Outcome']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train SVM
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

# Accuracy display
train_accuracy = accuracy_score(Y_train, model.predict(X_train))
test_accuracy = accuracy_score(Y_test, model.predict(X_test))

st.success(f"✅ Training Accuracy: {train_accuracy * 100:.2f}%")
st.info(f"🧪 Test Accuracy: {test_accuracy * 100:.2f}%")

# Sidebar input
st.sidebar.header("👤 Enter Patient Details")

def user_input():
    Pregnancies = st.sidebar.number_input('Pregnancies', 0, 20, step=1)
    Glucose = st.sidebar.slider('Glucose', 0, 200, 100)
    BloodPressure = st.sidebar.slider('BloodPressure', 0, 150, 70)
    SkinThickness = st.sidebar.slider('SkinThickness', 0, 100, 20)
    Insulin = st.sidebar.slider('Insulin', 0, 900, 80)
    BMI = st.sidebar.slider('BMI', 0.0, 70.0, 25.0)
    DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction', 0.0, 3.0, 0.5)
    Age = st.sidebar.slider('Age', 15, 100, 30)

    user_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                    Insulin, BMI, DiabetesPedigreeFunction, Age]])
    return user_data

input_data = user_input()
input_data_scaled = scaler.transform(input_data)
prediction = model.predict(input_data_scaled)

if st.button("Predict"):
    if prediction[0] == 1:
        st.error("🔴 The person is **diabetic**.")
    else:
        st.success("🟢 The person is **not diabetic**.")
