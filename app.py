import numpy as np
import pickle
import streamlit as st

# Load the saved model
with open('label.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Function for crop recommendation
def crop_recommendation(input_data):
    # Convert input data to float
    input_data = [float(value) for value in input_data[:4]]  # Only first 4 features: N, P, K, pH

    # Convert input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
 
    # Get probabilities for each class
    probabilities = loaded_model.predict_proba(input_data_reshaped)[0]

    # Get index of the predicted crop
    crop_index = np.argmax(probabilities)

    # Mapping crop index to crop name
    crops = ["Pomegranate", "Mango", "Grapes", "Mulberry", "Banana", "Apple"]
    recommended_crop = crops[crop_index]

    # Debugging: Print probabilities for each crop
    st.write("Probabilities for each crop:")
    for i, crop in enumerate(crops):
        st.write(f"{crop}: {probabilities[i]}")
    return recommended_crop

def main():
    # Title of the web app
    st.title('Crop Recommendation Web App')

    # Input fields for soil nutrients
    N = st.text_input('Nitrogen (in ppm)')
    P = st.text_input('Phosphorus (in ppm)')
    K = st.text_input('Potassium (in ppm)')
    pH = st.text_input('pH')

    # Code for crop recommendation
    recommendation = ''

    # Button for recommendation
    if st.button('Get Crop Recommendation'):
        input_data = [N, P, K, pH]  # Adjusted to include only first 4 features
        recommendation = crop_recommendation(input_data)

    st.success('Recommended Crop: ' + recommendation)

if __name__ == '__main__':
    main()
