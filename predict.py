import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from dotenv import load_dotenv
from groq import Groq
import requests
import io

# Load the environment variables
load_dotenv()
# Load the pre-trained model
model = load_model('potato_disease_model.h5')

# Define class labels, prevention tips, disease definitions, and treatment instructions
class_labels = ['Early Blight', 'Late Blight', 'Healthy']

prevention_tips = {
    'Early Blight': (
        "1. Use disease-resistant potato varieties.\n"
        "2. Avoid overhead irrigation and use drip irrigation instead.\n"
        "3. Remove and destroy infected plant debris.\n"
        "4. Apply appropriate fungicides if necessary."
    ),
    'Late Blight': (
        "1. Use resistant potato varieties and certified disease-free seed.\n"
        "2. Maintain proper plant spacing for good air circulation.\n"
        "3. Remove and destroy infected plant material promptly.\n"
        "4. Apply fungicides regularly, especially during humid conditions."
    ),
    'Healthy': (
        "Your potato plants are healthy! Continue with regular care:\n"
        "1. Ensure proper watering and nutrition.\n"
        "2. Regularly inspect plants for pests and diseases.\n"
        "3. Maintain good field hygiene."
    )
}

disease_definitions = {
    'Early Blight': (
        "Early Blight is a common disease caused by the fungus *Alternaria solani*. It affects the leaves, stems, and tubers of potatoes, "
        "leading to dark, concentric lesions on the leaves, which can cause early defoliation and reduced yield."
    ),
    'Late Blight': (
        "Late Blight is a serious disease caused by the oomycete *Phytophthora infestans*. It results in water-soaked lesions on leaves, "
        "stems, and tubers, which can lead to rapid plant death and significant yield loss. It thrives in cool, moist conditions."
    ),
    'Healthy': (
        "Your potato plants show no signs of disease. Continue with routine care and monitoring to ensure they remain in good health."
    )
}

treatment_instructions = {
    'Early Blight': (
        "To treat Early Blight:\n"
        "1. Apply fungicides such as chlorothalonil, azoxystrobin, or pyraclostrobin according to label instructions.\n"
        "2. Regularly monitor and inspect plants for signs of the disease.\n"
        "3. Remove any affected plant parts and dispose of them properly.\n"
        "4. Rotate crops to avoid soil-borne pathogens."
    ),
    'Late Blight': (
        "To treat Late Blight:\n"
        "1. Use fungicides like mefenoxam or metalaxyl to manage the disease.\n"
        "2. Implement preventive measures such as using resistant varieties and maintaining proper plant spacing.\n"
        "3. Regularly scout fields and remove infected plants promptly.\n"
        "4. Ensure proper drainage and avoid overhead irrigation."
    ),
    'Healthy': (
        "No treatment needed. Continue with regular care and monitoring to maintain plant health."
    )
}

def preprocess_image(img):
    """Resize and normalize image for model prediction."""
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def is_potato_leaf(img):
    """Check if the uploaded image is likely a potato leaf based on color properties."""
    hsv_img = img.convert("HSV")
    img_array = np.array(hsv_img)
    avg_hue = np.mean(img_array[:, :, 0])
    avg_saturation = np.mean(img_array[:, :, 1])
    avg_value = np.mean(img_array[:, :, 2])
    return 25 < avg_hue < 85 and avg_saturation > 40 and avg_value > 40

def predict_disease(image):
    """Predict the disease of a potato leaf from the image."""
    img = Image.open(image)
    if is_potato_leaf(img):
        img_array = preprocess_image(img)
        predictions = model.predict(img_array)
        confidence = np.max(predictions)
        predicted_class_index = np.argmax(predictions)

        if predicted_class_index < len(class_labels):
            predicted_class = class_labels[predicted_class_index]
        else:
            return "Prediction index out of bounds.", None
        
        if confidence < 0.5:
            return "The model could not confidently identify the disease in the image.", None
        else:
            result = {
                'prediction': predicted_class,
                'confidence': confidence * 100,  # Convert confidence to percentage
                'definition': disease_definitions.get(predicted_class, "Definition not available."),
                'prevention': prevention_tips.get(predicted_class, "Prevention tips not available."),
                'treatment': treatment_instructions.get(predicted_class, "Treatment instructions not available.")
            }
            return None, result
    else:
        return "The uploaded image does not appear to be a potato leaf. Please upload a valid image.", None

def chatbot_question(question):
    """Handle chatbot interactions."""
    groq_api_key = os.getenv('GROQ_API_KEY', '')
    client = Groq(api_key=groq_api_key)

    if question:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ],
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=False,
        )
        return chat_completion.choices[0].message.content

    return "No question asked."


# Streamlit interface
st.title("Potato Disease Classifier and Chatbot")

# Image classification section
st.header("Potato Leaf Disease Classification")
uploaded_file = st.file_uploader("Upload a potato leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    error, result = predict_disease(uploaded_file)
    if error:
        st.error(error)
    else:
        st.image(uploaded_file, caption="Uploaded Potato Leaf", use_column_width=True)
        st.subheader(f"Prediction: {result['prediction']}")
        st.write(f"Confidence: {result['confidence']:.2f}%")
        st.write(f"Definition: {result['definition']}")
        st.write(f"Prevention Tips: {result['prevention']}")
        st.write(f"Treatment Instructions: {result['treatment']}")

# Chatbot section
st.header("Chatbot")
question = st.text_input("Ask a question")
if st.button("Submit Question"):
    response = chatbot_question(question)
    st.write(f"Chatbot Response: {response}")
