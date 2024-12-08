import streamlit as st
import tensorflow as tf
import numpy as np
import time

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return predictions  # Return predictions as an array

# Disease descriptions, treatment recommendations, and pesticide links
disease_info = {
    'Apple___Apple_scab': {
        'description': "Apple scab is a fungal disease that affects the leaves, fruit, and shoots of apple trees. It causes dark, scabby lesions on fruit and leaves, leading to premature leaf drop.",
        'treatment': [
            "Apply fungicides that are effective against scab.",
            "Prune infected branches to improve air circulation and reduce the spread of the fungus.",
            "Use resistant apple varieties."
        ],
        'pesticide_link': "https://www.amazon.com/s?k=apple+scab+fungicide",
    },
    'Apple___Black_rot': {
        'description': "Black rot is a bacterial disease that affects apple trees, causing dark lesions and blackened fruit. It spreads rapidly and can severely damage the tree.",
        'treatment': [
            "Remove and destroy infected fruit and leaves.",
            "Use copper-based fungicides for control.",
            "Ensure proper spacing and air circulation around the trees."
        ],
        'pesticide_link': "https://www.amazon.com/s?k=black+rot+fungicide",
    },
    'Apple___Cedar_apple_rust': {
        'description': "Cedar apple rust is a fungal disease that affects apple and cedar trees. It causes orange, rust-colored spots on the leaves of apple trees.",
        'treatment': [
            "Prune infected branches and remove fallen leaves.",
            "Apply fungicides during the growing season.",
            "Use resistant apple varieties."
        ],
        'pesticide_link': "https://www.amazon.com/s?k=cedar+apple+rust+fungicide",
    },
    'Apple___healthy': {
        'description': "The apple tree is healthy with no signs of disease.",
        'treatment': [
            "No treatment needed. Continue regular care and maintenance."
        ],
        'pesticide_link': "https://www.amazon.com/s?k=apple+tree+care",
    },
    'Blueberry___healthy': {
        'description': "The blueberry plant is healthy with no signs of disease.",
        'treatment': [
            "No treatment needed. Continue regular care and maintenance."
        ],
        'pesticide_link': "https://www.amazon.com/s?k=blueberry+care",
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'description': "Powdery mildew is a fungal disease that affects cherry trees, causing a white powdery coating on leaves and fruit.",
        'treatment': [
            "Apply fungicides to control the spread.",
            "Remove infected leaves and fruit.",
            "Prune trees to improve air circulation."
        ],
        'pesticide_link': "https://www.amazon.com/s?k=cherry+powdery+mildew+fungicide",
    },
    'Cherry_(including_sour)___healthy': {
        'description': "The cherry tree is healthy with no signs of disease.",
        'treatment': [
            "No treatment needed. Continue regular care and maintenance."
        ],
        'pesticide_link': "https://www.amazon.com/s?k=cherry+tree+care",
    },
    'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot': {
        'description': "Gray leaf spot is a fungal disease that affects corn, causing grayish lesions on leaves, leading to reduced yield.",
        'treatment': [
            "Apply fungicides to control the spread of the disease.",
            "Remove infected leaves and improve soil drainage.",
            "Practice crop rotation to prevent recurrence."
        ],
        'pesticide_link': "https://www.amazon.com/s?k=cercospora+gray+leaf+spot+fungicide",
    },
    'Corn_(maize)___Common_rust_': {
        'description': "Common rust is a fungal disease that affects corn, causing orange pustules on the leaves.",
        'treatment': [
            "Apply fungicides during the early stages of infection.",
            "Remove infected leaves and improve plant spacing."
        ],
        'pesticide_link': "https://www.amazon.com/s?k=common+rust+corn+fungicide",
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'description': "Northern leaf blight is a fungal disease that affects corn, causing long, grayish lesions on the leaves.",
        'treatment': [
            "Apply fungicides and remove infected leaves.",
            "Use resistant corn varieties to minimize damage."
        ],
        'pesticide_link': "https://www.amazon.com/s?k=northern+leaf+blight+corn+fungicide",
    },
    'Corn_(maize)___healthy': {
        'description': "The corn plant is healthy with no signs of disease.",
        'treatment': [
            "No treatment needed. Continue regular care and maintenance."
        ],
        'pesticide_link': "https://www.amazon.com/s?k=corn+care",
    },
    'Grape___Black_rot': {
        'description': "Black rot is a fungal disease that affects grape vines, causing dark lesions on the leaves, stems, and fruit.",
        'treatment': [
            "Remove infected plant parts.",
            "Use fungicides to control the spread of the disease.",
            "Practice good vineyard sanitation."
        ],
        'pesticide_link': "https://www.amazon.com/s?k=grape+black+rot+fungicide",
    },
    'Grape___Esca_(Black_Measles)': {
        'description': "Esca is a fungal disease that affects grapevines, leading to leaf and shoot dieback.",
        'treatment': [
            "Prune infected vines and remove diseased wood.",
            "Ensure proper spacing and air circulation to reduce humidity."
        ],
        'pesticide_link': "https://www.amazon.com/s?k=grape+esca+fungicide",
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'description': "Leaf blight is a fungal disease that causes lesions on grape leaves, reducing photosynthesis and overall plant health.",
        'treatment': [
            "Apply fungicides to control the disease.",
            "Remove infected leaves and ensure proper vineyard maintenance."
        ],
        'pesticide_link': "https://www.amazon.com/s?k=grape+leaf+blight+fungicide",
    },
    'Grape___healthy': {
        'description': "The grape vine is healthy with no signs of disease.",
        'treatment': [
            "No treatment needed. Continue regular care and maintenance."
        ],
        'pesticide_link': "https://www.amazon.com/s?k=grape+vine+care",
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'description': "Huanglongbing, or citrus greening, is a bacterial disease that affects citrus trees, causing yellowing leaves and misshapen fruit.",
        'treatment': [
            "Use antibiotics to control bacterial spread.",
            "Remove infected trees and monitor for new infections."
        ],
        'pesticide_link': "https://www.amazon.com/s?k=citrus+greening+pesticide",
    },
    'Peach___Bacterial_spot': {
        'description': "Bacterial spot is a bacterial disease that affects peach trees, causing dark spots on leaves and fruit.",
        'treatment': [
            "Use copper-based bactericides for control.",
            "Prune infected branches and remove fallen leaves."
        ],
        'pesticide_link': "https://www.amazon.com/s?k=peach+bacterial+spot+pesticide",
    },
    'Peach___healthy': {
        'description': "The peach tree is healthy with no signs of disease.",
        'treatment': [
            "No treatment needed. Continue regular care and maintenance."
        ],
        'pesticide_link': "https://www.amazon.com/s?k=peach+tree+care",
    },
    'Pepper,_bell___Bacterial_spot': {
        'description': "Bacterial spot is a disease that affects bell peppers, causing lesions on the leaves, stems, and fruit.",
        'treatment': [
            "Apply copper-based bactericides to control the spread.",
            "Remove infected plant parts and ensure proper spacing."
        ],
        'pesticide_link': "https://www.amazon.com/s?k=bell+pepper+bacterial+spot+pesticide",
    },
    'Pepper,_bell___healthy': {
        'description': "The bell pepper plant is healthy with no signs of disease.",
        'treatment': [
            "No treatment needed. Continue regular care and maintenance."
        ],
        'pesticide_link': "https://www.amazon.com/s?k=bell+pepper+care",
    },
    # Add more classes as needed...
}

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return predictions  # Return predictions as an array

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """) 

elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

elif app_mode == "Disease Recognition":
    st.header("Upload an Image to Identify Disease")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Run prediction
        st.image(uploaded_image, use_column_width=True)
        prediction = model_prediction(uploaded_image)
        predicted_class = np.argmax(prediction)
        disease_name = list(disease_info.keys())[predicted_class]
        
        # Display Disease Info
        st.subheader(disease_name)
        st.write(disease_info[disease_name]["description"])
        st.write("### Treatment Recommendations:")
        for treatment in disease_info[disease_name]['treatment']:
            st.write(f"- {treatment}")
        
        # Link to pesticide buying page
        st.write(f"### Buy Pesticides:")
        st.write(f"[Click here to buy pesticide for {disease_name}]({disease_info[disease_name]['pesticide_link']})")
