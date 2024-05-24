import streamlit as st
import requests
import base64

# Function to encode image file to base64
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
    return encoded_string

# Streamlit UI
def main():
    st.title("Machine Learning Prediction")
    image_path = st.file_uploader("Upload an image", type=["jpg"])
    if st.button("Predict") and image_path is not None:
        encoded_image = encode_image(image_path)
        response = requests.post("http://localhost:8000/predict", json={"image": encoded_image})
        if response.status_code == 200:
            set_id = response.json()["set_id"]
            poke_id = response.json()["poke_id"]
            st.success(f"Set ID: {set_id}, Poke ID: {poke_id}")
        else:
            st.error("Failed to get prediction")

if __name__ == "__main__":
    main()
