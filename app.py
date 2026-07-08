import streamlit as st
import pickle

# Load the saved model and vectorizer
with open("RFModel.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="Fake Review Detection")

st.title("📝 Fake Review Detection")
st.write("Enter a review below to check whether it is Genuine or Fake.")

review = st.text_area("Enter Review")

if st.button("Predict"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Convert review into TF-IDF features
        review_vector = vectorizer.transform([review])

        # Predict
        prediction = model.predict(review_vector)

        if prediction[0] == 1:
            st.error("🚨 Fake Review")
        else:
            st.success("✅ Genuine Review")