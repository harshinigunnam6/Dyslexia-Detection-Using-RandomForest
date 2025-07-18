import streamlit as st
from textblob import TextBlob
import language_tool_python
import joblib
import pandas as pd
from PIL import Image
import base64
import speech_recognition as sr
import random
import json
from datetime import datetime
from difflib import SequenceMatcher
import pytesseract

# Tesseract path for OCR
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# Generate report text file

def generate_report(input_text, prediction, features=None, accuracy=None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""
Dyslexia Detection Report 🧠

🗕️ Date & Time: {now}

📝 Input Text:
{input_text.strip()}

🎯 Prediction: {prediction}
"""

    if features is not None:
        report += f"""
🔍 Feature Analysis:
- Word Count: {features['word_count'].values[0]}
- Avg Word Length: {features['avg_word_len'].values[0]:.2f}
- Spelling Errors: {features['spelling_errors'].values[0]}
- Grammar Errors: {features['grammar_errors'].values[0]}
"""

    if accuracy is not None:
        report += f"\n🎊 Pronunciation Accuracy: {accuracy:.2f}%\n"

    return report

# Load pronunciation test data
with open("pronunciation_test_data.json", "r") as f:
    test_data = json.load(f)

# Show logo and title

def show_header_with_rounded_logo(logo_path, title_text):
    with open(logo_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: flex-start;">
            <img src="data:image/png;base64,{encoded}" style="border-radius: 50%; width: 80px; height: 80px; margin-right: 20px;">
            <h1 style="color: #4B8BBE; font-size: 32px;">🧠 {title_text}</h1>
        </div>
        <hr style="margin-top: 10px;"/>
        """,
        unsafe_allow_html=True
    )

show_header_with_rounded_logo("logo.png", "Dyslexia Detection Using Random Forest")

# Load model
model = joblib.load("rf_custom_model.pkl")
tool = language_tool_python.LanguageTool('en-US')

# Sidebar navigation
page = st.sidebar.radio("Go to", ["Home", "Dyslexia Detection", "Pronunciation Test", "Dyslexia Test", "About"])

# Feature extraction
def extract_features(text):
    words = text.split()
    word_count = len(words)
    avg_word_len = sum(len(w) for w in words) / word_count if word_count > 0 else 0
    corrected = str(TextBlob(text).correct())
    spelling_errors = sum(1 for a, b in zip(text.split(), corrected.split()) if a != b)
    grammar_errors = len(tool.check(text))

    return pd.DataFrame([{
        'word_count': word_count,
        'avg_word_len': avg_word_len,
        'spelling_errors': spelling_errors,
        'grammar_errors': grammar_errors
    }])

# Home page
if page == "Home":
    st.subheader("📘 What is Dyslexia?")
    st.write("""
    Dyslexia is a learning disorder that affects reading due to difficulties in recognizing sounds and how they relate to letters. This project uses AI to make screening easier and faster.
    
    ## 🤖 Why This Project?
    This project uses AI to make dyslexia screening:
    - Fast 💨
    - Accessible 💻
    - Low-cost 💰
    """)

elif page == "Dyslexia Detection":
    st.subheader("✍️ Enter Text for Dyslexia Prediction")
    user_input = st.text_area("Type or paste the text here:", "")

    if st.button("Predict from Text"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            features = extract_features(user_input)
            prediction = model.predict(features)[0]
            st.markdown("### 🧠 Prediction from Typed Text:")
            if prediction == "dyslexic":
                st.error("🔴 Likely Dyslexic")
            else:
                st.success("🟢 Not Dyslexic")

            st.markdown("### 🔍 Feature Analysis:")
            st.write(f"- Word Count: {features['word_count'].values[0]}")
            st.write(f"- Avg Word Length: {features['avg_word_len'].values[0]:.2f}")
            st.write(f"- Spelling Errors: {features['spelling_errors'].values[0]}")
            st.write(f"- Grammar Errors: {features['grammar_errors'].values[0]}")

            report_text = generate_report(user_input, prediction, features)
            st.download_button("📥 Download Report", report_text, file_name="dyslexia_report.txt")

    st.markdown("---")
    st.subheader("🖼️ Upload Handwriting Image for Prediction")

    uploaded_file = st.file_uploader("Upload handwriting image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        extracted_text = pytesseract.image_to_string(image)
        cleaned_text = extracted_text.replace('\n', ' ').strip()

        st.markdown("### 📝 Extracted Text (Edit if needed):")
        user_corrected = st.text_area("Edit extracted text here:", cleaned_text)

        if st.button("Predict from Image Text"):
            if user_corrected.strip() == "":
                st.warning("No text to analyze.")
            else:
                features = extract_features(user_corrected)
                prediction = model.predict(features)[0]

                st.markdown("### 🧠 Prediction from Handwriting:")
                if prediction == "dyslexic":
                    st.error("🔴 Likely Dyslexic")
                else:
                    st.success("🟢 Not Dyslexic")

                st.markdown("### 🔍 Feature Analysis:")
                st.write(f"- Word Count: {features['word_count'].values[0]}")
                st.write(f"- Avg Word Length: {features['avg_word_len'].values[0]:.2f}")
                st.write(f"- Spelling Errors: {features['spelling_errors'].values[0]}")
                st.write(f"- Grammar Errors: {features['grammar_errors'].values[0]}")

                report_text = generate_report(user_corrected, prediction, features)
                st.download_button("📥 Download Report", report_text, file_name="dyslexia_report.txt")
elif page == "Dyslexia Test":
    st.title("🧠 Complete Dyslexia Screening Test")

    st.header("1️⃣ Dictation Task")

    with open("dictation_audio.mp3", "rb") as audio_file:
        st.audio(audio_file.read(), format="audio/mp3")

    with open("dictation_text.txt", "r") as f:
        actual_sentence = f.read().strip()

    st.markdown("### ✍️ Option A: Type what you heard")
    typed_text = st.text_area("Write here")

    st.markdown("### 🖼️ Option B: Upload a handwritten response")
    uploaded_image = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

    extracted_text = ""
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        extracted_text = pytesseract.image_to_string(image).replace("\n", " ").strip()

    st.markdown("---")
    st.header("2️⃣ Pronunciation Task")

    target_word = random.choice(["Psychology", "Encyclopedia", "Intelligence", "Communication"])
    st.markdown(f"### Say this word: **{target_word}**")

    accuracy = 0  # default

    if st.button("🎤 Start Recording"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Recording... speak now.")
            audio = r.listen(source, timeout=10, phrase_time_limit=15)

        try:
            spoken = r.recognize_google(audio)
            st.success(f"🗣️ You said: {spoken}")

            similarity = SequenceMatcher(None, spoken.lower(), target_word.lower()).ratio()
            accuracy = round(similarity * 100, 2)
            st.markdown(f"🔊 Pronunciation Accuracy: **{accuracy}%**")

        except sr.UnknownValueError:
            st.error("😥 Sorry, couldn't understand the audio.")
        except sr.RequestError:
            st.error("🌐 Could not connect to speech service.")

    st.markdown("---")
    st.header("3️⃣ Final Evaluation")

    if st.button("🧠 Evaluate Dyslexia Test"):
        input_text = typed_text if typed_text.strip() else extracted_text

        if not input_text.strip():
            st.warning("Please type or upload some text.")
        else:
            # Compare typed/handwritten with actual sentence
            dictation_score = SequenceMatcher(None, input_text.lower(), actual_sentence.lower()).ratio()
            st.markdown(f"🔹 Dictation Match Accuracy: **{round(dictation_score * 100, 2)}%**")

            # Extract features and predict
            features = extract_features(input_text)
            prediction = model.predict(features)[0]

            # Display result
            st.markdown(f"### 📌 Final Result: **{prediction.upper()}**")
            st.markdown("### 🔍 Feature Analysis:")
            st.write(f"- Word Count: {features['word_count'].values[0]}")
            st.write(f"- Spelling Errors: {features['spelling_errors'].values[0]}")
            st.write(f"- Grammar Errors: {features['grammar_errors'].values[0]}")

            # Generate report
            report = generate_report(input_text, prediction, features, accuracy=accuracy)
            st.download_button("📅 Download Report", report, file_name="final_dyslexia_test.txt")

elif page == "Pronunciation Test":
    st.title("🎤 Pronunciation Practice")

    test_type = st.radio("Choose Test Type:", ["Word", "Sentence"])

    if test_type == "Word":
        target = random.choice(test_data["words"])
    else:
        target = random.choice(test_data["sentences"])

    st.markdown(f"### 👇 Pronounce This {test_type}:")
    st.markdown(f"**🗣️ {target}**")

    if st.button("🎙️ Start Recording"):
        import speech_recognition as sr
        from difflib import SequenceMatcher

        def get_similarity_ratio(a, b):
            return SequenceMatcher(None, a.lower(), b.lower()).ratio()

        r = sr.Recognizer()

        with sr.Microphone() as source:
            st.info("🎙️ Recording... Please speak clearly.")
            audio = r.listen(source, timeout=10, phrase_time_limit=15)


        try:
            st.success("✅ Recording captured! Transcribing...")
            spoken_text = r.recognize_google(audio)
            st.markdown(f"### 📝 You Said:\n{spoken_text}")

            # 📊 Accuracy Calculation
            similarity = SequenceMatcher(None, spoken_text.lower(), target.lower()).ratio()
            accuracy = round(similarity * 100, 2)


            st.markdown(f"### 🎯 Pronunciation Accuracy: {accuracy}%")

            if accuracy > 90:
                st.success("💚 Excellent pronunciation!")
            elif accuracy > 70:
                st.warning("🟡 Good, but could be clearer.")
            else:
                st.error("🔴 Poor pronunciation — try again slowly.")

            # ✅ Generate and offer report only if transcription succeeds
            report_text = generate_report(spoken_text, "Pronunciation Test", accuracy=accuracy)
            st.download_button("📥 Download Pronunciation Report", report_text, file_name="pronunciation_report.txt")

        except sr.UnknownValueError:
            st.error("😢 Sorry, couldn't understand the audio.")
        except sr.RequestError:
            st.error("🌐 Speech service unavailable. Check your connection.")
elif page == "About":
    st.subheader("👩‍💻 Project By")
    st.markdown("""
    **Harshini Gunnam**  
    CSE - Artificial Intelligence & Machine Learning  
    Built with ❤️ using Streamlit, NLP & Random Forest  
    [GitHub Repo](https://github.com/your-username/your-repo)  
    """)