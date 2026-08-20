# ============================================================
# SPAM EMAIL DETECTION SYSTEM
# Complete Application
#
# Person 1 -> Dataset & Data Preprocessing
# Person 2 -> Feature Extraction using TF-IDF
# Person 3 -> Machine Learning using Naive Bayes
# Person 4 -> Website/UI & Integration
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Spam Email Detection",
    page_icon="📧",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin-top: 20px;
}

.info-box {
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📧 Spam Email Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning based Spam / Ham Email Classification'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# TEXT CLEANING FUNCTION
# PERSON 1
# ============================================================

def clean_text(text):
    """
    Clean email text before feature extraction.
    """

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " ",
        text
    )

    # Remove non-alphabetic characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# LOAD DATASET
# PERSON 1
# ============================================================

@st.cache_data
def load_dataset(uploaded_file=None):

    if uploaded_file is not None:

        try:
            data = pd.read_csv(
                uploaded_file,
                encoding="latin-1"
            )
        except Exception:
            data = pd.read_csv(uploaded_file)

    else:

        if not os.path.exists("spam.csv"):
            return None

        try:
            data = pd.read_csv(
                "spam.csv",
                encoding="latin-1"
            )
        except Exception:
            data = pd.read_csv("spam.csv")

    return data


# ============================================================
# PREPROCESS DATASET
# PERSON 1
# ============================================================

def preprocess_dataset(data):

    data = data.copy()

    # --------------------------------------------------------
    # Identify columns
    # --------------------------------------------------------

    columns = data.columns.tolist()

    # Common SMS Spam Collection dataset
    if "v1" in columns and "v2" in columns:

        data = data[["v1", "v2"]]

        data.columns = [
            "label",
            "message"
        ]

    # Already correctly named
    elif "label" in columns and "message" in columns:

        data = data[[
            "label",
            "message"
        ]]

    # Alternative common format
    elif "category" in columns and "message" in columns:

        data = data[[
            "category",
            "message"
        ]]

        data.columns = [
            "label",
            "message"
        ]

    else:

        # Use first two columns
        if len(columns) >= 2:

            data = data.iloc[:, :2]

            data.columns = [
                "label",
                "message"
            ]

        else:

            raise ValueError(
                "Dataset must contain at least two columns."
            )

    # --------------------------------------------------------
    # Remove missing values
    # --------------------------------------------------------

    data = data.dropna(
        subset=[
            "label",
            "message"
        ]
    )

    # --------------------------------------------------------
    # Standardize labels
    # --------------------------------------------------------

    data["label"] = (
        data["label"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # --------------------------------------------------------
    # Keep only spam and ham
    # --------------------------------------------------------

    data = data[
        data["label"].isin([
            "spam",
            "ham"
        ])
    ]

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    data = data.drop_duplicates()

    # --------------------------------------------------------
    # Clean messages
    # --------------------------------------------------------

    data["cleaned_message"] = (
        data["message"]
        .apply(clean_text)
    )

    # --------------------------------------------------------
    # Remove empty messages
    # --------------------------------------------------------

    data = data[
        data["cleaned_message"].str.strip() != ""
    ]

    # Reset index
    data = data.reset_index(drop=True)

    return data


# ============================================================
# TRAIN MACHINE LEARNING MODEL
# PERSON 2 + PERSON 3
# ============================================================

@st.cache_resource
def train_model(data):

    X = data["cleaned_message"]
    y = data["label"]

    # --------------------------------------------------------
    # TF-IDF FEATURE EXTRACTION
    # PERSON 2
    # --------------------------------------------------------

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_tfidf = vectorizer.fit_transform(X)

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # PERSON 3
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # NAIVE BAYES MODEL
    # PERSON 3
    # --------------------------------------------------------

    model = MultinomialNB()

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[
            "ham",
            "spam"
        ]
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "Ham",
            "Spam"
        ],
        output_dict=True,
        zero_division=0
    )

    return (
        vectorizer,
        model,
        accuracy,
        cm,
        report,
        X_train,
        X_test,
        y_train,
        y_test,
        y_pred
    )


# ============================================================
# SIDEBAR
# PERSON 4 - UI
# ============================================================

st.sidebar.title("⚙️ Project Menu")

page = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Home",
        "📊 Dataset",
        "🤖 Model Performance",
        "📧 Spam Detection"
    ]
)


# ============================================================
# DATASET UPLOAD
# ============================================================

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload Spam Dataset",
    type=["csv"]
)


# ============================================================
# LOAD DATA
# ============================================================

try:

    raw_data = load_dataset(
        uploaded_file
    )

    if raw_data is None:

        st.warning(
            "⚠️ spam.csv was not found. "
            "Please upload your CSV dataset from the sidebar."
        )

        st.stop()

    data = preprocess_dataset(
        raw_data
    )

except Exception as e:

    st.error(
        f"Dataset Error: {e}"
    )

    st.stop()


# ============================================================
# TRAIN MODEL
# ============================================================

try:

    (
        vectorizer,
        model,
        accuracy,
        cm,
        report,
        X_train,
        X_test,
        y_train,
        y_test,
        y_pred

    ) = train_model(data)

except Exception as e:

    st.error(
        f"Model training error: {e}"
    )

    st.stop()


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.header(
        "Welcome to Spam Email Detection 📧"
    )

    st.write(
        """
        This application uses Natural Language Processing
        and Machine Learning to classify messages as
        **Spam** or **Ham (Not Spam)**.
        """
    )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Emails",
            len(data)
        )

    with col2:

        st.metric(
            "Spam Emails",
            int(
                (data["label"] == "spam").sum()
            )
        )

    with col3:

        st.metric(
            "Ham Emails",
            int(
                (data["label"] == "ham").sum()
            )
        )

    with col4:

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    st.markdown("---")

    st.subheader(
        "🔄 How the System Works"
    )

    steps = [
        "1️⃣ Dataset Collection",
        "2️⃣ Data Cleaning",
        "3️⃣ Text Preprocessing",
        "4️⃣ TF-IDF Feature Extraction",
        "5️⃣ Train/Test Split",
        "6️⃣ Naive Bayes Training",
        "7️⃣ Email Classification",
        "8️⃣ Result Display"
    ]

    for step in steps:

        st.write(step)


# ============================================================
# DATASET PAGE
# ============================================================

elif page == "📊 Dataset":

    st.header(
        "📊 Dataset & Data Preprocessing"
    )

    st.subheader(
        "Dataset Overview"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Records",
            len(data)
        )

    with col2:

        st.metric(
            "Spam",
            int(
                (data["label"] == "spam").sum()
            )
        )

    with col3:

        st.metric(
            "Ham",
            int(
                (data["label"] == "ham").sum()
            )
        )

    st.markdown("---")

    st.subheader(
        "Label Distribution"
    )

    distribution = (
        data["label"]
        .value_counts()
    )

    st.bar_chart(
        distribution
    )

    st.markdown("---")

    st.subheader(
        "Preprocessed Dataset"
    )

    display_data = data[
        [
            "label",
            "message",
            "cleaned_message"
        ]
    ]

    st.dataframe(
        display_data,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader(
        "Dataset Information"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "**Columns:**"
        )

        st.write(
            data.columns.tolist()
        )

    with col2:

        st.write(
            "**Missing Values:**"
        )

        st.write(
            data.isnull().sum()
        )


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

elif page == "🤖 Model Performance":

    st.header(
        "🤖 Machine Learning Model Performance"
    )

    # Accuracy
    st.subheader(
        "Model Accuracy"
    )

    st.metric(
        "Naive Bayes Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    st.progress(
        float(accuracy)
    )

    st.markdown("---")

    # Confusion Matrix
    st.subheader(
        "Confusion Matrix"
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Ham",
            "Actual Spam"
        ],
        columns=[
            "Predicted Ham",
            "Predicted Spam"
        ]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    st.markdown("---")

    # Classification Report
    st.subheader(
        "Classification Report"
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader(
        "Training Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Training Samples",
            X_train.shape[0]
        )

    with col2:

        st.metric(
            "Testing Samples",
            X_test.shape[0]
        )

    with col3:

        st.metric(
            "TF-IDF Features",
            X_train.shape[1]
        )


# ============================================================
# SPAM DETECTION PAGE
# PERSON 4 - FINAL APPLICATION
# ============================================================

elif page == "📧 Spam Detection":

    st.header(
        "📧 Check Your Email"
    )

    st.write(
        "Enter an email/message below and the trained "
        "machine learning model will classify it."
    )

    email_text = st.text_area(
        "Enter Email Message",
        height=220,
        placeholder=(
            "Example: Congratulations! "
            "You have won a free prize..."
        )
    )

    if st.button(
        "🔍 Check Email",
        use_container_width=True
    ):

        if not email_text.strip():

            st.warning(
                "Please enter an email message."
            )

        else:

            # Clean input
            cleaned_email = clean_text(
                email_text
            )

            # TF-IDF transform
            email_vector = vectorizer.transform(
                [cleaned_email]
            )

            # Prediction
            prediction = model.predict(
                email_vector
            )[0]

            # Probability
            probability = model.predict_proba(
                email_vector
            )[0]

            # Find probability for predicted class
            classes = model.classes_

            predicted_index = list(
                classes
            ).index(
                prediction
            )

            confidence = probability[
                predicted_index
            ]

            st.markdown("---")

            # Result
            if prediction == "spam":

                st.error(
                    "🚨 SPAM EMAIL"
                )

                st.markdown(
                    f"""
                    <div class="result-box">
                    🚨 This message is likely SPAM
                    <br>
                    Confidence: {confidence * 100:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.success(
                    "✅ HAM / NOT SPAM"
                )

                st.markdown(
                    f"""
                    <div class="result-box">
                    ✅ This message appears to be HAM
                    <br>
                    Confidence: {confidence * 100:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("---")

            st.subheader(
                "Processed Text"
            )

            st.code(
                cleaned_email
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Spam Email Detection System | "
    "Python • Pandas • TF-IDF • Naive Bayes • Streamlit"
)