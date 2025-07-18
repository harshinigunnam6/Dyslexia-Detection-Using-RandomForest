import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from textblob import TextBlob
import language_tool_python
import joblib


# Load your dataset
df = pd.read_csv("dyslexia_dataset_500.csv")

# Initialize grammar checker
tool = language_tool_python.LanguageTool('en-US')

# Feature extraction function
def extract_features(text):
    words = text.split()
    word_count = len(words)
    avg_word_len = sum(len(w) for w in words) / word_count if word_count > 0 else 0

    # Spelling error count using TextBlob
    corrected = str(TextBlob(text).correct())
    spelling_errors = sum(1 for a, b in zip(text.split(), corrected.split()) if a != b)

    # Grammar error count using LanguageTool
    grammar_errors = len(tool.check(text))

    return pd.Series({
        'word_count': word_count,
        'avg_word_len': avg_word_len,
        'spelling_errors': spelling_errors,
        'grammar_errors': grammar_errors
    })

# Apply to the dataset
features = df['text'].apply(extract_features)
X = features
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "rf_custom_model.pkl")
print("✅ Model saved as rf_custom_model.pkl")
