import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

print("Running SVM Chatbot")

df = pd.read_csv("faq_dataset.csv")

def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text

df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["intent"]

print("Total rows:", len(df))
print("Number of classes:", y.nunique())
print("\nExamples per class:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LinearSVC())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nSVM Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

responses = {
    "admission": "You can apply through the university admission portal and submit the required documents.",
    "fees": "You can check tuition fees and payment deadlines through the finance office or student portal.",
    "courses": "The university offers programmes such as IT, Business, and other diploma or degree courses.",
    "timetable": "You can view your class timetable through the student portal.",
    "contact": "You can contact the university through the official email or phone number listed on the website.",
    "greeting": "Hello. How can I help you today?",
    "thanks": "You are welcome.",
    "goodbye": "Goodbye. Have a nice day."
}

print("\nUniversity FAQ Chatbot (SVM)")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit","goodbye"]:
        print("Bot: Goodbye.")
        break

    cleaned = clean_text(user_input)
    intent = model.predict([cleaned])[0]
    print("Predicted intent:", intent)
    print("Bot:", responses.get(intent, "Sorry, I do not understand your question."))
