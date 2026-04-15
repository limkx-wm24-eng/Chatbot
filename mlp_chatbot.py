import pandas as pd
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Running MLP Chatbot")

# Load dataset
df = pd.read_csv("faq_dataset.csv", engine="python", on_bad_lines="skip")

# Basic cleaning
def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text

# Remove empty rows first
df = df.dropna(subset=["text", "intent"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()
df = df[(df["text"] != "") & (df["intent"] != "")]

# Clean text
df["text"] = df["text"].apply(clean_text)

# Remove unknown class from training
df = df[~df["intent"].isin(["unknown", "greeting", "thanks", "goodbye"])].copy()
X = df["text"]
y = df["intent"]

print("Total rows:", len(df))
print("Number of classes:", y.nunique())
print("\nExamples per class:")
print(y.value_counts())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Pipeline
model = Pipeline([
        ("tfidf", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.90,
        sublinear_tf=True
    )),
    ("clf", MLPClassifier(
        hidden_layer_sizes=(128,64),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        learning_rate_init=0.001,
        max_iter=800,
        random_state=42,
        early_stopping=False
    ))
])

# Train
model.fit(X_train, y_train)
joblib.dump(model, "mlp_chatbot_model.pkl")

# Evaluate
y_pred = model.predict(X_test)
print("\nMLP Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

# Response bank
responses = {
    "admission": "You can apply online through the TARUMT website. There are multiple intakes such as January, May or June, and September. Make sure to prepare the required documents before submitting your application.",
    "fees": "Tuition fees at TARUMT vary depending on the programme and level of study. Generally, diploma and degree fees range from about RM18,000 to RM44,000. You should check the latest fee structure and payment deadline for your course.",
    "courses": "TARUMT offers a wide range of programmes including diploma, degree, and postgraduate studies in areas such as Information Technology, Business, Engineering, Science, and more.",
    "timetable": "You can view your timetable through the TARUMT student portal after registration. The portal will show your class schedule and timetable updates.",
    "contact": "You can contact TARUMT through the official website, email, or hotline for more information and support.",
    "greeting": "Hello! How can I assist you today?",
    "thanks": "You're welcome! Feel free to ask anything else.",
    "goodbye": "Goodbye! Have a great day.",
    "unknown": "Sorry, I am not sure about that. You can ask about TARUMT admission, fees, courses, facilities, timetable, scholarships, or student life.",
    "general": "TARUMT is a well-known private university in Malaysia that offers affordable and quality education. Its main campus is located in Setapak, Kuala Lumpur.They are exceeding 34,000 studentsstudying in TARUMT",
    "facility": "TARUMT provides facilities such as libraries, computer labs, WiFi, sports facilities, study areas, cafeterias, and student accommodation.",
    "student_life": "Student life at TARUMT is vibrant, with clubs, societies, events, sports, and various campus activities for students to join.",
    "international": "TARUMT welcomes international students and provides support services such as admission guidance, visa-related assistance, and orientation support.",
    "scholarship": "TARUMT offers scholarships and financial aid based on academic performance, including merit-based support and possible tuition fee waivers."
}

def detect_small_talk(user_input: str):
    text = clean_text(user_input)

    greeting_words = ["hi", "hello", "hey"]
    thanks_words = ["thanks", "thank you", "appreciate"]
    goodbye_words = ["bye", "goodbye", "see you"]

    faq_keywords = [
        "fee", "fees", "payment", "tuition",
        "course", "courses", "programme", "program", "diploma", "degree",
        "apply", "admission", "register", "enroll",
        "timetable", "schedule", "class",
        "contact", "phone", "email", "hotline"
    ]

    has_faq = any(word in text for word in faq_keywords)

    if not has_faq:
        if any(word in text for word in greeting_words):
            return "greeting"
        if any(word in text for word in thanks_words):
            return "thanks"
        if any(word in text for word in goodbye_words):
            return "goodbye"

    return None


THRESHOLD = 0.45

print("\nUniversity FAQ Chatbot (MLP)")
while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["exit", "quit", "goodbye"]:
        print("Bot: Goodbye.")
        break

    cleaned = clean_text(user_input)
    small_talk_intent = detect_small_talk(user_input)

    location_keywords = [
        "where is tarumt",
        "tarumt location",
        "main campus",
        "campus located",
        "located in",
        "setapak",
        "kuala lumpur",
        "where is the university",
        "where is the campus",
        "location of tarumt"
    ]

    if any(keyword in cleaned for keyword in location_keywords):
        print("Predicted intent: general")
        print("Bot:", responses["general"])
        continue

    student_keywords = [
        "how many student",
        "number of student",
        "total student",
        "student population"
    ]

    if any(keyword in cleaned for keyword in student_keywords):
        print("Predicted intent: general")
        print("Bot:", responses["general"])
        continue

    if small_talk_intent:
        print("Predicted intent:", small_talk_intent)
        print("Bot:", responses[small_talk_intent])
        continue

    probs = model.predict_proba([cleaned])[0]
    sorted_idx = probs.argsort()[::-1]

    best_index = sorted_idx[0]
    second_index = sorted_idx[1]

    best_score = probs[best_index]
    second_score = probs[second_index]
    gap = best_score - second_score

    intent = model.classes_[best_index]
    second_intent = model.classes_[second_index]

    if best_score >= 0.75:
        confidence_level = "High"
    elif best_score >= 0.50:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"
    print("Confidence level:", confidence_level)

    if best_score < THRESHOLD or gap < 0.08:
        print("Predicted intent: unknown")
        print("Second intent:", second_intent)
        print(f"Confidence: {best_score:.2f}")
        print(f"Gap: {gap:.2f}")
        print("Bot:", responses["unknown"])
    else:
        print("Predicted intent:", intent)
        print("Second intent:", second_intent)
        print(f"Confidence: {best_score:.2f}")
        print(f"Gap: {gap:.2f}")
        print("Bot:", responses.get(intent, responses["unknown"]))