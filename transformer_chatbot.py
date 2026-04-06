import pandas as pd
import re
import string
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

print("Running Transformer Chatbot")

# =========================
# Basic cleaning
# =========================
def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


# =========================
# Load dataset
# =========================
df = pd.read_csv("faq_dataset.csv")

# Remove empty rows if any
df = df.dropna(subset=["text", "intent"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()
df = df[(df["text"] != "") & (df["intent"] != "")]

# Clean text
df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["intent"]

print("Total rows:", len(df))
print("Number of classes:", y.nunique())
print("\nExamples per class:")
print(y.value_counts())

# =========================
# Encode labels
# =========================
label_list = sorted(y.unique())
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

df["label"] = df["intent"].map(label2id)

# =========================
# Train/test split
# =========================
train_df, test_df = train_test_split(
    df[["text", "intent", "label"]],
    test_size=0.25,
    random_state=42,
    stratify=df["intent"]
)

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(
    train_df[["text", "label"]].reset_index(drop=True)
)
test_dataset = Dataset.from_pandas(
    test_df[["text", "label"]].reset_index(drop=True)
)

# =========================
# Tokenizer + model
# =========================
# DistilBERT is lighter and easier for student projects
model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(batch):
    return tokenizer(batch["text"], truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# =========================
# Metrics
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}

# =========================
# Training arguments
# =========================
training_args = TrainingArguments(
    output_dir="./transformer_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    report_to="none"
)

# =========================
# Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# =========================
# Train
# =========================
trainer.train()

# =========================
# Evaluate
# =========================
pred_output = trainer.predict(test_dataset)
y_pred = np.argmax(pred_output.predictions, axis=1)
y_true = test_df["label"].to_numpy()

print("\nTransformer Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(label_list))),
    target_names=label_list,
    zero_division=0
))

# =========================
# Response bank
# =========================
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

# =========================
# Chat loop
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("\nUniversity FAQ Chatbot (Transformer)")
while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit", "goodbye"]:
        print("Bot: Goodbye.")
        break

    cleaned = clean_text(user_input)

    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = int(torch.argmax(outputs.logits, dim=1).item())

    intent = id2label[pred_id]

    print("Predicted intent:", intent)
    print("Bot:", responses.get(intent, "Sorry, I do not understand your question."))