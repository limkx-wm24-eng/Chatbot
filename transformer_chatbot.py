import os
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
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

print("Running Improved Transformer Chatbot")

MODEL_DIR = "final_transformer_model"
CONFIDENCE_THRESHOLD = 0.40

# =========================
# Basic cleaning
# =========================
def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text

# Simple typo / shorthand normalization
COMMON_TYPOS = {
    "hell": "hello",
    "helo": "hello",
    "hii": "hi",
    "thx": "thanks",
    "pls": "please",
    "gudbye": "goodbye",
    "feee": "fee"
}

def normalize_input(text: str) -> str:
    cleaned = clean_text(text)
    words = cleaned.split()
    words = [COMMON_TYPOS.get(word, word) for word in words]
    return " ".join(words)


# =========================
# Load dataset
# =========================
df = pd.read_csv("faq_dataset.csv")

df = df.dropna(subset=["text", "intent"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()
df = df[(df["text"] != "") & (df["intent"] != "")]
df["text"] = df["text"].apply(normalize_input)

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

print("\nLabel mapping:")
print(label2id)

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

train_dataset = Dataset.from_pandas(
    train_df[["text", "label"]].reset_index(drop=True)
)
test_dataset = Dataset.from_pandas(
    test_df[["text", "label"]].reset_index(drop=True)
)

# =========================
# Tokenizer
# =========================
model_name = "distilbert-base-uncased"

if os.path.exists(MODEL_DIR):
    print(f"\nLoading saved model from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
else:
    print(f"\nLoading base tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# =========================
# Model
# =========================
if os.path.exists(MODEL_DIR):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
else:
    model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.3
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
# Train only if no saved model
# =========================
if not os.path.exists(MODEL_DIR):
    training_args = TrainingArguments(
        output_dir="./transformer_results",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16, # make it 8 if PC cannot handle
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": test_dataset,
        "processing_class": tokenizer,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics
    }

    # Some transformers versions use eval_strategy, some evaluation_strategy
    try:
        training_args = TrainingArguments(
            output_dir="./transformer_results",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            warmup_ratio=0.1,
            load_best_model_at_end=True,
            report_to="none"
        )
        trainer_kwargs["args"] = training_args
    except TypeError:
        training_args = TrainingArguments(
            output_dir="./transformer_results",
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=6,
            weight_decay=0.01,
            warmup_ratio=0.1,
            load_best_model_at_end=True,
            report_to="none"
        )
        trainer_kwargs["args"] = training_args

    trainer = Trainer(**trainer_kwargs)

    print("\nTraining model...")
    trainer.train()

    print(f"\nSaving model to: {MODEL_DIR}")
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
else:
    print("\nSaved model found. Skipping retraining.")

# =========================
# Evaluation
# =========================
pred_trainer_args = None
try:
    pred_trainer_args = TrainingArguments(
        output_dir="./transformer_results_eval",
        evaluation_strategy="no",
        report_to="none"
    )
except TypeError:
    pred_trainer_args = TrainingArguments(
        output_dir="./transformer_results_eval",
        eval_strategy="no",
        report_to="none"
    )

pred_trainer = Trainer(
    model=model,
    args=pred_trainer_args,
    processing_class=tokenizer,
    data_collator=data_collator
)

pred_output = pred_trainer.predict(test_dataset)
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
    "goodbye": "Goodbye. Have a nice day.",
    "unknown": "Sorry, I can only answer questions about admission, fees, courses, timetable, contact, greetings, thanks, and goodbye."
}

# =========================
# Chat loop
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("\nUniversity FAQ Chatbot (Transformer)")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit", "goodbye"]:
        print("Bot: Goodbye.")
        break

    cleaned = normalize_input(user_input)

    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        probs_np = probs.cpu().numpy()[0]
        sorted_probs = np.sort(probs_np)

        top1 = sorted_probs[-1]
        top2 = sorted_probs[-2] if len(sorted_probs) > 1 else 0
        margin = top1 - top2

        pred_id = int(np.argmax(probs_np))
        intent = id2label[pred_id]
        confidence = float(top1)

    print(f"Predicted intent: {intent} (confidence: {confidence:.2f})")

    if confidence < CONFIDENCE_THRESHOLD or margin < 0.15:
        print("Bot: I'm not fully sure. Did you mean admission, fees, courses, timetable, or contact?")
    else:
        print("Bot:", responses.get(intent, "Sorry, I do not understand your question."))