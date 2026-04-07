import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pandas as pd
import re
import string
import torch

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

# =========================
# Shared cleaning
# =========================
def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


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
# Rule-based patterns
# =========================
rules = {
    "admission": [
        r"\bapply\b", r"\badmission\b", r"\benroll\b", r"\bentry\b",
        r"\bregister\b", r"\bnew student\b"
    ],
    "fees": [
        r"\bfee\b", r"\bfees\b", r"\btuition\b", r"\bpayment\b",
        r"\bbalance\b", r"\binstallment\b", r"\btransfer\b"
    ],
    "courses": [
        r"\bcourse\b", r"\bcourses\b", r"\bprogramme\b", r"\bprogram\b",
        r"\bdiploma\b", r"\bdegree\b", r"\bcomputer science\b",
        r"\binformation technology\b", r"\bbusiness\b"
    ],
    "timetable": [
        r"\btimetable\b", r"\bschedule\b", r"\bclass\b", r"\blecture\b"
    ],
    "contact": [
        r"\bcontact\b", r"\bphone\b", r"\bemail\b", r"\bhotline\b",
        r"\benquiries\b", r"\bstudent services\b"
    ],
    "greeting": [
        r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bgood morning\b",
        r"\bgood afternoon\b", r"\bgood evening\b"
    ],
    "thanks": [
        r"\bthank\b", r"\bthanks\b", r"\bappreciate\b"
    ],
    "goodbye": [
        r"\bbye\b", r"\bgoodbye\b", r"\bsee you\b", r"\bcatch you later\b"
    ]
}


# =========================
# Rule-based predictor
# =========================
def predict_rule_based(user_input: str):
    text = clean_text(user_input)
    scores = {}

    for intent, patterns in rules.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, text):
                score += 1
        if score > 0:
            scores[intent] = score

    if not scores:
        return None, None

    intent = max(scores, key=scores.get)
    return intent, None


# =========================
# Load dataset
# =========================
def load_dataset():
    df = pd.read_csv("faq_dataset.csv")
    df = df.dropna(subset=["text", "intent"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["intent"] != "")]
    df["text"] = df["text"].apply(clean_text)
    return df


# =========================
# Classical ML trainer
# =========================
def train_ml_model(model_type="SVM"):
    df = load_dataset()
    X = df["text"]
    y = df["intent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    if model_type == "Naive Bayes":
        clf = MultinomialNB()
    elif model_type == "Logistic Regression":
        clf = LogisticRegression(max_iter=2000, random_state=42)
    else:
        clf = LinearSVC()

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("clf", clf)
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


# =========================
# Transformer trainer
# =========================
def train_transformer_model():
    df = load_dataset()

    label_list = sorted(df["intent"].unique())
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    df["label"] = df["intent"].map(label2id)

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

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(batch):
        return tokenizer(batch["text"], truncation=True, padding=True)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="./transformer_results",
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    return model, tokenizer, id2label


# =========================
# GUI Application
# =========================
class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("University FAQ Chatbot")
        self.root.geometry("1000x700")
        self.root.minsize(850, 600)
        self.root.configure(bg="#e9eef5")

        self.current_model_name = tk.StringVar(value="Rule-Based")
        self.ml_model = None
        self.transformer_model = None
        self.transformer_tokenizer = None
        self.transformer_id2label = None

        self.setup_styles()
        self.setup_root_grid()
        self.build_widgets()

    def setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "TCombobox",
            fieldbackground="white",
            background="white",
            foreground="black"
        )

    def setup_root_grid(self):
        self.root.grid_rowconfigure(0, weight=0)  # header
        self.root.grid_rowconfigure(1, weight=0)  # controls
        self.root.grid_rowconfigure(2, weight=1)  # chat
        self.root.grid_rowconfigure(3, weight=0)  # input
        self.root.grid_rowconfigure(4, weight=0)  # status
        self.root.grid_columnconfigure(0, weight=1)

    def build_widgets(self):
        header = tk.Frame(self.root, bg="#1f4e79", height=90)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)

        title = tk.Label(
            header,
            text="University FAQ Chatbot System",
            font=("Segoe UI", 22, "bold"),
            fg="white",
            bg="#1f4e79"
        )
        title.place(relx=0.5, rely=0.5, anchor="center")

        top_panel = tk.Frame(self.root, bg="#e9eef5", height=70)
        top_panel.grid(row=1, column=0, sticky="ew", padx=15, pady=10)
        top_panel.grid_propagate(False)
        top_panel.grid_columnconfigure(10, weight=1)

        tk.Label(
            top_panel,
            text="Select Model:",
            font=("Segoe UI", 11, "bold"),
            bg="#e9eef5"
        ).grid(row=0, column=0, padx=(0, 8), pady=10, sticky="w")

        self.model_box = ttk.Combobox(
            top_panel,
            textvariable=self.current_model_name,
            values=["Rule-Based", "Naive Bayes", "Logistic Regression", "SVM", "Transformer"],
            state="readonly",
            width=22,
            font=("Segoe UI", 10)
        )
        self.model_box.grid(row=0, column=1, padx=(0, 12), pady=10, sticky="w")

        load_btn = tk.Button(
            top_panel,
            text="Load Model",
            command=self.load_model,
            font=("Segoe UI", 10, "bold"),
            bg="#2e86de",
            fg="white",
            activebackground="#2165a8",
            relief="flat",
            padx=14,
            pady=8
        )
        load_btn.grid(row=0, column=2, padx=(0, 12), pady=10, sticky="w")

        clear_btn = tk.Button(
            top_panel,
            text="Clear Chat",
            command=self.clear_chat,
            font=("Segoe UI", 10, "bold"),
            bg="#6c757d",
            fg="white",
            activebackground="#4f5962",
            relief="flat",
            padx=14,
            pady=8
        )
        clear_btn.grid(row=0, column=3, pady=10, sticky="w")

        chat_container = tk.Frame(self.root, bg="#e9eef5")
        chat_container.grid(row=2, column=0, sticky="nsew", padx=15, pady=(0, 10))
        chat_container.grid_rowconfigure(0, weight=1)
        chat_container.grid_columnconfigure(0, weight=1)

        self.chat_area = scrolledtext.ScrolledText(
            chat_container,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            bg="white",
            fg="#222222",
            relief="solid",
            bd=1,
            padx=10,
            pady=10
        )
        self.chat_area.grid(row=0, column=0, sticky="nsew")
        self.chat_area.config(state="disabled")

        bottom_frame = tk.Frame(self.root, bg="#e9eef5", height=80)
        bottom_frame.grid(row=3, column=0, sticky="ew", padx=15, pady=(0, 10))
        bottom_frame.grid_propagate(False)
        bottom_frame.grid_columnconfigure(0, weight=1)

        self.entry = tk.Entry(
            bottom_frame,
            font=("Segoe UI", 12),
            relief="solid",
            bd=1
        )
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 10), pady=15, ipady=10)
        self.entry.bind("<Return>", lambda event: self.send_message())

        send_btn = tk.Button(
            bottom_frame,
            text="Send",
            command=self.send_message,
            font=("Segoe UI", 11, "bold"),
            bg="#28a745",
            fg="white",
            activebackground="#1f7d34",
            relief="flat",
            padx=22,
            pady=10
        )
        send_btn.grid(row=0, column=1, pady=15, sticky="e")

        self.status_label = tk.Label(
            self.root,
            text="Status: Ready",
            anchor="w",
            font=("Segoe UI", 10),
            bg="#dfe6ee",
            fg="#333333",
            padx=10,
            pady=6
        )
        self.status_label.grid(row=4, column=0, sticky="ew")

        self.append_chat("Bot", "Welcome. Please select a model and click 'Load Model'.")

    def set_status(self, text):
        self.status_label.config(text=f"Status: {text}")
        self.root.update_idletasks()

    def append_chat(self, sender, message):
        self.chat_area.config(state="normal")

        self.chat_area.tag_config("user_tag", foreground="#0b5394", font=("Segoe UI", 11, "bold"))
        self.chat_area.tag_config("user_msg", foreground="#222222", font=("Segoe UI", 11))
        self.chat_area.tag_config("bot_tag", foreground="#38761d", font=("Segoe UI", 11, "bold"))
        self.chat_area.tag_config("bot_msg", foreground="#222222", font=("Segoe UI", 11))
        self.chat_area.tag_config("sys_tag", foreground="#7f6000", font=("Segoe UI", 11, "bold"))
        self.chat_area.tag_config("sys_msg", foreground="#555555", font=("Segoe UI", 11, "italic"))

        if sender == "You":
            self.chat_area.insert(tk.END, "You:\n", "user_tag")
            self.chat_area.insert(tk.END, f"{message}\n\n", "user_msg")
        elif sender == "Bot":
            self.chat_area.insert(tk.END, "Bot:\n", "bot_tag")
            self.chat_area.insert(tk.END, f"{message}\n\n", "bot_msg")
        else:
            self.chat_area.insert(tk.END, f"{sender}:\n", "sys_tag")
            self.chat_area.insert(tk.END, f"{message}\n\n", "sys_msg")

        self.chat_area.config(state="disabled")
        self.chat_area.see(tk.END)

    def clear_chat(self):
        self.chat_area.config(state="normal")
        self.chat_area.delete("1.0", tk.END)
        self.chat_area.config(state="disabled")
        self.append_chat("Bot", "Chat cleared. You can continue asking questions.")

    def load_model(self):
        selected = self.current_model_name.get()
        self.set_status(f"Loading {selected} model...")
        self.append_chat("System", f"Loading {selected} model...")

        try:
            if selected == "Rule-Based":
                self.ml_model = None
                self.transformer_model = None
                self.transformer_tokenizer = None
                self.transformer_id2label = None

            elif selected in ["Naive Bayes", "Logistic Regression", "SVM"]:
                self.ml_model = train_ml_model(selected)
                self.transformer_model = None
                self.transformer_tokenizer = None
                self.transformer_id2label = None

            elif selected == "Transformer":
                model, tokenizer, id2label = train_transformer_model()
                self.transformer_model = model
                self.transformer_tokenizer = tokenizer
                self.transformer_id2label = id2label
                self.ml_model = None

            self.append_chat("System", f"{selected} model loaded successfully.")
            self.set_status(f"{selected} model loaded")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.append_chat("System", f"Failed to load model: {e}")
            self.set_status("Error while loading model")

    def get_response(self, user_input):
        selected = self.current_model_name.get()

        if selected == "Rule-Based":
            intent, confidence = predict_rule_based(user_input)

        elif selected in ["Naive Bayes", "Logistic Regression", "SVM"]:
            if self.ml_model is None:
                return None, None, "Please load the selected model first."
            cleaned = clean_text(user_input)
            intent = self.ml_model.predict([cleaned])[0]
            confidence = None

        elif selected == "Transformer":
            if self.transformer_model is None:
                return None, None, "Please load the transformer model first."

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.transformer_model.to(device)
            self.transformer_model.eval()

            cleaned = clean_text(user_input)
            inputs = self.transformer_tokenizer(
                cleaned,
                return_tensors="pt",
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.transformer_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                confidence = float(torch.max(probs).item())
                pred_id = int(torch.argmax(probs, dim=1).item())
                intent = self.transformer_id2label[pred_id]
        else:
            return None, None, "Unknown model selected."

        if intent is None:
            return None, confidence, "Sorry, I do not understand your question."

        if confidence is not None and confidence < 0.5:
            return intent, confidence, "I'm not confident. Can you rephrase your question?"

        return intent, confidence, responses.get(intent, "Sorry, I do not understand your question.")

    def send_message(self):
        user_input = self.entry.get().strip()
        if not user_input:
            return

        self.append_chat("You", user_input)
        self.entry.delete(0, tk.END)

        if user_input.lower() in ["exit", "quit", "goodbye"]:
            self.append_chat("Bot", "Goodbye. Have a nice day.")
            self.set_status("Conversation ended")
            return

        self.set_status("Generating response...")
        intent, confidence, bot_reply = self.get_response(user_input)

        if intent is None:
            self.append_chat("Bot", bot_reply)
        else:
            if confidence is None:
                self.append_chat("Bot", f"Predicted intent: {intent}\n{bot_reply}")
            else:
                self.append_chat("Bot", f"Predicted intent: {intent} (confidence: {confidence:.2f})\n{bot_reply}")

        self.set_status("Ready")


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()