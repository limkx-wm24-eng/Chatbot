import re
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import webbrowser


class TFIDFRAGChatbot:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None

        self.word_vectorizer = None
        self.char_vectorizer = None
        self.word_matrix = None
        self.char_matrix = None

        self.synonym_map = {
            "program": "programme",
            "programs": "programme",
            "programmes": "programme",
            "courses": "course",
            "fees": "fee",
            "cost": "fee",
            "costs": "fee",
            "price": "fee",
            "prices": "fee",
            "intakes": "intake",
            "branches": "campus",
            "branch": "campus",
            "campuses": "campus",
            "facilities": "facility",
            "apply": "apply",
            "application": "apply",
            "applications": "apply",
            "applying": "apply",
            "register": "apply",
            "registration": "apply",
            "enroll": "enroll",
            "enrol": "enroll",
            "enrollment": "enroll",
            "enrolment": "enroll",
            "make": "apply",
            "admissions": "admission",
            "minimal": "minimum",
            "requirement": "requirements",
            "qualification": "requirements",
            "qualifications": "requirements",
            "eligible": "eligibility",
            "enter": "entry",
            "joining": "join",
            "joined": "join"
        }

        self.load_data()

    # =============================
    # TEXT NORMALIZATION
    # =============================
    def normalize_text(self, text):
        text = str(text).lower().strip()

        phrase_map = {
            "minimum requirement": "entry requirements",
            "minimal requirement": "entry requirements",
            "requirements to apply": "entry requirements",
            "requirement to apply": "entry requirements",
            "requirements to entry": "entry requirements",
            "requirement to entry": "entry requirements",

            "where is the campus": "campus location",
            "where is campus": "campus location",

            "how to apply": "admission process",
            "how do i apply": "admission process",
            "how can i apply": "admission process",
            "where to apply": "admission process",
            "application process": "admission process",

            "how to make the application": "admission process",
            "how to make application": "admission process",
            "make application": "admission process",
            "make an application": "admission process",

            "when can i enroll": "intake",
            "when can i enrol": "intake",
            "what month can i enroll": "intake",
            "what month can i enrol": "intake",
            "month to enroll": "intake",
            "month to enrol": "intake",
            "what is the month that i can enroll": "intake",
            "what is the month that i can enrol": "intake",

            "register the course": "entry requirements",
            "requirement needed to register the course": "entry requirements",

            "what courses": "course",
            "what programme": "programme",
            "what programs": "programme"
        }

        for old, new in phrase_map.items():
            text = text.replace(old, new)

        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        words = text.split()
        words = [self.synonym_map.get(w, w) for w in words]

        return " ".join(words)

    # =============================
    # LOAD DATA
    # =============================
    def load_data(self):
        self.df = pd.read_csv(self.csv_file).fillna("")

        self.df["combined_text"] = (
            self.df["question"].astype(str) + " " +
            self.df["context"].astype(str) + " " +
            self.df["answer"].astype(str)
        )

        self.df["clean_text"] = self.df["combined_text"].apply(self.normalize_text)
        self.build_index()

    # =============================
    # BUILD TF-IDF
    # =============================
    def build_index(self):
        self.word_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.word_matrix = self.word_vectorizer.fit_transform(self.df["clean_text"])

        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.char_matrix = self.char_vectorizer.fit_transform(self.df["clean_text"])

    # =============================
    # INTENT DETECTION
    # =============================
    def predict_intent(self, query):
        q = self.normalize_text(query)

        if "intake" in q:
            return "intake_info"

        if "admission process" in q or ("apply" in q and "fee" not in q):
            return "admission_apply"

        if ("campus" in q and "location" in q) or ("where" in q and "campus" in q):
            return "location_info"

        if "course" in q and "requirements" not in q:
            return "course_list"

        if "programme" in q or "diploma" in q or "degree" in q or "postgraduate" in q:
            return "programme_list"

        if "entry" in q or "requirements" in q:
            return "requirements_info"

        if "scholarship" in q or "loan" in q or "financial aid" in q or "ptptn" in q:
            return "scholarship_info"

        if "hostel" in q or "accommodation" in q or "room" in q:
            return "hostel_info"

        if "fee" in q or "payment" in q or "cost" in q or "price" in q or "tuition" in q:
            return "fees_detail"

        if "library" in q or "wifi" in q or "facility" in q or "lab" in q or "canteen" in q or "sports" in q:
            return "facilities_info"

        if "document" in q or "certificate" in q or "transcript" in q:
            return "document_info"

        if "deadline" in q or "closing date" in q or "last date" in q:
            return "deadline_info"

        return "None"

    # =============================
    # RULE-BASED ANSWERS
    # =============================
    def rule_based_answer(self, query):
        q = self.normalize_text(query)

        if "intake" in q:
            return "TARUMT intakes are January, May/June, September and November."

        if "admission process" in q or ("apply" in q and "fee" not in q):
            return "You can apply through the TARUMT admission portal via the official website by selecting your programme and submitting the required documents."

        if ("campus" in q and "location" in q) or ("where" in q and "campus" in q):
            return "TARUMT has 6 campuses located in Kuala Lumpur, Penang, Perak, Johor, Pahang and Sabah."

        if "course" in q and "requirements" not in q:
            return "TARUMT offers courses in Engineering, Information Technology, Business and Accounting."

        if "programme" in q or "diploma" in q or "degree" in q or "postgraduate" in q:
            return "TARUMT offers Diploma, Degree and Postgraduate programmes."

        if "entry" in q or "requirements" in q:
            return "Entry requirements depend on the programme level and relevant academic qualifications such as SPM, UEC, STPM or equivalent."

        return None

    # =============================
    # TF-IDF FALLBACK
    # =============================
    def tfidf_fallback(self, user_query):
        query = self.normalize_text(user_query)

        word_vec = self.word_vectorizer.transform([query])
        char_vec = self.char_vectorizer.transform([query])

        scores = (
            0.6 * cosine_similarity(word_vec, self.word_matrix).flatten() +
            0.4 * cosine_similarity(char_vec, self.char_matrix).flatten()
        )

        best_idx = scores.argmax()
        best_score = scores[best_idx]

        if best_score < 0.15:
            return "Sorry, I could not find a relevant answer.", float(best_score)

        row = self.df.iloc[best_idx]
        return row["context"] + " " + row["answer"], float(best_score)

    # =============================
    # MAIN RESPONSE
    # =============================
    def get_response(self, user_query):
        predicted_intent = self.predict_intent(user_query)
        rule = self.rule_based_answer(user_query)

        if rule:
            return {
                "answer": rule,
                "score": 1.0,
                "predicted_intent": predicted_intent
            }

        answer, score = self.tfidf_fallback(user_query)
        return {
            "answer": answer,
            "score": score,
            "predicted_intent": predicted_intent
        }

    # =============================
    # EVALUATION
    # =============================
    def safe_divide(self, a, b):
        return a / b if b != 0 else 0.0

    def classification_metrics(self, y_true, y_pred):
        labels = sorted(set(y_true) | set(y_pred))
        rows = []

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for label in labels:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

            precision = self.safe_divide(tp, tp + fp)
            recall = self.safe_divide(tp, tp + fn)
            relevant = self.safe_divide(2 * precision * recall, precision + recall)

            rows.append({
                "intent": label,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "relevant_score": round(relevant, 4),
                "support": sum(1 for t in y_true if t == label)
            })

            total_tp += tp
            total_fp += fp
            total_fn += fn

        micro_precision = self.safe_divide(total_tp, total_tp + total_fp)
        micro_recall = self.safe_divide(total_tp, total_tp + total_fn)
        micro_relevant = self.safe_divide(
            2 * micro_precision * micro_recall,
            micro_precision + micro_recall
        )

        macro_precision = sum(r["precision"] for r in rows) / len(rows) if rows else 0.0
        macro_recall = sum(r["recall"] for r in rows) / len(rows) if rows else 0.0
        macro_relevant = sum(r["relevant_score"] for r in rows) / len(rows) if rows else 0.0

        summary = {
            "micro_precision": round(micro_precision, 4),
            "micro_recall": round(micro_recall, 4),
            "micro_relevant": round(micro_relevant, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_relevant": round(macro_relevant, 4)
        }

        return rows, summary

    def evaluate_from_file(self, test_csv):
        try:
            test_df = pd.read_csv(test_csv).fillna("")
        except Exception as e:
            return False, f"Could not read evaluation file: {e}"

        required_cols = ["question", "expected_intent", "expected_answer"]
        missing = [c for c in required_cols if c not in test_df.columns]
        if missing:
            return False, f"Evaluation CSV missing columns: {', '.join(missing)}"

        y_true = []
        y_pred = []
        detailed_rows = []

        for _, row in test_df.iterrows():
            question = str(row["question"]).strip()
            expected_intent = str(row["expected_intent"]).strip()

            result = self.get_response(question)
            predicted_intent = str(result.get("predicted_intent", "None"))

            y_true.append(expected_intent)
            y_pred.append(predicted_intent)

            detailed_rows.append({
                "question": question,
                "expected_intent": expected_intent,
                "predicted_intent": predicted_intent,
                "score": result.get("score", 0)
            })

        report_rows, summary = self.classification_metrics(y_true, y_pred)

        report_df = pd.DataFrame(report_rows)
        detailed_df = pd.DataFrame(detailed_rows)

        report_df.to_csv("tfidf_chatbot_evaluation_intent_report.csv", index=False)
        detailed_df.to_csv("tfidf_chatbot_evaluation_detailed_results.csv", index=False)

        with open("tfidf_chatbot_evaluation_report.txt", "w", encoding="utf-8") as f:
            f.write("TFIDF CHATBOT EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write("Intent Classification Metrics\n")
            f.write(str(report_df.to_string(index=False)))
            f.write("\n\n")
            f.write(f"Micro Precision : {summary['micro_precision']}\n")
            f.write(f"Micro Recall    : {summary['micro_recall']}\n")
            f.write(f"Micro Relevant  : {summary['micro_relevant']}\n")
            f.write(f"Macro Precision : {summary['macro_precision']}\n")
            f.write(f"Macro Recall    : {summary['macro_recall']}\n")
            f.write(f"Macro Relevant  : {summary['macro_relevant']}\n")

        result_text = []
        result_text.append("Intent Classification Summary\n")
        result_text.append(report_df.to_string(index=False))
        result_text.append("\n")
        result_text.append(f"Micro Precision : {summary['micro_precision']}")
        result_text.append(f"Micro Recall    : {summary['micro_recall']}")
        result_text.append(f"Micro Relevant  : {summary['micro_relevant']}")
        result_text.append(f"Macro Precision : {summary['macro_precision']}")
        result_text.append(f"Macro Recall    : {summary['macro_recall']}")
        result_text.append(f"Macro Relevant  : {summary['macro_relevant']}")

        return True, "\n".join(result_text)

    # =============================
    # OPTIONAL EXTERNAL API EXAMPLE
    # =============================
    def get_joke_from_api(self):
        """
        Example of external API integration.
        This uses a public joke API just to show platform/API extension.
        """
        try:
            response = requests.get("https://official-joke-api.appspot.com/random_joke", timeout=5)
            if response.status_code == 200:
                data = response.json()
                setup = data.get("setup", "")
                punchline = data.get("punchline", "")
                return f"{setup}\n{punchline}"
            return "Could not get data from external API."
        except Exception as e:
            return f"API error: {e}"


class ChatbotGUI:
    def __init__(self, root, chatbot):
        self.root = root
        self.chatbot = chatbot

        self.root.title("TARUMT Chatbot GUI")
        self.root.geometry("900x650")
        self.root.configure(bg="#f4f6f8")

        self.build_gui()

    def build_gui(self):
        title = tk.Label(
            self.root,
            text="TARUMT TF-IDF Chatbot",
            font=("Arial", 20, "bold"),
            bg="#f4f6f8",
            fg="#1f3b73"
        )
        title.pack(pady=10)

        subtitle = tk.Label(
            self.root,
            text="GUI + Intent Detection + TF-IDF Retrieval + API Integration",
            font=("Arial", 11),
            bg="#f4f6f8",
            fg="#555555"
        )
        subtitle.pack()

        # Chat display
        self.chat_area = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            width=100,
            height=25,
            font=("Arial", 11),
            bg="white",
            fg="black",
            state="disabled"
        )
        self.chat_area.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)

        # Quick buttons
        quick_frame = tk.Frame(self.root, bg="#f4f6f8")
        quick_frame.pack(pady=5)

        quick_questions = [
            ("Intake", "What are the intake months?"),
            ("Course", "What courses are offered?"),
            ("Programme", "What programmes are offered?"),
            ("Campus", "Where are the campuses located?"),
            ("Apply", "How do I apply?"),
            ("Requirements", "What are the entry requirements?")
        ]

        for text, question in quick_questions:
            btn = tk.Button(
                quick_frame,
                text=text,
                width=14,
                command=lambda q=question: self.quick_ask(q),
                bg="#dbe9ff"
            )
            btn.pack(side=tk.LEFT, padx=4)

        # Input frame
        input_frame = tk.Frame(self.root, bg="#f4f6f8")
        input_frame.pack(fill=tk.X, padx=15, pady=10)

        self.user_input = tk.Entry(input_frame, font=("Arial", 12))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.user_input.bind("<Return>", lambda event: self.send_message())

        send_btn = tk.Button(
            input_frame,
            text="Send",
            font=("Arial", 11, "bold"),
            bg="#4caf50",
            fg="white",
            width=10,
            command=self.send_message
        )
        send_btn.pack(side=tk.LEFT, padx=5)

        clear_btn = tk.Button(
            input_frame,
            text="Clear",
            font=("Arial", 11, "bold"),
            bg="#f44336",
            fg="white",
            width=10,
            command=self.clear_chat
        )
        clear_btn.pack(side=tk.LEFT, padx=5)

        # Bottom buttons
        bottom_frame = tk.Frame(self.root, bg="#f4f6f8")
        bottom_frame.pack(pady=5)

        eval_btn = tk.Button(
            bottom_frame,
            text="Run Evaluation",
            width=18,
            command=self.run_evaluation,
            bg="#ffcc80"
        )
        eval_btn.pack(side=tk.LEFT, padx=5)

        api_btn = tk.Button(
            bottom_frame,
            text="Test External API",
            width=18,
            command=self.call_external_api,
            bg="#b2dfdb"
        )
        api_btn.pack(side=tk.LEFT, padx=5)

        website_btn = tk.Button(
            bottom_frame,
            text="Open TARUMT Website",
            width=18,
            command=self.open_website,
            bg="#c5cae9"
        )
        website_btn.pack(side=tk.LEFT, padx=5)

        self.append_chat("Bot", "Hello. Welcome to the TARUMT Chatbot GUI.\nAsk me about intake, courses, programmes, campus, admission, or requirements.")

    def append_chat(self, sender, message):
        self.chat_area.config(state="normal")
        self.chat_area.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_area.config(state="disabled")
        self.chat_area.see(tk.END)

    def send_message(self):
        question = self.user_input.get().strip()
        if not question:
            messagebox.showwarning("Warning", "Please enter a question.")
            return

        self.append_chat("You", question)

        result = self.chatbot.get_response(question)
        answer = result["answer"]
        intent = result["predicted_intent"]
        score = round(result["score"], 4)

        full_reply = f"{answer}\n\nPredicted Intent: {intent}\nScore: {score}"
        self.append_chat("Bot", full_reply)

        self.user_input.delete(0, tk.END)

    def quick_ask(self, question):
        self.user_input.delete(0, tk.END)
        self.user_input.insert(0, question)
        self.send_message()

    def clear_chat(self):
        self.chat_area.config(state="normal")
        self.chat_area.delete(1.0, tk.END)
        self.chat_area.config(state="disabled")
        self.append_chat("Bot", "Chat cleared. Ask me a new question.")

    def run_evaluation(self):
        file_path = filedialog.askopenfilename(
            title="Select Evaluation CSV",
            filetypes=[("CSV files", "*.csv")]
        )

        if not file_path:
            return

        success, result_text = self.chatbot.evaluate_from_file(file_path)

        if success:
            messagebox.showinfo("Evaluation Completed", "Evaluation files have been saved successfully.")
            self.append_chat("System", result_text)
        else:
            messagebox.showerror("Evaluation Error", result_text)

    def call_external_api(self):
        api_result = self.chatbot.get_joke_from_api()
        self.append_chat("External API", api_result)

    def open_website(self):
        webbrowser.open("https://www.tarc.edu.my/")

def main():
    try:
        chatbot = TFIDFRAGChatbot("tarumt_faq_dataset.csv")
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Startup Error", f"Failed to load chatbot data:\n{e}")
        return

    root = tk.Tk()
    app = ChatbotGUI(root, chatbot)
    root.mainloop()


if __name__ == "__main__":
    main()