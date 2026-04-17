import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
            self.df["question"] + " " +
            self.df["context"] + " " +
            self.df["answer"]
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

        if "programme" in q or "diploma" in q or "degree" in q:
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

        scores = 0.6 * cosine_similarity(word_vec, self.word_matrix).flatten() + \
                 0.4 * cosine_similarity(char_vec, self.char_matrix).flatten()

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
            print(f"Could not read evaluation file: {e}")
            return

        required_cols = ["question", "expected_intent", "expected_answer"]
        missing = [c for c in required_cols if c not in test_df.columns]
        if missing:
            print("Evaluation CSV missing columns:", ", ".join(missing))
            return

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

        print("\nIntent report saved to tfidf_chatbot_evaluation_intent_report.csv")
        print("Detailed results saved to tfidf_chatbot_evaluation_detailed_results.csv")
        print("Full text report saved to tfidf_chatbot_evaluation_report.txt")

        print("\nIntent Classification Summary")
        print(report_df.to_string(index=False))
        print(f"\nMicro Precision : {summary['micro_precision']}")
        print(f"Micro Recall    : {summary['micro_recall']}")
        print(f"Micro Relevant  : {summary['micro_relevant']}")
        print(f"Macro Precision : {summary['macro_precision']}")
        print(f"Macro Recall    : {summary['macro_recall']}")
        print(f"Macro Relevant  : {summary['macro_relevant']}")

    # =============================
    # RUN
    # =============================
    def run(self):
        print("TARUMT TF-IDF Chatbot Running...")
        print("Type 'evaluate' to run testing.")
        print("Type 'exit' or 'quit' to stop.\n")

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye.")
                break

            if not user_input:
                print("Please enter a valid question.")
                continue

            if user_input.lower() == "evaluate":
                test_csv = input("Enter evaluation CSV file path: ").strip()
                self.evaluate_from_file(test_csv)
                continue

            result = self.get_response(user_input)
            print("Bot:", result["answer"])
            print("Predicted Intent:", result["predicted_intent"])
            print("Score:", round(result["score"], 4))


if __name__ == "__main__":
    bot = TFIDFRAGChatbot("tarumt_faq_dataset.csv")
    bot.run()