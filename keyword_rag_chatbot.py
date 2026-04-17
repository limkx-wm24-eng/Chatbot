import pandas as pd
import re
import string
from difflib import SequenceMatcher


print("Running Smart Keyword RAG Chatbot")


class SmartKeywordRAGChatbot:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None

        self.stop_words = {
            "what","is","are","the","a","an","do","does","did","can","could",
            "would","should","how","where","why","which","who","whom",
            "this","that","these","those","i","you","we","they","he","she",
            "it","my","your","our","their","to","for","of","in","on","at",
            "by","with","about","from","and","or","if","then","than","be",
            "been","being","am","was","were","will","shall","may","might",
            "me","tarumt","has","have"
        }

        self.topic_keywords = {
            "courses": {"course","courses","programme","programmes","diploma","degree"},
            "admission": {"admission","apply","application","intake","requirements"},
            "fees": {"fee","fees","tuition","payment","cost","price"},
            "accommodation": {"hostel","accommodation","room"},
            "scholarship": {"scholarship","loan","financial"},
            "location": {"location","campus","kuala","penang","perak","johor","pahang","sabah"},
            "facilities": {"facility","facilities","library","lab","wifi","gym","canteen"},
        }

        self.load_data()

    def clean_text(self, text):
        text = str(text).lower().strip()
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text)
        return text

    def similarity(self, a, b):
        return SequenceMatcher(None, a, b).ratio()

    def load_data(self):
        self.df = pd.read_csv(self.csv_file).fillna("")

        self.df["question_clean"] = self.df["question"].apply(self.clean_text)
        self.df["context_clean"] = self.df["context"].apply(self.clean_text)
        self.df["answer_clean"] = self.df["answer"].apply(self.clean_text)

        print("Dataset loaded successfully.")
        print(f"Total records: {len(self.df)}")

    def predict_intent(self, query):
        q = self.clean_text(query)

        if "intake" in q:
            return "intake_info"
        if "apply" in q or "application" in q:
            return "admission_apply"
        if "course" in q:
            return "course_list"
        if "programme" in q or "diploma" in q or "degree" in q:
            return "programme_list"
        if "fee" in q or "cost" in q:
            return "fees_detail"
        if "hostel" in q:
            return "hostel_info"
        if "scholarship" in q:
            return "scholarship_info"
        if "campus" in q or "location" in q:
            return "location_info"
        if "facility" in q:
            return "facilities_info"

        return "None"

    def get_response(self, query):
        q_clean = self.clean_text(query)
        predicted_intent = self.predict_intent(query)

        best_score = 0
        best_row = None

        for _, row in self.df.iterrows():
            score = self.similarity(q_clean, row["question_clean"])

            if score > best_score:
                best_score = score
                best_row = row

        if best_score < 0.3:
            answer = "Sorry, I could not find a relevant answer."
        else:
            answer = best_row["answer"]

        return {
            "answer": answer,
            "score": best_score,
            "predicted_intent": predicted_intent
        }

    # =============================
    # EVALUATION (ONLY P/R/RELEVANT)
    # =============================
    def safe_divide(self, a, b):
        return a / b if b != 0 else 0.0

    def classification_metrics(self, y_true, y_pred):
        labels = sorted(set(y_true) | set(y_pred))
        rows = []

        total_tp = total_fp = total_fn = 0

        for label in labels:
            tp = sum(1 for t,p in zip(y_true,y_pred) if t==label and p==label)
            fp = sum(1 for t,p in zip(y_true,y_pred) if t!=label and p==label)
            fn = sum(1 for t,p in zip(y_true,y_pred) if t==label and p!=label)

            precision = self.safe_divide(tp, tp+fp)
            recall = self.safe_divide(tp, tp+fn)
            relevant = self.safe_divide(2*precision*recall, precision+recall)

            rows.append({
                "intent": label,
                "precision": round(precision,4),
                "recall": round(recall,4),
                "relevant_score": round(relevant,4),
                "support": sum(1 for t in y_true if t==label)
            })

            total_tp += tp
            total_fp += fp
            total_fn += fn

        micro_precision = self.safe_divide(total_tp, total_tp+total_fp)
        micro_recall = self.safe_divide(total_tp, total_tp+total_fn)
        micro_relevant = self.safe_divide(
            2*micro_precision*micro_recall,
            micro_precision+micro_recall
        )

        macro_precision = sum(r["precision"] for r in rows)/len(rows)
        macro_recall = sum(r["recall"] for r in rows)/len(rows)
        macro_relevant = sum(r["relevant_score"] for r in rows)/len(rows)

        summary = {
            "micro_precision": round(micro_precision,4),
            "micro_recall": round(micro_recall,4),
            "micro_relevant": round(micro_relevant,4),
            "macro_precision": round(macro_precision,4),
            "macro_recall": round(macro_recall,4),
            "macro_relevant": round(macro_relevant,4)
        }

        return rows, summary

    def evaluate_from_file(self, test_csv):
        test_df = pd.read_csv(test_csv).fillna("")

        y_true = []
        y_pred = []

        for _, row in test_df.iterrows():
            result = self.get_response(row["question"])

            y_true.append(row["expected_intent"])
            y_pred.append(result["predicted_intent"])

        report_rows, summary = self.classification_metrics(y_true, y_pred)

        report_df = pd.DataFrame(report_rows)
        report_df.to_csv("keyword_chatbot_evaluation.csv", index=False)

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
        print("Smart Keyword Chatbot Running...")
        print("Type 'evaluate' to test\n")

        while True:
            user = input("You: ")

            if user == "exit":
                break

            if user == "evaluate":
                file = input("Enter test CSV: ")
                self.evaluate_from_file(file)
                continue

            res = self.get_response(user)
            print("Bot:", res["answer"])
            print("Intent:", res["predicted_intent"])
            print("Score:", round(res["score"],4))


if __name__ == "__main__":
    bot = SmartKeywordRAGChatbot("tarumt_faq_dataset.csv")
    bot.run()