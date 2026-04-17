import pandas as pd
import os
import re
import string
from difflib import SequenceMatcher


print("Running Smart Keyword RAG Chatbot")


class SmartKeywordRAGChatbot:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None

        self.stop_words = {
            "what", "is", "are", "the", "a", "an", "do", "does", "did", "can", "could",
            "would", "should", "how", "where", "why", "which", "who", "whom",
            "this", "that", "these", "those", "i", "you", "we", "they", "he", "she",
            "it", "my", "your", "our", "their", "to", "for", "of", "in", "on", "at",
            "by", "with", "about", "from", "and", "or", "if", "then", "than", "be",
            "been", "being", "am", "was", "were", "will", "shall", "may", "might",
            "me", "tarumt", "has", "have"
        }

        self.topic_keywords = {
            "courses": {"course", "courses", "programme", "programmes", "program", "programs", "diploma", "degree"},
            "admission": {"admission", "apply", "application", "requirements", "entry", "register"},
            "fees": {"fee", "fees", "tuition", "payment", "cost", "price"},
            "accommodation": {"hostel", "accommodation", "room", "stay", "dorm"},
            "scholarship": {"scholarship", "loan", "financial", "ptptn"},
            "location": {"location", "campus", "branch", "kuala", "penang", "perak", "johor", "pahang", "sabah"},
            "facilities": {"facility", "facilities", "library", "lab", "wifi", "gym", "canteen", "sports"},
            "documents": {"document", "documents", "certificate", "transcript"},
            "intake": {"intake", "january", "may", "june", "september", "november"}
        }

        self.load_data()

    def clean_text(self, text):
        text = str(text).lower().strip()
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text)
        return text

    def preprocess_text(self, text):
        text = self.clean_text(text)
        words = [w for w in text.split() if w not in self.stop_words]
        return " ".join(words)

    def similarity(self, a, b):
        return SequenceMatcher(None, a, b).ratio()

    def keyword_overlap_score(self, query, text):
        q_words = set(query.split())
        t_words = set(text.split())

        if not q_words or not t_words:
            return 0.0

        overlap = q_words.intersection(t_words)
        return len(overlap) / max(len(q_words), 1)

    def detect_query_type(self, clean_query):
        words = clean_query.split()

        if not words:
            return "empty"

        # Simple / keyword-heavy query
        if len(words) <= 4:
            return "keyword"

        matched_topic_count = 0
        for topic_words in self.topic_keywords.values():
            if any(w in topic_words for w in words):
                matched_topic_count += 1

        if matched_topic_count >= 1 and len(words) <= 6:
            return "keyword"

        return "hybrid"

    def choose_retrieval_method(self, clean_query):
        query_type = self.detect_query_type(clean_query)

        if query_type == "keyword":
            return "keyword_matching"
        elif query_type == "hybrid":
            return "hybrid_similarity"
        return "none"

    def load_data(self):
        self.df = pd.read_csv(self.csv_file).fillna("")

        required_columns = ["question", "context", "answer"]
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        self.df["question"] = self.df["question"].astype(str)
        self.df["context"] = self.df["context"].astype(str)
        self.df["answer"] = self.df["answer"].astype(str)

        self.df["question_clean"] = self.df["question"].apply(self.preprocess_text)
        self.df["context_clean"] = self.df["context"].apply(self.preprocess_text)
        self.df["answer_clean"] = self.df["answer"].apply(self.preprocess_text)

        print("Dataset loaded successfully.")
        print(f"Total records: {len(self.df)}")

    def predict_intent(self, query):
        q = self.clean_text(query)

        if "intake" in q:
            return "intake_info"
        if "apply" in q or "application" in q or "admission" in q:
            return "admission_apply"
        if "requirement" in q or "entry" in q or "qualification" in q:
            return "requirements_info"
        if "document" in q or "certificate" in q or "transcript" in q:
            return "document_info"
        if "course" in q:
            return "course_list"
        if "programme" in q or "program" in q or "diploma" in q or "degree" in q:
            return "programme_list"
        if "fee" in q or "cost" in q or "payment" in q or "tuition" in q:
            return "fees_detail"
        if "hostel" in q or "accommodation" in q or "room" in q:
            return "hostel_info"
        if "scholarship" in q or "loan" in q or "financial" in q or "ptptn" in q:
            return "scholarship_info"
        if "campus" in q or "location" in q or "branch" in q:
            return "location_info"
        if "facility" in q or "facilities" in q or "library" in q or "lab" in q or "wifi" in q or "gym" in q or "canteen" in q:
            return "facilities_info"

        return "None"

    def retrieve_by_keyword(self, clean_query):
        best_score = 0.0
        best_row = None
        candidates = []

        for _, row in self.df.iterrows():
            q_score = self.keyword_overlap_score(clean_query, row["question_clean"])
            c_score = self.keyword_overlap_score(clean_query, row["context_clean"])
            a_score = self.keyword_overlap_score(clean_query, row["answer_clean"])

            final_score = max(q_score, c_score, a_score)
            candidates.append((row, final_score, q_score, c_score, a_score))

            if final_score > best_score:
                best_score = final_score
                best_row = row

        candidates.sort(key=lambda x: x[1], reverse=True)
        return best_row, best_score, candidates[:5]

    def retrieve_by_hybrid(self, clean_query):
        best_score = 0.0
        best_row = None
        candidates = []

        for _, row in self.df.iterrows():
            sim_q = self.similarity(clean_query, row["question_clean"])
            sim_c = self.similarity(clean_query, row["context_clean"])
            keyword_q = self.keyword_overlap_score(clean_query, row["question_clean"])
            keyword_c = self.keyword_overlap_score(clean_query, row["context_clean"])

            final_score = (
                0.45 * sim_q +
                0.20 * sim_c +
                0.25 * keyword_q +
                0.10 * keyword_c
            )

            candidates.append((row, final_score, sim_q, sim_c, keyword_q, keyword_c))

            if final_score > best_score:
                best_score = final_score
                best_row = row

        candidates.sort(key=lambda x: x[1], reverse=True)
        return best_row, best_score, candidates[:5]

    def get_response(self, query):
        clean_query = self.preprocess_text(query)
        predicted_intent = self.predict_intent(query)
        retrieval_method = self.choose_retrieval_method(clean_query)

        if not clean_query:
            return {
                "retrieved_question": None,
                "retrieved_context": None,
                "answer": "Please enter a valid TAR UMT-related question.",
                "score": 0.0,
                "confidence": "Low",
                "predicted_intent": "None",
                "retrieval_method": "none",
                "top_matches": []
            }

        if retrieval_method == "keyword_matching":
            best_row, best_score, top_candidates = self.retrieve_by_keyword(clean_query)
        else:
            best_row, best_score, top_candidates = self.retrieve_by_hybrid(clean_query)

        if best_row is None or best_score < 0.30:
            return {
                "retrieved_question": None,
                "retrieved_context": None,
                "answer": "Sorry, I could not find a relevant answer.",
                "score": float(best_score if best_row is not None else 0.0),
                "confidence": "Low",
                "predicted_intent": predicted_intent,
                "retrieval_method": retrieval_method,
                "top_matches": []
            }

        if best_score >= 0.75:
            confidence = "High"
        elif best_score >= 0.50:
            confidence = "Medium"
        else:
            confidence = "Low"

        top_matches = []
        used_questions = set()

        for item in top_candidates:
            row = item[0]
            score = item[1]
            q = row["question"].strip()

            if q in used_questions:
                continue
            used_questions.add(q)

            top_matches.append({
                "question": q,
                "score": round(float(score), 4)
            })

        return {
            "retrieved_question": best_row["question"],
            "retrieved_context": best_row["context"],
            "answer": best_row["answer"],
            "score": float(best_score),
            "confidence": confidence,
            "predicted_intent": predicted_intent,
            "retrieval_method": retrieval_method,
            "top_matches": top_matches
        }

    def display_result(self, result):
        print("\n" + "=" * 80)
        print("Chatbot Type       : Keyword-Based RAG")
        print(f"Retrieval Method   : {result.get('retrieval_method', 'None')}")
        print(f"Similarity Score   : {result['score']:.4f}")
        print(f"Predicted Intent   : {result.get('predicted_intent', 'None')}")
        print(f"Confidence Level   : {result.get('confidence', 'Low')}")
        print("-" * 80)
        print(f"Retrieved Question : {result['retrieved_question'] if result['retrieved_question'] else 'None'}")
        print("-" * 80)
        print("Retrieved Context  :")
        print(result["retrieved_context"] if result["retrieved_context"] else "None")
        print("-" * 80)
        print("Bot Answer         :")
        print(result["answer"])
        print("-" * 80)

        if result.get("top_matches"):
            print("Top Matches:")
            for i, item in enumerate(result["top_matches"], start=1):
                print(f"{i}. {item['question']} (score={item['score']})")

        print("=" * 80 + "\n")

    # =============================
    # EVALUATION
    # =============================
    def safe_divide(self, a, b):
        return a / b if b != 0 else 0.0

    def classification_metrics(self, y_true, y_pred):
        labels = sorted(set(y_true) | set(y_pred))
        rows = []

        total_tp = total_fp = total_fn = 0

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
        import os

        base_dir = os.path.dirname(os.path.abspath(__file__))

        # 1. Use direct path if user entered full/relative valid path
        possible_paths = [
            test_csv,
            os.path.join(base_dir, test_csv),
            os.path.join(base_dir, "..", test_csv),
            os.path.join(base_dir, "..", "dataset", test_csv)
        ]

        resolved_path = None
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                resolved_path = abs_path
                break

        if resolved_path is None:
            print(f"Evaluation file not found: {test_csv}")
            print("Tried these locations:")
            for path in possible_paths:
                print(" -", os.path.abspath(path))
            return

        print("Using evaluation file:", resolved_path)

        test_df = pd.read_csv(resolved_path).fillna("")

        required_cols = ["question", "expected_intent"]
        missing = [c for c in required_cols if c not in test_df.columns]
        if missing:
            print("Evaluation CSV missing columns:", ", ".join(missing))
            return

        y_true = []
        y_pred = []

        for _, row in test_df.iterrows():
            result = self.get_response(row["question"])
            y_true.append(str(row["expected_intent"]).strip())
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
        print("Enhanced TAR UMT Keyword RAG Chatbot")
        print("Type 'evaluate' to test")
        print("Type 'exit' or 'quit' to stop\n")

        while True:
            user = input("You: ").strip()

            if user.lower() in ["exit", "quit"]:
                print("Goodbye.")
                break

            if not user:
                print("Please enter a valid question.")
                continue

            if user.lower() == "evaluate":
                file = input("Enter test CSV: ").strip()
                self.evaluate_from_file(file)
                continue

            res = self.get_response(user)
            self.display_result(res)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # go up one level → then into dataset folder
    csv_path = os.path.join(BASE_DIR, "..", "dataset", "tarumt_faq_dataset.csv")
    csv_path = os.path.abspath(csv_path)

    print("Using dataset:", csv_path)

    bot = SmartKeywordRAGChatbot(csv_path)
    bot.run()