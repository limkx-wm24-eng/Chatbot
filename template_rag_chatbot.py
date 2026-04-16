import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TemplateRAGChatbot:
    def __init__(self, csv_file, similarity_threshold=0.18):
        self.csv_file = csv_file
        self.similarity_threshold = similarity_threshold
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None

        self.allowed_keywords = {
            "tarumt", "tarc", "admission", "admissions", "apply", "application",
            "document", "documents", "offer", "offers", "admit", "admitted",
            "registration", "register", "enrol", "enrollment", "enrolment",
            "fee", "fees", "payment", "payments", "scholarship", "scholarships",
            "financial", "aid", "loan", "ptptn", "hostel", "accommodation",
            "campus", "programme", "program", "course", "courses", "intake",
            "student", "students", "foundation", "diploma", "degree",
            "bachelor", "master", "phd", "faculty", "faculties", "merit",
            "result", "results", "transcript", "certificate", "orientation",
            "freshmen", "freshman", "new", "international", "local", "study"
        }

        self.load_data()

    def preprocess_text(self, text):
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        custom_stopwords = {
            "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
            "is", "are", "am", "was", "were", "be", "been", "being",
            "do", "does", "did", "can", "could", "should", "would", "will",
            "may", "might", "shall", "must",
            "i", "me", "my", "mine", "we", "our", "ours",
            "you", "your", "yours",
            "please", "tell", "explain", "about", "know", "want", "need",
            "there", "any", "some", "the", "a", "an", "in"
        }

        words = [word for word in text.split() if word not in custom_stopwords]
        return " ".join(words)

    def is_in_scope(self, clean_query):
        query_words = set(clean_query.split())
        return len(query_words.intersection(self.allowed_keywords)) > 0

    def load_data(self):
        try:
            self.df = pd.read_csv(self.csv_file)

            required_columns = ["question", "context", "answer"]
            for col in required_columns:
                if col not in self.df.columns:
                    raise ValueError(f"Missing required column: {col}")

            self.df = self.df.fillna("")

            self.df["clean_question"] = self.df["question"].astype(str).apply(self.preprocess_text)
            self.df["clean_context"] = self.df["context"].astype(str).apply(self.preprocess_text)

            self.df["combined_text"] = (
                self.df["clean_question"] + " " +
                self.df["clean_question"] + " " +
                self.df["clean_question"] + " " +
                self.df["clean_context"]
            )

            self.vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True
            )

            self.tfidf_matrix = self.vectorizer.fit_transform(self.df["combined_text"])

            print("Dataset loaded successfully.")
            print(f"Total records: {len(self.df)}")

        except FileNotFoundError:
            print(f"Error: File '{self.csv_file}' not found.")
            raise
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def fallback_scope_message(self):
        return (
            "Sorry, I can only answer TAR UMT FAQ-related questions such as "
            "admission, application, required documents, fees, financial aid, "
            "scholarships, hostel, registration, programmes, and intake."
        )

    def fallback_no_match_message(self):
        return (
            "Sorry, I could not find a confident answer in the TAR UMT FAQ dataset. "
            "Please try asking in a more specific way."
        )

    def get_response(self, user_query):
        clean_query = self.preprocess_text(user_query)

        if not clean_query:
            return {
                "retrieved_question": None,
                "retrieved_context": None,
                "answer": "Please enter a valid TAR UMT-related question.",
                "score": 0.0
            }

        if not self.is_in_scope(clean_query):
            return {
                "retrieved_question": None,
                "retrieved_context": None,
                "answer": self.fallback_scope_message(),
                "score": 0.0
            }

        user_vector = self.vectorizer.transform([clean_query])
        similarities = cosine_similarity(user_vector, self.tfidf_matrix).flatten()

        best_index = similarities.argmax()
        best_score = similarities[best_index]

        if best_score < self.similarity_threshold:
            return {
                "retrieved_question": None,
                "retrieved_context": None,
                "answer": self.fallback_no_match_message(),
                "score": float(best_score)
            }

        best_row = self.df.iloc[best_index]

        return {
            "retrieved_question": best_row["question"],
            "retrieved_context": best_row["context"],
            "answer": best_row["answer"].strip(),
            "score": float(best_score)
        }

    def display_result(self, result):
        print("\n" + "=" * 70)
        print("Chatbot Type       : Template-Based RAG")
        print(f"Similarity Score   : {result['score']:.4f}")
        print("-" * 70)
        print(f"Retrieved Question : {result['retrieved_question'] if result['retrieved_question'] else 'None'}")
        print("-" * 70)
        print("Retrieved Context  :")
        print(result["retrieved_context"] if result["retrieved_context"] else "None")
        print("-" * 70)
        print("Bot Answer         :")
        print(result["answer"])
        print("=" * 70 + "\n")

    def run(self):
        while True:
            user_query = input("You: ").strip()

            if user_query.lower() in ["exit", "quit"]:
                print("Goodbye.")
                break

            if not user_query:
                print("Please enter a valid question.")
                continue

            result = self.get_response(user_query)
            self.display_result(result)


if __name__ == "__main__":
    csv_path = "tarumt_faq_dataset.csv"
    chatbot = TemplateRAGChatbot(
        csv_file=csv_path,
        similarity_threshold=0.18
    )
    chatbot.run()