import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TemplateRAGChatbot:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.load_data()

    def load_data(self):
        try:
            self.df = pd.read_csv(self.csv_file)
            required_columns = ["question", "context", "answer"]
            for col in required_columns:
                if col not in self.df.columns:
                    raise ValueError(f"Missing required column: {col}")

            self.df = self.df.fillna("")
            self.df["combined_text"] = self.df["question"].astype(str) + " " + self.df["context"].astype(str)

            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df["combined_text"])

            print("Dataset loaded successfully.")
            print(f"Total records: {len(self.df)}")

        except FileNotFoundError:
            print(f"Error: File '{self.csv_file}' not found.")
            raise
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def get_response(self, user_query):
        user_vector = self.vectorizer.transform([user_query])
        similarities = cosine_similarity(user_vector, self.tfidf_matrix).flatten()

        best_index = similarities.argmax()
        best_score = similarities[best_index]

        if best_score > 0:
            best_row = self.df.iloc[best_index]
            context = str(best_row["context"]).strip()
            answer = str(best_row["answer"]).strip()

            generated_response = (
                f"Based on the retrieved information, {answer} "
                f"This answer is supported by the following context: {context}"
            )

            return {
                "retrieved_question": best_row["question"],
                "retrieved_context": context,
                "answer": generated_response,
                "score": float(best_score)
            }

        return {
            "retrieved_question": None,
            "retrieved_context": None,
            "answer": "Sorry, I could not find enough information to generate a response.",
            "score": 0.0
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
    chatbot = TemplateRAGChatbot(csv_path)
    chatbot.run()
