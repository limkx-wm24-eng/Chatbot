import pandas as pd
import re
import string


class KeywordRAGChatbot:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None
        self.load_data()

    def load_data(self):
        try:
            self.df = pd.read_csv(self.csv_file)
            required_columns = ["question", "context", "answer"]
            for col in required_columns:
                if col not in self.df.columns:
                    raise ValueError(f"Missing required column: {col}")
            self.df = self.df.fillna("")
            print("Dataset loaded successfully.")
            print(f"Total records: {len(self.df)}")
        except FileNotFoundError:
            print(f"Error: File '{self.csv_file}' not found.")
            raise
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def preprocess_text(self, text):
        text = str(text).lower()
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        return set(text.split())

    def get_response(self, user_query):
        user_words = self.preprocess_text(user_query)

        best_score = 0
        best_row = None

        for _, row in self.df.iterrows():
            row_words = self.preprocess_text(row["question"] + " " + row["context"])
            overlap = len(user_words.intersection(row_words))

            if overlap > best_score:
                best_score = overlap
                best_row = row

        if best_row is not None and best_score > 0:
            return {
                "retrieved_question": best_row["question"],
                "retrieved_context": best_row["context"],
                "answer": best_row["answer"],
                "score": best_score
            }

        return {
            "retrieved_question": None,
            "retrieved_context": None,
            "answer": "Sorry, I could not find a relevant answer in the dataset.",
            "score": 0
        }

    def display_result(self, result):
        print("\n" + "=" * 70)
        print("Chatbot Type       : Keyword-Based RAG")
        print(f"Similarity Score   : {result['score']}")
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
    chatbot = KeywordRAGChatbot(csv_path)
    chatbot.run()
