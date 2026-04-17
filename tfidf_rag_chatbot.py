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
            # ENTRY REQUIREMENTS
            "minimum requirement": "entry requirements",
            "minimal requirement": "entry requirements",
            "requirements to apply": "entry requirements",
            "requirement to apply": "entry requirements",
            "requirements to entry": "entry requirements",
            "requirement to entry": "entry requirements",

            # CAMPUS
            "where is the campus": "campus location",
            "where is campus": "campus location",

            # APPLY
            "how to apply": "admission process",
            "how do i apply": "admission process",
            "how can i apply": "admission process",
            "where to apply": "admission process",
            "application process": "admission process",

            # MAKE APPLICATION (IMPORTANT FIX)
            "how to make the application": "admission process",
            "how to make application": "admission process",
            "make application": "admission process",
            "make an application": "admission process",

            # ENROLL → INTAKE (IMPORTANT FIX)
            "when can i enroll": "intake",
            "when can i enrol": "intake",
            "what month can i enroll": "intake",
            "what month can i enrol": "intake",
            "month to enroll": "intake",
            "month to enrol": "intake",
            "what is the month that i can enroll": "intake",
            "what is the month that i can enrol": "intake",

            # REGISTER COURSE
            "register the course": "entry requirements",
            "requirement needed to register the course": "entry requirements",

            # COURSE / PROGRAMME
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
    # RULE-BASED ANSWERS
    # =============================
    def rule_based_answer(self, query):
        q = self.normalize_text(query)

        # 🔥 1. INTAKE FIRST (IMPORTANT)
        if "intake" in q:
            return "TARUMT intakes are January, May/June, September and November."

        # 2. APPLY
        if "admission process" in q or ("apply" in q and "fee" not in q):
            return "You can apply through the TARUMT admission portal via the official website by selecting your programme and submitting the required documents."

        # 3. CAMPUS
        if ("campus" in q and "location" in q) or ("where" in q and "campus" in q):
            return "TARUMT has 6 campuses located in Kuala Lumpur, Penang, Perak, Johor, Pahang and Sabah."

        # 4. COURSE
        if "course" in q and "requirements" not in q:
            return "TARUMT offers courses in Engineering, Information Technology, Business and Accounting."

        # 5. PROGRAMME
        if "programme" in q or "diploma" in q or "degree" in q:
            return "TARUMT offers Diploma, Degree and Postgraduate programmes."

        # 6. ENTRY REQUIREMENTS
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
            return "Sorry, I could not find a relevant answer."

        row = self.df.iloc[best_idx]

        return row["context"] + " " + row["answer"]

    # =============================
    # MAIN RESPONSE
    # =============================
    def get_response(self, user_query):
        rule = self.rule_based_answer(user_query)
        if rule:
            return rule

        return self.tfidf_fallback(user_query)

    # =============================
    # RUN
    # =============================
    def run(self):
        print("TARUMT Chatbot Running...\n")

        while True:
            user_input = input("You: ")

            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye.")
                break

            response = self.get_response(user_input)
            print("Bot:", response)


if __name__ == "__main__":
    bot = TFIDFRAGChatbot("tarumt_faq_dataset.csv")
    bot.run()