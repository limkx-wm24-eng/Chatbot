import pandas as pd
import re
import string
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Running Transformer RAG Chatbot")


class TransformerRAGChatbot:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.doc_embeddings = None
        self.confidence_log = []

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
            "courses": {
                "course", "courses", "program", "programs", "programme", "programmes",
                "programe", "study", "studies", "subject", "subjects", "diploma",
                "degree", "postgraduate", "faculty", "major"
            },
            "admission": {
                "admission", "apply", "application", "register", "registration",
                "intake", "intakes", "entry", "requirement", "requirements",
                "document", "documents", "enroll", "enrol", "month", "months",
                "january", "september", "november", "may", "june"
            },
            "fees": {
                "fee", "fees", "tuition", "payment", "payments", "cost", "costs",
                "price", "prices", "charges", "expensive"
            },
            "accommodation": {
                "hostel", "accommodation", "room", "rooms"
            },
            "scholarship": {
                "scholarship", "scholarships", "financial", "aid", "bursary",
                "bursaries", "loan", "loans", "waiver"
            },
            "contact": {
                "contact", "phone", "email", "hotline", "office"
            },
            "timetable": {
                "timetable", "schedule", "class", "classes", "calendar"
            },
            "location": {
                "location", "located", "campus", "address", "setapak", "kuala", "lumpur",
                "penang", "perak", "johor", "pahang", "sabah"
            },
            "facilities": {
                "facility", "facilities", "library", "lab", "labs", "wifi",
                "sports", "gym", "cafeteria", "canteen", "club", "clubs", "society", "societies"
            },
            "transport": {
                "bus", "buses", "transport", "transportation", "travel", "shuttle"
            }
        }

        self.sub_intent_rules = {
            "course_list": [
                "what course", "what courses", "which course", "which courses",
                "courses provided", "courses offered", "courses available",
                "faculty", "faculties", "engineering", "information technology",
                "business", "accounting", "field of study", "what can i study"
            ],
            "programme_list": [
                "what programme", "what programmes", "what program", "what programs",
                "programme provided", "programmes provided", "program provided", "programs provided",
                "programme offered", "programmes offered", "program offered", "programs offered",
                "diploma", "degree", "postgraduate"
            ],
            "intake_info": [
                "when is the intake", "when are the intakes", "intake", "intakes",
                "intake month", "intake months", "what month", "what months",
                "new intake", "january", "may june", "september", "november"
            ],
            "admission_apply": [
                "how to apply", "how do i apply", "application", "apply",
                "application portal", "submit application", "documents required",
                "required documents", "registration", "register"
            ],
            "fees_detail": [
                "how much", "fee", "fees", "tuition fee", "cost", "price", "payment"
            ],
            "hostel_info": [
                "hostel", "accommodation", "hostel room", "student accommodation", "hostel available"
            ],
            "scholarship_info": [
                "scholarship", "scholarships", "financial aid", "bursary", "loan"
            ],
            "contact_info": [
                "contact", "email", "phone", "hotline"
            ],
            "timetable_info": [
                "timetable", "schedule", "class schedule"
            ],
            "location_info": [
                "where is", "where are", "location", "located", "address",
                "campus location", "where are the campus", "where is the campus",
                "setapak", "kuala lumpur", "penang", "perak", "johor", "pahang", "sabah"
            ],
            "transport_info": [
                "bus", "buses", "transport", "transportation", "travel",
                "shuttle", "campus bus", "transport service", "how do students travel"
            ],
            "facilities_info": [
                "facilities", "facility", "library", "lab", "wifi", "gym", "canteen", "cafeteria",
                "club", "clubs", "society", "societies", "sports"
            ]
        }

        self.campus_branches = {
            "setapak", "kuala lumpur", "penang", "perak", "johor", "pahang", "sabah"
        }

        self.load_data()
        self.build_embeddings()

    def clean_text(self, text):
        text = str(text).lower().strip()
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text)
        return text

    def normalize_word(self, word):
        synonym_map = {
            "course": "courses",
            "program": "programme",
            "programs": "programme",
            "programme": "programme",
            "programmes": "programme",
            "programe": "programme",
            "study": "courses",
            "studies": "courses",
            "subject": "courses",
            "subjects": "courses",
            "major": "courses",

            "fee": "fees",
            "tuition": "fees",
            "payment": "fees",
            "payments": "fees",
            "cost": "fees",
            "costs": "fees",
            "price": "fees",
            "prices": "fees",
            "charges": "fees",

            "apply": "admission",
            "application": "admission",
            "register": "admission",
            "registration": "admission",
            "enroll": "admission",
            "enrol": "admission",

            "intakes": "intake",
            "month": "intake",
            "months": "intake",

            "hostel": "accommodation",
            "room": "accommodation",
            "rooms": "accommodation",

            "scholarships": "scholarship",
            "bursary": "scholarship",
            "bursaries": "scholarship",
            "loan": "scholarship",
            "loans": "scholarship",

            "email": "contact",
            "phone": "contact",
            "hotline": "contact",

            "schedule": "timetable",
            "class": "timetable",
            "classes": "timetable",

            "located": "location",
            "campus": "location",

            "facility": "facilities",
            "library": "facilities",
            "lab": "facilities",
            "labs": "facilities",
            "wifi": "facilities",
            "gym": "facilities",
            "canteen": "facilities",
            "cafeteria": "facilities",
            "club": "facilities",
            "clubs": "facilities",
            "society": "facilities",
            "societies": "facilities",

            "bus": "transport",
            "buses": "transport",
            "travel": "transport",
            "transportation": "transport",
            "shuttle": "transport"
        }
        return synonym_map.get(word, word)

    def tokenize(self, text):
        cleaned = self.clean_text(text)
        words = []
        for word in cleaned.split():
            normalized = self.normalize_word(word)
            if normalized not in self.stop_words and len(normalized) > 2:
                words.append(normalized)
        return words

    def detect_branch(self, query):
        query_clean = self.clean_text(query)
        for branch in sorted(self.campus_branches, key=len, reverse=True):
            if branch in query_clean:
                return branch
        return None

    def detect_topic(self, text):
        words = set(self.tokenize(text))
        best_topic = None
        best_score = 0

        for topic, keywords in self.topic_keywords.items():
            normalized_keywords = {self.normalize_word(k) for k in keywords}
            score = len(words.intersection(normalized_keywords))
            if score > best_score:
                best_score = score
                best_topic = topic

        return best_topic

    def detect_sub_intent(self, text):
        cleaned = self.clean_text(text)
        tokens = set(self.tokenize(cleaned))

        if "transport" in tokens:
            return "transport_info"

        intake_words = {"intake", "january", "september", "november", "may", "june"}
        if tokens.intersection(intake_words):
            return "intake_info"

        hostel_words = {"accommodation", "hostel"}
        if tokens.intersection(hostel_words):
            return "hostel_info"

        scholarship_words = {"scholarship", "bursary", "loan", "waiver"}
        if tokens.intersection(scholarship_words):
            return "scholarship_info"

        fees_words = {"fees", "tuition", "payment", "cost", "price"}
        if tokens.intersection(fees_words):
            return "fees_detail"

        contact_words = {"contact", "email", "phone", "hotline", "office"}
        if tokens.intersection(contact_words):
            return "contact_info"

        timetable_words = {"timetable", "schedule", "calendar"}
        if tokens.intersection(timetable_words):
            return "timetable_info"

        location_words = {"location", "address", "setapak", "kuala", "lumpur", "penang", "perak", "johor", "pahang", "sabah"}
        if tokens.intersection(location_words):
            return "location_info"

        course_words = {
            "courses", "course", "faculty", "faculties",
            "engineering", "information", "technology", "business", "accounting",
            "study", "studies", "major", "subject", "subjects"
        }
        if tokens.intersection(course_words):
            return "course_list"

        programme_words = {"diploma", "degree", "postgraduate", "programme", "program", "programmes", "programs"}
        if tokens.intersection(programme_words):
            return "programme_list"

        admission_words = {"admission", "application", "apply", "register", "registration", "documents", "requirements"}
        if tokens.intersection(admission_words):
            return "admission_apply"

        facilities_words = {"facilities", "library", "lab", "wifi", "gym", "canteen", "cafeteria", "club", "society", "sports"}
        if tokens.intersection(facilities_words):
            return "facilities_info"

        for sub_intent, phrases in self.sub_intent_rules.items():
            for phrase in phrases:
                if phrase in cleaned:
                    return sub_intent

        return None

    def detect_question_type(self, query):
        query = self.clean_text(query)

        if any(x in query for x in ["how many", "number of", "total number"]):
            return "count"

        if any(x in query for x in ["when", "what month", "what months"]):
            return "time"

        if any(x in query for x in ["list", "show", "all", "what are", "which are"]):
            return "list"

        if query.startswith(("is ", "are ", "does ", "do ", "can ", "has ", "have ")):
            return "yesno"

        return "normal"

    def get_confidence_level(self, confidence):
        if confidence >= 75:
            return "High"
        elif confidence >= 25:
            return "Medium"
        else:
            return "Low"

    def load_data(self):
        self.df = pd.read_csv(self.csv_file).fillna("")

        required_columns = ["question", "context", "answer"]
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        self.df["question"] = self.df["question"].astype(str).str.strip()
        self.df["context"] = self.df["context"].astype(str).str.strip()
        self.df["answer"] = self.df["answer"].astype(str).str.strip()

        self.df = self.df[
            (self.df["question"] != "") &
            (self.df["context"] != "") &
            (self.df["answer"] != "")
        ].reset_index(drop=True)

        self.df["question_clean"] = self.df["question"].apply(self.clean_text)
        self.df["context_clean"] = self.df["context"].apply(self.clean_text)
        self.df["answer_clean"] = self.df["answer"].apply(self.clean_text)

        self.df["topic"] = self.df.apply(
            lambda row: self.detect_topic(f"{row['question']} {row['context']} {row['answer']}"),
            axis=1
        )

        self.df["sub_intent"] = self.df.apply(
            lambda row: self.detect_sub_intent(f"{row['question']} {row['context']} {row['answer']}"),
            axis=1
        )

        self.df["combined_text"] = self.df.apply(
            lambda row: f"Question: {row['question']} Context: {row['context']} Answer: {row['answer']}",
            axis=1
        )

        print("Dataset loaded successfully.")
        print(f"Total records: {len(self.df)}")

    def build_embeddings(self):
        texts = self.df["combined_text"].tolist()
        self.doc_embeddings = self.model.encode(texts, convert_to_numpy=True)
        print("Transformer embeddings built successfully.")

    def retrieve_top_k(self, query, top_k=3):
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        scores = cosine_similarity(query_embedding, self.doc_embeddings)[0]

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []

        for idx in top_indices:
            row = self.df.iloc[idx]
            results.append((float(scores[idx]), row))

        return results

    def apply_logic_bonus(self, score, row, user_topic, user_sub_intent, branch_name):
        final_score = score
        combined = f"{row['question_clean']} {row['context_clean']} {row['answer_clean']}"

        if user_topic and row["topic"] == user_topic:
            final_score += 0.08
        elif user_topic and row["topic"] != user_topic:
            final_score -= 0.08

        if user_sub_intent and row["sub_intent"] == user_sub_intent:
            final_score += 0.10
        elif user_sub_intent and row["sub_intent"] != user_sub_intent:
            final_score -= 0.10

        if branch_name:
            if branch_name in combined:
                final_score += 0.10
            else:
                final_score -= 0.10

        return final_score

    def build_result_dict(
        self,
        best_score,
        confidence,
        confidence_level,
        question_type,
        branch_name,
        user_topic,
        user_sub_intent,
        best_row,
        answer
    ):
        return {
            "score": round(best_score, 4),
            "confidence": round(confidence, 2),
            "confidence_level": confidence_level,
            "question_type": question_type,
            "detected_branch": branch_name if branch_name else "None",
            "matched_topic": user_topic or "None",
            "matched_sub_intent": user_sub_intent or "None",
            "retrieved_topic": best_row["topic"] if best_row["topic"] else "None",
            "retrieved_sub_intent": best_row["sub_intent"] if best_row["sub_intent"] else "None",
            "retrieved_question": best_row["question"],
            "retrieved_context": best_row["context"],
            "answer": answer
        }

    def get_response(self, user_query):
        if not user_query.strip():
            return {
                "score": 0.0,
                "confidence": 0.0,
                "confidence_level": "Low",
                "question_type": "invalid",
                "detected_branch": "None",
                "matched_topic": "None",
                "matched_sub_intent": "None",
                "retrieved_topic": "None",
                "retrieved_sub_intent": "None",
                "retrieved_question": None,
                "retrieved_context": None,
                "answer": "Please enter a valid question."
            }

        user_topic = self.detect_topic(user_query)
        user_sub_intent = self.detect_sub_intent(user_query)
        question_type = self.detect_question_type(user_query)
        branch_name = self.detect_branch(user_query)

        ranked = self.retrieve_top_k(user_query, top_k=3)

        if not ranked:
            return {
                "score": 0.0,
                "confidence": 0.0,
                "confidence_level": "Low",
                "question_type": question_type,
                "detected_branch": branch_name if branch_name else "None",
                "matched_topic": user_topic or "None",
                "matched_sub_intent": user_sub_intent or "None",
                "retrieved_topic": "None",
                "retrieved_sub_intent": "None",
                "retrieved_question": None,
                "retrieved_context": None,
                "answer": "Sorry, I could not find a relevant answer in the dataset."
            }

        rescored = []
        for score, row in ranked:
            new_score = self.apply_logic_bonus(score, row, user_topic, user_sub_intent, branch_name)
            rescored.append((new_score, row))

        rescored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_row = rescored[0]

        confidence = max(0, min(best_score * 100, 100))
        confidence_level = self.get_confidence_level(confidence)

        answer = best_row["answer"]

        if user_sub_intent == "course_list":
            answer = "TARUMT offers courses in Engineering, Information Technology, Business and Accounting."
        elif user_sub_intent == "programme_list":
            answer = "TARUMT offers Diploma, Degree and Postgraduate programmes."

        if question_type == "yesno":
            if branch_name:
                combined = f"{best_row['question_clean']} {best_row['context_clean']} {best_row['answer_clean']}"
                if branch_name in combined:
                    if not answer.lower().startswith(("yes", "no")):
                        answer = f"Yes, {answer}"
                else:
                    answer = "No, there is no information about that campus."
            else:
                if not answer.lower().startswith(("yes", "no")):
                    answer = f"Yes, {answer}"

        if confidence < 25:
            answer = "I'm not confident enough. Could you rephrase your question?"

        result = self.build_result_dict(
            best_score=best_score,
            confidence=confidence,
            confidence_level=confidence_level,
            question_type=question_type,
            branch_name=branch_name,
            user_topic=user_topic,
            user_sub_intent=user_sub_intent,
            best_row=best_row,
            answer=answer
        )

        self.confidence_log.append({
            "query": user_query,
            "confidence": result["confidence"],
            "confidence_level": result["confidence_level"],
            "retrieved_question": result["retrieved_question"],
            "retrieved_topic": result["retrieved_topic"]
        })

        return result

    def display_result(self, result):
        print("\n" + "=" * 70)
        print("Chatbot Type       : Transformer RAG")
        print(f"Similarity Score   : {result['score']}")
        print(f"Confidence Score   : {result['confidence']}%")
        print(f"Confidence Level   : {result['confidence_level']}")
        print(f"Question Type      : {result['question_type']}")
        print(f"Detected Branch    : {result['detected_branch']}")
        print(f"Detected Topic     : {result['matched_topic']}")
        print(f"Detected Sub-Intent: {result['matched_sub_intent']}")
        print(f"Retrieved Topic    : {result['retrieved_topic']}")
        print(f"Retrieved SubIntent: {result['retrieved_sub_intent']}")
        print("-" * 70)
        print(f"Retrieved Question : {result['retrieved_question'] if result['retrieved_question'] else 'None'}")
        print("-" * 70)
        print("Retrieved Context  :")
        print(result["retrieved_context"] if result["retrieved_context"] else "None")
        print("-" * 70)
        print("Bot Answer         :")
        print(result["answer"])
        print("=" * 70 + "\n")

    def export_confidence_log(self, filename="confidence_results.csv"):
        if not self.confidence_log:
            print("No confidence results to export.")
            return

        df = pd.DataFrame(self.confidence_log)
        df.to_csv(filename, index=False)
        print(f"Confidence results exported to {filename}")

    def run(self):
        while True:
            user_query = input("You: ").strip()

            if user_query.lower() in ["exit", "quit"]:
                self.export_confidence_log()
                print("Goodbye.")
                break

            if not user_query:
                print("Please enter a valid question.")
                continue

            result = self.get_response(user_query)
            self.display_result(result)


if __name__ == "__main__":
    csv_path = "tarumt_faq_dataset.csv"
    bot = TransformerRAGChatbot(csv_path)
    bot.run()