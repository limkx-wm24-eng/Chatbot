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
                "sports", "gym", "cafeteria", "canteen"
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
                "facilities", "facility", "library", "lab", "wifi", "gym", "canteen", "cafeteria"
            ]
        }

        self.campus_branches = {
            "setapak", "kuala lumpur", "penang", "perak", "johor", "pahang", "sabah"
        }

        self.load_data()

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

    def similarity(self, a, b):
        return SequenceMatcher(None, a, b).ratio()

    def detect_branch(self, query):
        query_clean = self.clean_text(query)
        for branch in sorted(self.campus_branches, key=len, reverse=True):
            if branch in query_clean:
                return branch
        return None

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

        print("Dataset loaded successfully.")
        print(f"Total records: {len(self.df)}")

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

        facilities_words = {"facilities", "library", "lab", "wifi", "gym", "canteen", "cafeteria"}
        if tokens.intersection(facilities_words):
            return "facilities_info"

        best_intent = None
        best_score = 0.0
        for sub_intent, phrases in self.sub_intent_rules.items():
            score = 0.0
            for phrase in phrases:
                if phrase in cleaned:
                    score = max(score, 1.0)
                else:
                    score = max(score, self.similarity(cleaned, phrase))
            if score > best_score:
                best_score = score
                best_intent = sub_intent

        if best_score >= 0.70:
            return best_intent

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

    def fuzzy_token_match_score(self, query_tokens, row_tokens):
        score = 0.0
        for q in query_tokens:
            best = 0.0
            for r in row_tokens:
                sim = self.similarity(q, r)
                if sim > best:
                    best = sim

            if best >= 0.90:
                score += 2.0
            elif best >= 0.80:
                score += 1.2
            elif best >= 0.70:
                score += 0.5

        return score

    def phrase_score(self, query_clean, row_clean):
        best = self.similarity(query_clean, row_clean)
        partial_bonus = 0.0

        for phrases in self.sub_intent_rules.values():
            for phrase in phrases:
                if phrase in query_clean and phrase in row_clean:
                    partial_bonus += 4.0

        return (best * 8.0) + partial_bonus

    def column_score(self, query, row, question_type):
        query_clean = self.clean_text(query)
        query_tokens = self.tokenize(query)

        q_tokens = self.tokenize(row["question_clean"])
        c_tokens = self.tokenize(row["context_clean"])
        a_tokens = self.tokenize(row["answer_clean"])

        if question_type == "count":
            question_weight = 4.0
            context_weight = 2.0
            answer_weight = 6.0
        else:
            question_weight = 4.0
            context_weight = 2.0
            answer_weight = 3.0

        question_score = self.fuzzy_token_match_score(query_tokens, q_tokens) * question_weight
        context_score = self.fuzzy_token_match_score(query_tokens, c_tokens) * context_weight
        answer_score = self.fuzzy_token_match_score(query_tokens, a_tokens) * answer_weight

        phrase_question = self.phrase_score(query_clean, row["question_clean"]) * 1.5
        phrase_context = self.phrase_score(query_clean, row["context_clean"]) * 1.0
        phrase_answer = self.phrase_score(query_clean, row["answer_clean"]) * 1.2

        return question_score + context_score + answer_score + phrase_question + phrase_context + phrase_answer

    def logic_bonus(self, row, user_topic, user_sub_intent, question_type, branch_name):
        combined = f"{row['question_clean']} {row['context_clean']} {row['answer_clean']}"
        score = 0.0

        if user_topic == row["topic"]:
            score += 10.0
        elif user_topic and row["topic"] and user_topic != row["topic"]:
            score -= 18.0

        if user_sub_intent == row["sub_intent"]:
            score += 20.0
        elif user_sub_intent and row["sub_intent"] and user_sub_intent != row["sub_intent"]:
            score -= 18.0

        if branch_name:
            if branch_name in combined:
                score += 25.0
            else:
                score -= 20.0

        if question_type == "count":
            good = ["how many", "number of", "total", "6 campuses"]
            bad = ["how do", "how to", "travel", "transport", "shuttle"]
            for x in good:
                if x in combined:
                    score += 12.0
            for x in bad:
                if x in combined:
                    score -= 18.0

        if question_type == "time" and user_sub_intent == "intake_info":
            if any(x in combined for x in ["january", "may", "june", "september", "november", "intake"]):
                score += 12.0
            if any(x in combined for x in ["graduation", "fees", "application portal", "documents"]):
                score -= 15.0

        if question_type == "yesno" and user_sub_intent == "location_info":
            good = ["located", "campus", "branch", "setapak", "kuala lumpur", "penang", "perak", "johor", "pahang", "sabah"]
            bad = ["how many", "6 campuses", "total campuses", "transport", "bus", "travel", "shuttle"]
            for x in good:
                if x in combined:
                    score += 8.0
            for x in bad:
                if x in combined:
                    score -= 12.0

        if user_sub_intent == "intake_info":
            good = ["intake", "january", "may", "june", "september", "november"]
            bad = [
                "application portal", "required documents", "graduation",
                "fees", "payment", "settle outstanding fees",
                "before graduation", "hostel", "scholarship"
            ]
            for x in good:
                if x in combined:
                    score += 15.0
            for x in bad:
                if x in combined:
                    score -= 20.0

            if user_sub_intent == "course_list":
                good = ["engineering", "information technology", "business", "accounting", "faculty", "faculties"]
                bad = ["application", "financial aid", "hostel", "intake"]
                for x in good:
                    if x in combined:
                        score += 10.0
                for x in bad:
                    if x in combined:
                        score -= 14.0

            if user_sub_intent == "programme_list":
                good = ["diploma", "degree", "postgraduate", "programme", "program"]
                bad = ["counselling", "placement", "application", "financial aid", "student support"]
                for x in good:
                    if x in combined:
                        score += 10.0
                for x in bad:
                    if x in combined:
                        score -= 14.0

        if user_sub_intent == "admission_apply":
            good = ["apply online", "application portal", "required documents", "official portal"]
            bad = ["january", "september", "november", "scholarship", "intake"]
            for x in good:
                if x in combined:
                    score += 10.0
            for x in bad:
                if x in combined:
                    score -= 10.0

        if user_sub_intent == "hostel_info":
            good = ["hostel", "accommodation", "hostel room", "student accommodation"]
            bad = ["stay updated", "announcements", "emails", "deadlines", "informed"]
            for x in good:
                if x in combined:
                    score += 10.0
            for x in bad:
                if x in combined:
                    score -= 15.0

        if user_sub_intent == "scholarship_info":
            good = ["scholarship", "financial aid", "bursary", "loan", "waiver"]
            bad = ["application portal", "hostel", "timetable", "announcements"]
            for x in good:
                if x in combined:
                    score += 10.0
            for x in bad:
                if x in combined:
                    score -= 12.0

        if user_sub_intent == "contact_info":
            good = ["contact", "email", "phone", "hotline", "office"]
            bad = ["scholarship", "hostel", "intake", "courses offered"]
            for x in good:
                if x in combined:
                    score += 8.0
            for x in bad:
                if x in combined:
                    score -= 10.0

        if user_sub_intent == "timetable_info":
            good = ["timetable", "schedule", "class schedule"]
            bad = ["hostel", "scholarship", "application portal"]
            for x in good:
                if x in combined:
                    score += 8.0
            for x in bad:
                if x in combined:
                    score -= 10.0

        if user_sub_intent == "location_info":
            good = ["located", "location", "address", "kuala lumpur", "penang", "perak", "johor", "pahang", "sabah", "campus", "setapak"]
            bad = ["bus", "transport", "travel", "shuttle"]
            for x in good:
                if x in combined:
                    score += 12.0
            for x in bad:
                if x in combined:
                    score -= 20.0

        if user_sub_intent == "transport_info":
            good = ["bus", "transport", "travel", "shuttle"]
            bad = ["located", "address", "kuala lumpur", "penang", "perak", "johor", "pahang", "sabah", "setapak"]
            for x in good:
                if x in combined:
                    score += 12.0
            for x in bad:
                if x in combined:
                    score -= 15.0

        if user_sub_intent == "fees_detail":
            good = ["fees", "tuition", "payment", "cost", "price"]
            bad = ["hostel", "scholarship", "intake", "application portal"]
            for x in good:
                if x in combined:
                    score += 10.0
            for x in bad:
                if x in combined:
                    score -= 12.0

        if user_sub_intent == "facilities_info":
            good = ["facilities", "library", "lab", "wifi", "gym", "canteen", "cafeteria"]
            bad = ["application portal", "scholarship", "intake", "hostel"]
            for x in good:
                if x in combined:
                    score += 8.0
            for x in bad:
                if x in combined:
                    score -= 10.0

        return score

    def total_score(self, query, row, user_topic, user_sub_intent, question_type, branch_name):
        return round(
            self.column_score(query, row, question_type) +
            self.logic_bonus(row, user_topic, user_sub_intent, question_type, branch_name),
            4
        )

    def rank_rows(self, user_query, top_k=5):
        user_topic = self.detect_topic(user_query)
        user_sub_intent = self.detect_sub_intent(user_query)
        question_type = self.detect_question_type(user_query)
        branch_name = self.detect_branch(user_query)

        candidate_df = self.df.copy()

        if branch_name:
            branch_df = candidate_df[
                candidate_df["question_clean"].str.contains(branch_name) |
                candidate_df["context_clean"].str.contains(branch_name) |
                candidate_df["answer_clean"].str.contains(branch_name)
            ]
            if not branch_df.empty:
                candidate_df = branch_df

        if user_topic:
            topic_df = candidate_df[candidate_df["topic"] == user_topic]
            if not topic_df.empty:
                candidate_df = topic_df

        if user_sub_intent:
            sub_df = candidate_df[candidate_df["sub_intent"] == user_sub_intent]
            if not sub_df.empty:
                candidate_df = sub_df
            elif user_sub_intent == "intake_info":
                intake_mask = (
                    candidate_df["question_clean"].str.contains("intake|january|september|november|may|june", regex=True) |
                    candidate_df["context_clean"].str.contains("intake|january|september|november|may|june", regex=True) |
                    candidate_df["answer_clean"].str.contains("intake|january|september|november|may|june", regex=True)
                )
                fallback_df = candidate_df[intake_mask]
                if not fallback_df.empty:
                    candidate_df = fallback_df

        scored = []
        for _, row in candidate_df.iterrows():
            score = self.total_score(
                user_query,
                row,
                user_topic,
                user_sub_intent,
                question_type,
                branch_name
            )
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        return user_topic, user_sub_intent, question_type, branch_name, scored[:top_k]

    def get_response(self, user_query):
        user_topic, user_sub_intent, question_type, branch_name, ranked = self.rank_rows(user_query, top_k=3)

        if not ranked:
            return {
                "score": 0.0,
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

        best_score, best_row = ranked[0]
        answer = best_row["answer"]

        query_clean = self.clean_text(user_query)

        if user_sub_intent == "course_list":
            answer = "TARUMT offers courses in Engineering, Information Technology, Business and Accounting."
        elif user_sub_intent == "programme_list":
            answer = "TARUMT offers Diploma, Degree and Postgraduate programmes."

        if question_type == "yesno":
            if branch_name:
                if branch_name in best_row["question_clean"] or branch_name in best_row["context_clean"] or branch_name in best_row["answer_clean"]:
                    if not answer.lower().startswith(("yes", "no")):
                        answer = f"Yes, {answer}"
                else:
                    answer = "No, there is no information about that campus."
            else:
                if not answer.lower().startswith(("yes", "no")):
                    answer = f"Yes, {answer}"

        if best_score < 22:
            return {
                "score": best_score,
                "question_type": question_type,
                "detected_branch": branch_name if branch_name else "None",
                "matched_topic": user_topic or "None",
                "matched_sub_intent": user_sub_intent or "None",
                "retrieved_topic": best_row["topic"] if best_row["topic"] else "None",
                "retrieved_sub_intent": best_row["sub_intent"] if best_row["sub_intent"] else "None",
                "retrieved_question": best_row["question"],
                "retrieved_context": best_row["context"],
                "answer": "Sorry, I am not confident enough to give the correct answer."
            }

        return {
            "score": best_score,
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

    def display_result(self, result):
        print("\n" + "=" * 70)
        print("Chatbot Type       : Smart Keyword RAG")
        print(f"Similarity Score   : {result['score']}")
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
    bot = SmartKeywordRAGChatbot(csv_path)
    bot.run()