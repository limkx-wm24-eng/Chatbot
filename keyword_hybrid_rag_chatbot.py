import re
import pandas as pd
from difflib import get_close_matches


class KeywordHybridRAGChatbot:
    def __init__(self, csv_file, similarity_threshold=0.18, top_k=5):
        self.csv_file = csv_file
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

        self.df = None

        self.allowed_keywords = {
            "tarumt", "tarc", "admission", "apply", "application", "register",
            "registration", "documents", "document", "requirements", "requirement",
            "entry", "qualification", "qualifications", "eligibility",
            "programme", "program", "programmes", "programs", "course", "courses",
            "faculty", "faculties", "diploma", "degree", "postgraduate",
            "fees", "fee", "payment", "payments", "cost", "price", "bank", "transfer",
            "scholarship", "scholarships", "financial", "aid", "loan", "ptptn",
            "hostel", "accommodation", "room", "rooms",
            "campus", "campuses", "branch", "branches", "location", "located",
            "intake", "intakes", "january", "may", "june", "september", "november",
            "contact", "phone", "email", "hotline", "office", "enquiry", "enquiries",
            "timetable", "schedule", "class", "classes", "calendar",
            "facilities", "facility", "library", "lab", "labs", "wifi",
            "gym", "canteen", "cafeteria", "sports", "clubs", "societies",
            "setapak", "penang", "perak", "johor", "pahang", "sabah"
        }

        self.synonym_map = {
            "program": "programme",
            "programs": "programmes",
            "programme": "programmes",
            "courses": "course",
            "fees": "fee",
            "payments": "payment",
            "costs": "cost",
            "prices": "price",
            "applications": "application",
            "branches": "campus",
            "branch": "campus",
            "campuses": "campus",
            "facilities": "facility",
            "documents": "document",
            "requirements": "requirement",
            "qualifications": "qualification",
            "enrol": "enroll",
            "enrolment": "enrollment",
            "hostel": "accommodation",
            "dorm": "accommodation",
            "dormitory": "accommodation",
            "location": "campus",
            "locations": "campus",
            "intakes": "intake",
            "subjects": "course",
            "computer lab": "lab",
            "food court": "canteen",
            "study area": "library",
            "wifi": "internet"
        }

        self.intent_patterns = {
            "admission": [
                r"\badmission\b", r"\bapply\b", r"\bapplication\b",
                r"\bregister\b", r"\bregistration\b", r"\benroll\b"
            ],
            "requirements": [
                r"\brequirement\b", r"\bentry\b", r"\bqualification\b",
                r"\beligibility\b", r"\bwhat.*need\b"
            ],
            "programme": [
                r"\bprogramme\b", r"\bprogram\b", r"\bprogrammes\b", r"\bprograms\b",
                r"\bdiploma\b", r"\bdegree\b", r"\bpostgraduate\b"
            ],
            "course": [
                r"\bcourse\b", r"\bcourses\b", r"\bfaculty\b", r"\bsubject\b"
            ],
            "fees": [
                r"\bfee\b", r"\bpayment\b", r"\bcost\b", r"\bprice\b", r"\bbank transfer\b"
            ],
            "scholarship": [
                r"\bscholarship\b", r"\bfinancial aid\b", r"\bloan\b", r"\bptptn\b"
            ],
            "accommodation": [
                r"\bhostel\b", r"\baccommodation\b", r"\broom\b"
            ],
            "campus": [
                r"\bcampus\b", r"\bbranch\b", r"\blocation\b", r"\blocated\b",
                r"\bsetapak\b", r"\bpenang\b", r"\bperak\b", r"\bjohor\b",
                r"\bpahang\b", r"\bsabah\b"
            ],
            "intake": [
                r"\bintake\b", r"\bjanuary\b", r"\bmay\b", r"\bjune\b",
                r"\bseptember\b", r"\bnovember\b"
            ],
            "contact": [
                r"\bcontact\b", r"\bphone\b", r"\bemail\b", r"\bhotline\b",
                r"\boffice\b", r"\benquiry\b"
            ],
            "timetable": [
                r"\btimetable\b", r"\bschedule\b", r"\bclass\b", r"\bcalendar\b"
            ],
            "facilities": [
                r"\bfacility\b", r"\bfacilities\b", r"\blibrary\b", r"\blab\b",
                r"\bwifi\b", r"\bgym\b", r"\bcanteen\b", r"\bcafeteria\b",
                r"\bsports\b", r"\bclubs\b", r"\bsocieties\b"
            ]
        }

        self.greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
        self.thanks_words = {"thanks", "thank you", "thankyou", "much appreciated"}
        self.goodbye_words = {"bye", "goodbye", "see you", "see you later", "catch you later"}

        self.load_data()

    # =============================
    # TEXT NORMALIZATION
    # =============================
    def normalize_text(self, text):
        text = str(text).lower().strip()
        text = text.replace("&", " and ")

        phrase_map = {
            "how do i apply": "how to apply",
            "how can i apply": "how to apply",
            "application process": "how to apply",
            "how to make application": "how to apply",
            "make an application": "how to apply",
            "where is the campus": "campus location",
            "where are the campuses": "campus location",
            "what programmes": "programme",
            "what programs": "programme",
            "what courses": "course",
            "what subjects": "course",
            "how can i check my timetable": "timetable",
            "show me my schedule": "schedule",
            "who should i contact": "contact",
            "what month can i enroll": "intake",
            "when can i enroll": "intake"
        }

        for old, new in phrase_map.items():
            text = text.replace(old, new)

        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        for old, new in sorted(self.synonym_map.items(), key=lambda x: len(x[0]), reverse=True):
            text = re.sub(rf"\b{re.escape(old)}\b", new, text)

        return text

    def preprocess_text(self, text):
        text = self.normalize_text(text)

        custom_stopwords = {
            "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
            "is", "are", "am", "was", "were", "be", "been", "being",
            "do", "does", "did", "can", "could", "should", "would", "will",
            "may", "might", "shall", "must",
            "i", "me", "my", "mine", "we", "our", "ours",
            "you", "your", "yours",
            "please", "tell", "explain", "about", "know", "want", "need",
            "there", "any", "some", "the", "a", "an", "in", "on", "for", "of", "to"
        }

        words = [word for word in text.split() if word not in custom_stopwords]
        return " ".join(words)

    # =============================
    # INTENT / SCOPE
    # =============================
    def detect_intents(self, text):
        intents = set()
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    intents.add(intent)
                    break
        return intents

    def is_greeting(self, query):
        return self.normalize_text(query) in self.greetings

    def is_thanks(self, query):
        return self.normalize_text(query) in self.thanks_words

    def is_goodbye(self, query):
        return self.normalize_text(query) in self.goodbye_words

    def is_in_scope(self, clean_query):
        query_words = clean_query.split()
        if not query_words:
            return False

        if len(set(query_words).intersection(self.allowed_keywords)) > 0:
            return True

        for word in query_words:
            close = get_close_matches(word, list(self.allowed_keywords), n=1, cutoff=0.82)
            if close:
                return True

        return False

    # =============================
    # LOAD DATA
    # =============================
    def load_data(self):
        try:
            self.df = pd.read_csv(self.csv_file, encoding="utf-8")
            self.df = self.df.drop_duplicates(subset=["question", "context", "answer"]).reset_index(drop=True)

            required_columns = ["question", "context", "answer"]
            for col in required_columns:
                if col not in self.df.columns:
                    raise ValueError(f"Missing required column: {col}")

            self.df = self.df.fillna("")
            self.df["question"] = self.df["question"].astype(str)
            self.df["context"] = self.df["context"].astype(str)
            self.df["answer"] = self.df["answer"].astype(str)

            self.df["clean_question"] = self.df["question"].apply(self.preprocess_text)
            self.df["clean_context"] = self.df["context"].apply(self.preprocess_text)
            self.df["clean_answer"] = self.df["answer"].apply(self.preprocess_text)

            self.df["combined_text"] = (
                self.df["clean_question"] + " " +
                self.df["clean_context"] + " " +
                self.df["clean_answer"]
            )

            print("Dataset loaded successfully.")
            print(f"Total records: {len(self.df)}")

        except FileNotFoundError:
            print(f"Error: File '{self.csv_file}' not found.")
            raise
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    # =============================
    # FLOWCHART STEP: SELECT METHOD
    # =============================
    def is_simple_keyword_query(self, clean_query):
        query_words = clean_query.split()

        if len(query_words) <= 3:
            return True

        direct_keywords = {
            "admission", "apply", "requirement", "document", "fee", "scholarship",
            "accommodation", "campus", "intake", "contact", "timetable", "facility",
            "programme", "course", "hostel", "location", "library", "wifi"
        }

        if len(set(query_words).intersection(direct_keywords)) >= 1:
            return True

        return False

    def select_retrieval_method(self, clean_query):
        if self.is_simple_keyword_query(clean_query):
            return "keyword_matching"
        return "weighted_keyword_scoring"

    # =============================
    # RETRIEVAL METHOD 1: EXACT KEYWORD
    # =============================
    def keyword_overlap_score(self, query, text):
        q_words = set(query.split())
        t_words = set(text.split())

        if not q_words or not t_words:
            return 0.0

        overlap = q_words.intersection(t_words)
        return len(overlap) / max(len(q_words), 1)

    def exact_keyword_search(self, clean_query):
        query_intents = self.detect_intents(clean_query)
        candidates = []

        for i in range(len(self.df)):
            row = self.df.iloc[i]
            overlap_q = self.keyword_overlap_score(clean_query, row["clean_question"])
            overlap_c = self.keyword_overlap_score(clean_query, row["clean_context"])
            overlap_a = self.keyword_overlap_score(clean_query, row["clean_answer"])

            overlap = max(overlap_q, overlap_c, overlap_a)

            row_intents = self.detect_intents(row["combined_text"])
            intent_score = 0.0
            if query_intents and row_intents:
                intent_score = len(query_intents.intersection(row_intents)) / max(len(query_intents), 1)

            final_score = 0.80 * overlap + 0.20 * intent_score
            candidates.append((i, final_score, overlap, intent_score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:self.top_k]

    # =============================
    # RETRIEVAL METHOD 2: WEIGHTED KEYWORD
    # =============================
    def weighted_keyword_search(self, clean_query):
        query_words = clean_query.split()
        query_intents = self.detect_intents(clean_query)
        candidates = []

        for i in range(len(self.df)):
            row = self.df.iloc[i]
            row_text = row["combined_text"]

            exact_match_count = 0
            fuzzy_match_count = 0

            for word in query_words:
                if word in row_text.split():
                    exact_match_count += 1
                else:
                    close = get_close_matches(word, row_text.split(), n=1, cutoff=0.85)
                    if close:
                        fuzzy_match_count += 1

            exact_score = exact_match_count / max(len(query_words), 1)
            fuzzy_score = fuzzy_match_count / max(len(query_words), 1)
            overlap_score = self.keyword_overlap_score(clean_query, row_text)

            row_intents = self.detect_intents(row_text)
            intent_score = 0.0
            if query_intents and row_intents:
                intent_score = len(query_intents.intersection(row_intents)) / max(len(query_intents), 1)

            final_score = (
                0.45 * exact_score +
                0.20 * fuzzy_score +
                0.20 * overlap_score +
                0.15 * intent_score
            )

            candidates.append((i, final_score, exact_score, fuzzy_score, overlap_score, intent_score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:self.top_k]

    # =============================
    # RESPONSE FORMAT
    # =============================
    def fallback_scope_message(self):
        return (
            "Sorry, I can only answer TAR UMT FAQ-related questions such as "
            "admission, entry requirements, required documents, fees, scholarships, "
            "hostel, facilities, registration, programmes, campuses, intake, and contact details."
        )

    def fallback_no_match_message(self):
        return (
            "Sorry, I could not find a confident answer in the TAR UMT FAQ dataset. "
            "Please try asking in a more specific way."
        )

    def format_answer(self, best_row, confidence):
        answer = best_row["answer"].strip()

        if confidence == "High":
            return f"Based on the TAR UMT FAQ information, {answer}"
        elif confidence == "Medium":
            return f"I found a likely answer: {answer}"
        else:
            return self.fallback_no_match_message()

    def choose_best_answer(self, candidates):
        if not candidates:
            return None, 0.0, []
        best = candidates[0]
        return best[0], best[1], candidates

    # =============================
    # MAIN RESPONSE
    # =============================
    def get_response(self, user_query):
        if self.is_greeting(user_query):
            return {
                "retrieval_method": "rule_based",
                "retrieved_question": None,
                "retrieved_context": None,
                "answer": "Hello! I am the TAR UMT FAQ chatbot. How can I help you today?",
                "score": 1.0,
                "confidence": "High",
                "top_matches": []
            }

        if self.is_thanks(user_query):
            return {
                "retrieval_method": "rule_based",
                "retrieved_question": None,
                "retrieved_context": None,
                "answer": "You're welcome. Feel free to ask another TAR UMT-related question.",
                "score": 1.0,
                "confidence": "High",
                "top_matches": []
            }

        if self.is_goodbye(user_query):
            return {
                "retrieval_method": "rule_based",
                "retrieved_question": None,
                "retrieved_context": None,
                "answer": "Goodbye. Have a nice day.",
                "score": 1.0,
                "confidence": "High",
                "top_matches": []
            }

        clean_query = self.preprocess_text(user_query)

        if not clean_query:
            return {
                "retrieval_method": "none",
                "retrieved_question": None,
                "retrieved_context": None,
                "answer": "Please enter a valid TAR UMT-related question.",
                "score": 0.0,
                "confidence": "Low",
                "top_matches": []
            }

        if not self.is_in_scope(clean_query):
            return {
                "retrieval_method": "scope_filter",
                "retrieved_question": None,
                "retrieved_context": None,
                "answer": self.fallback_scope_message(),
                "score": 0.0,
                "confidence": "Low",
                "top_matches": []
            }

        retrieval_method = self.select_retrieval_method(clean_query)

        if retrieval_method == "keyword_matching":
            candidates = self.exact_keyword_search(clean_query)
        else:
            candidates = self.weighted_keyword_search(clean_query)

        best_index, best_score, top_candidates = self.choose_best_answer(candidates)

        if best_index is None or best_score < self.similarity_threshold:
            return {
                "retrieval_method": retrieval_method,
                "retrieved_question": None,
                "retrieved_context": None,
                "answer": self.fallback_no_match_message(),
                "score": float(best_score if best_index is not None else 0.0),
                "confidence": "Low",
                "top_matches": []
            }

        best_row = self.df.iloc[best_index]

        if best_score >= 0.60:
            confidence = "High"
        elif best_score >= 0.30:
            confidence = "Medium"
        else:
            confidence = "Low"

        final_answer = self.format_answer(best_row, confidence)

        top_matches = []
        used_questions = set()

        for item in top_candidates:
            idx = item[0]
            row = self.df.iloc[idx]
            q = row["question"].strip()

            if q in used_questions:
                continue
            used_questions.add(q)

            if retrieval_method == "keyword_matching":
                _, final_score, overlap_score, intent_score = item
                top_matches.append({
                    "question": q,
                    "score": round(float(final_score), 4),
                    "overlap_score": round(float(overlap_score), 4),
                    "intent_score": round(float(intent_score), 4)
                })
            else:
                _, final_score, exact_score, fuzzy_score, overlap_score, intent_score = item
                top_matches.append({
                    "question": q,
                    "score": round(float(final_score), 4),
                    "exact_score": round(float(exact_score), 4),
                    "fuzzy_score": round(float(fuzzy_score), 4),
                    "overlap_score": round(float(overlap_score), 4),
                    "intent_score": round(float(intent_score), 4)
                })

        return {
            "retrieval_method": retrieval_method,
            "retrieved_question": best_row["question"],
            "retrieved_context": best_row["context"],
            "answer": final_answer,
            "score": float(best_score),
            "confidence": confidence,
            "top_matches": top_matches
        }

    def display_result(self, result):
        print("\n" + "=" * 80)
        print("Chatbot Type       : Keyword Hybrid RAG")
        print(f"Retrieval Method   : {result['retrieval_method']}")
        print(f"Similarity Score   : {result['score']:.4f}")
        print(f"Confidence Level   : {result['confidence']}")
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
                if result["retrieval_method"] == "keyword_matching":
                    print(
                        f"{i}. {item['question']} "
                        f"(final={item['score']}, overlap={item['overlap_score']}, intent={item['intent_score']})"
                    )
                else:
                    print(
                        f"{i}. {item['question']} "
                        f"(final={item['score']}, exact={item['exact_score']}, "
                        f"fuzzy={item['fuzzy_score']}, overlap={item['overlap_score']}, "
                        f"intent={item['intent_score']})"
                    )

        print("=" * 80 + "\n")

    def run(self):
        print("TAR UMT Keyword Hybrid RAG Chatbot")
        print("Type 'exit' or 'quit' to stop.\n")

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

    chatbot = KeywordHybridRAGChatbot(
        csv_file=csv_path,
        similarity_threshold=0.18,
        top_k=5
    )
    chatbot.run()