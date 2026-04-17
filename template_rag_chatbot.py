import re
import pandas as pd
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TemplateRAGChatbot:
    def __init__(self, csv_file, similarity_threshold=0.22, top_k=5):
        self.csv_file = csv_file
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

        self.df = None
        self.word_vectorizer = None
        self.char_vectorizer = None
        self.word_matrix = None
        self.char_matrix = None

        self.allowed_keywords = {
            "tarumt", "tarc", "admission", "admissions", "apply", "application",
            "document", "documents", "offer", "offers", "admit", "admitted",
            "registration", "register", "enrol", "enroll", "enrollment", "enrolment",
            "fee", "fees", "payment", "payments", "scholarship", "scholarships",
            "financial", "aid", "loan", "ptptn", "hostel", "accommodation",
            "campus", "campuses", "programme", "program", "programmes", "programs",
            "course", "courses", "intake", "student", "students", "foundation",
            "diploma", "degree", "bachelor", "master", "phd", "faculty",
            "faculties", "merit", "result", "results", "transcript", "certificate",
            "orientation", "freshmen", "freshman", "new", "international",
            "local", "study", "studies", "location", "branch", "branches",
            "facility", "facilities", "library", "lab", "labs", "computer",
            "wifi", "internet", "sports", "gym", "canteen", "cafeteria",
            "classroom", "hall", "available", "provided", "provide", "amenities",
            "requirement", "requirements", "entry", "eligibility", "qualify",
            "qualification", "qualifications",
            "deadline", "closing", "close", "date", "last", "latest"
        }

        self.synonym_map = {
            "uni": "university",
            "varsity": "university",
            "college": "university",
            "tar uc": "tarumt",
            "tarc": "tarumt",
            "tunku abdul rahman university": "tarumt",
            "tunku abdul rahman university of management and technology": "tarumt",

            "program": "programme",
            "programs": "programmes",
            "prog": "programme",

            "fees": "fee",
            "payments": "payment",

            "hostel": "accommodation",
            "dorm": "accommodation",
            "dormitory": "accommodation",

            "branch": "campus",
            "branches": "campuses",
            "location": "campus",
            "locations": "campuses",

            "intakes": "intake",
            "subjects": "programme",
            "degree course": "degree",
            "diploma course": "diploma",

            "facility": "facilities",
            "amenities": "facilities",
            "service": "facilities",
            "services": "facilities",
            "provided": "available",
            "provide": "available",
            "computer lab": "labs",
            "study area": "library",
            "food court": "canteen",

            "closing date": "deadline",
            "close date": "deadline",
            "last date": "deadline"
        }

        self.intent_patterns = {
            "programme": [
                r"\bprogramme\b", r"\bprogram\b", r"\bprogrammes\b", r"\bprograms\b"
            ],
            "course": [
                r"\bcourse\b", r"\bcourses\b", r"\bfield of study\b", r"\bsubjects\b"
            ],
            "campus": [
                r"\bcampus\b", r"\bcampuses\b", r"\bbranch\b", r"\bbranches\b",
                r"\blocation\b", r"\blocations\b", r"\bwhere.*located\b"
            ],
            "intake": [
                r"\bintake\b", r"\bintakes\b", r"\bwhen.*admission\b",
                r"\bwhen.*register\b", r"\bwhen.*apply\b"
            ],
            "admission": [
                r"\badmission\b", r"\bapply\b", r"\bapplication\b",
                r"\bhow to apply\b", r"\bapplication process\b", r"\bhow can i apply\b"
            ],
            "requirements": [
                r"\brequirement\b", r"\brequirements\b",
                r"\bentry requirement\b", r"\bentry requirements\b",
                r"\beligibility\b", r"\bqualification\b", r"\bqualifications\b",
                r"\bwhat.*need.*to enter\b", r"\bwhat.*need.*to apply\b"
            ],
            "document": [
                r"\bdocument\b", r"\bdocuments\b", r"\brequired document\b",
                r"\bwhat.*need.*submit\b", r"\bcertificate\b", r"\btranscript\b"
            ],
            "fee": [
                r"\bfee\b", r"\bfees\b", r"\bpayment\b", r"\btuition\b"
            ],
            "scholarship": [
                r"\bscholarship\b", r"\bscholarships\b", r"\bfinancial aid\b", r"\bptptn\b"
            ],
            "accommodation": [
                r"\bhostel\b", r"\baccommodation\b", r"\bstay\b", r"\bdorm\b"
            ],
            "facilities": [
                r"\bfacility\b", r"\bfacilities\b", r"\bamenities\b",
                r"\bwhat.*provided\b", r"\bwhat.*available\b",
                r"\blibrary\b", r"\blab\b", r"\blabs\b",
                r"\bcanteen\b", r"\bcafeteria\b", r"\bgym\b",
                r"\bsports\b", r"\bwifi\b", r"\bclassroom\b",
                r"\bhall\b", r"\bcomputer\b"
            ],
            "deadline": [
                r"\bdeadline\b", r"\bclosing date\b", r"\bclose date\b",
                r"\blast date\b", r"\bwhen.*deadline\b", r"\bwhen.*closing\b"
            ]
        }

        self.load_data()

    def normalize_text(self, text):
        text = str(text).lower().strip()
        text = text.replace("&", " and ")

        text = re.sub(r"\bwhat facility provided\b", "what facilities are provided", text)
        text = re.sub(r"\bwhat facility\b", "what facilities", text)
        text = re.sub(r"\bhow do i apply\b", "how to apply", text)
        text = re.sub(r"\bhow can i apply\b", "how to apply", text)
        text = re.sub(r"\bapplication process\b", "how to apply", text)

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

    def detect_intents(self, text):
        intents = set()
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    intents.add(intent)
                    break
        return intents

    def smart_query_expand(self, query):
        expanded = [query]

        if "programme" in query or "program" in query:
            expanded.append(query + " course diploma degree postgraduate faculty field of study")

        if "course" in query or "courses" in query:
            expanded.append(query + " programme faculty field of study")

        if "campus" in query or "campuses" in query:
            expanded.append(query + " location branch branches kuala lumpur penang perak johor pahang sabah")

        if "intake" in query:
            expanded.append(query + " admission application registration january may june september november")

        if "admission" in query:
            expanded.append(query + " apply application entry requirement online portal process")

        if "requirement" in query or "requirements" in query or "entry" in query:
            expanded.append(query + " entry requirement qualification qualifications eligibility spm uec stpm diploma degree")

        if "document" in query:
            expanded.append(query + " required submit certificate transcript result")

        if "accommodation" in query:
            expanded.append(query + " hostel stay room dorm")

        if "facilities" in query or "facility" in query:
            expanded.append(query + " library labs sports canteen wifi classroom hall computer student facilities")

        if "deadline" in query or "closing" in query or "last" in query:
            expanded.append(query + " application deadline closing date intake admission last date apply january may june september november")

        return " ".join(expanded)

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
                self.df["clean_question"] + " " +
                self.df["clean_context"] + " " +
                self.df["clean_answer"]
            )

            self.word_vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 3),
                sublinear_tf=True,
                min_df=1
            )

            self.char_vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                sublinear_tf=True,
                min_df=1
            )

            self.word_matrix = self.word_vectorizer.fit_transform(self.df["combined_text"])
            self.char_matrix = self.char_vectorizer.fit_transform(self.df["combined_text"])

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
            "admission, entry requirements, required documents, fees, scholarships, "
            "hostel, facilities, registration, programmes, campuses, intake, and deadlines."
        )

    def fallback_no_match_message(self):
        return (
            "Sorry, I could not find a confident answer in the TAR UMT FAQ dataset. "
            "Please try asking in a more specific way."
        )

    def keyword_overlap_score(self, query, text):
        q_words = set(query.split())
        t_words = set(text.split())

        if not q_words or not t_words:
            return 0.0

        overlap = q_words.intersection(t_words)
        return len(overlap) / max(len(q_words), 1)

    def intent_boost_score(self, query_intents, row_text):
        row_intents = self.detect_intents(row_text)
        if not query_intents or not row_intents:
            return 0.0
        return len(query_intents.intersection(row_intents)) / max(len(query_intents), 1)

    def get_filtered_indices(self, clean_query):
        query_intents = self.detect_intents(clean_query)
        filtered_indices = list(range(len(self.df)))

        if "deadline" in query_intents:
            deadline_words = [
                "deadline", "closing", "date", "last", "intake",
                "application", "admission", "january", "may", "june",
                "september", "november"
            ]
            filtered_indices = [
                i for i in range(len(self.df))
                if any(word in self.df.iloc[i]["combined_text"] for word in deadline_words)
            ]

        elif "requirements" in query_intents:
            requirement_words = [
                "requirement", "requirements", "entry", "qualification",
                "qualifications", "eligibility", "spm", "uec", "stpm"
            ]
            filtered_indices = [
                i for i in range(len(self.df))
                if any(word in self.df.iloc[i]["combined_text"] for word in requirement_words)
            ]

        elif "facilities" in query_intents:
            facility_words = [
                "facility", "facilities", "library", "lab", "labs", "canteen",
                "cafeteria", "sports", "wifi", "classroom", "hall", "computer"
            ]
            filtered_indices = [
                i for i in range(len(self.df))
                if any(word in self.df.iloc[i]["combined_text"] for word in facility_words)
            ]

        elif "accommodation" in query_intents:
            accommodation_words = ["accommodation", "hostel", "room", "stay", "dorm"]
            filtered_indices = [
                i for i in range(len(self.df))
                if any(word in self.df.iloc[i]["combined_text"] for word in accommodation_words)
            ]

        elif "admission" in query_intents:
            admission_words = ["apply", "application", "admission", "portal", "register", "process"]
            filtered_indices = [
                i for i in range(len(self.df))
                if any(word in self.df.iloc[i]["combined_text"] for word in admission_words)
            ]

        elif "document" in query_intents:
            document_words = ["document", "documents", "certificate", "transcript", "submit"]
            filtered_indices = [
                i for i in range(len(self.df))
                if any(word in self.df.iloc[i]["combined_text"] for word in document_words)
            ]

        elif "campus" in query_intents:
            campus_words = [
                "campus", "campuses", "location", "branch", "kuala lumpur",
                "penang", "perak", "johor", "pahang", "sabah"
            ]
            filtered_indices = [
                i for i in range(len(self.df))
                if any(word in self.df.iloc[i]["combined_text"] for word in campus_words)
            ]

        elif "programme" in query_intents or "course" in query_intents:
            programme_words = [
                "programme", "programmes", "program", "course", "courses",
                "diploma", "degree", "postgraduate", "faculty"
            ]
            filtered_indices = [
                i for i in range(len(self.df))
                if any(word in self.df.iloc[i]["combined_text"] for word in programme_words)
            ]

        elif "intake" in query_intents:
            intake_words = ["intake", "january", "may", "june", "september", "november"]
            filtered_indices = [
                i for i in range(len(self.df))
                if any(word in self.df.iloc[i]["combined_text"] for word in intake_words)
            ]

        if not filtered_indices:
            filtered_indices = list(range(len(self.df)))

        return filtered_indices

    def retrieve_candidates(self, clean_query):
        expanded_query = self.smart_query_expand(clean_query)

        word_vector = self.word_vectorizer.transform([expanded_query])
        char_vector = self.char_vectorizer.transform([expanded_query])

        word_scores = cosine_similarity(word_vector, self.word_matrix).flatten()
        char_scores = cosine_similarity(char_vector, self.char_matrix).flatten()

        query_intents = self.detect_intents(clean_query)
        filtered_indices = self.get_filtered_indices(clean_query)

        scored_candidates = []

        for i in filtered_indices:
            row = self.df.iloc[i]

            overlap_q = self.keyword_overlap_score(clean_query, row["clean_question"])
            overlap_c = self.keyword_overlap_score(clean_query, row["clean_context"])
            overlap_a = self.keyword_overlap_score(clean_query, row["clean_answer"])
            overlap = max(overlap_q, overlap_c, overlap_a)

            intent_boost = self.intent_boost_score(query_intents, row["combined_text"])

            final_score = (
                0.60 * word_scores[i] +
                0.25 * char_scores[i] +
                0.10 * overlap +
                0.05 * intent_boost
            )

            scored_candidates.append((i, final_score, word_scores[i], char_scores[i], overlap, intent_boost))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:self.top_k]

    def choose_best_answer(self, candidates):
        if not candidates:
            return None, 0.0, []

        best = candidates[0]
        return best[0], best[1], candidates

    def get_response(self, user_query):
        clean_query = self.preprocess_text(user_query)

        if not clean_query:
            return {
                "retrieved_question": None,
                "retrieved_context": None,
                "answer": "Please enter a valid TAR UMT-related question.",
                "score": 0.0,
                "top_matches": []
            }

        if not self.is_in_scope(clean_query):
            return {
                "retrieved_question": None,
                "retrieved_context": None,
                "answer": self.fallback_scope_message(),
                "score": 0.0,
                "top_matches": []
            }

        candidates = self.retrieve_candidates(clean_query)
        best_index, best_score, top_candidates = self.choose_best_answer(candidates)

        if best_index is None or best_score < self.similarity_threshold:
            return {
                "retrieved_question": None,
                "retrieved_context": None,
                "answer": self.fallback_no_match_message(),
                "score": float(best_score if best_index is not None else 0.0),
                "top_matches": []
            }

        best_row = self.df.iloc[best_index]

        if best_score >= 0.60:
            confidence = "High"
        elif best_score >= 0.35:
            confidence = "Medium"
        else:
            confidence = "Low"

        top_matches = []
        used_questions = set()

        for idx, final_score, word_s, char_s, overlap_s, intent_s in top_candidates:
            row = self.df.iloc[idx]
            q = row["question"].strip()

            if q in used_questions:
                continue
            used_questions.add(q)

            top_matches.append({
                "question": q,
                "score": round(float(final_score), 4),
                "word_score": round(float(word_s), 4),
                "char_score": round(float(char_s), 4),
                "overlap_score": round(float(overlap_s), 4),
                "intent_score": round(float(intent_s), 4)
            })

        return {
            "retrieved_question": best_row["question"],
            "retrieved_context": best_row["context"],
            "answer": best_row["answer"].strip(),
            "score": float(best_score),
            "confidence": confidence,
            "top_matches": top_matches
        }

    def display_result(self, result):
        print("\n" + "=" * 80)
        print("Chatbot Type       : Enhanced Template-Based RAG")
        print(f"Similarity Score   : {result['score']:.4f}")

        if "confidence" in result:
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
                print(
                    f"{i}. {item['question']} "
                    f"(final={item['score']}, word={item['word_score']}, "
                    f"char={item['char_score']}, overlap={item['overlap_score']}, "
                    f"intent={item['intent_score']})"
                )

        print("=" * 80 + "\n")

    def run(self):
        print("Enhanced TAR UMT Template RAG Chatbot")
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

    chatbot = TemplateRAGChatbot(
        csv_file=csv_path,
        similarity_threshold=0.22,
        top_k=5
    )
    chatbot.run()