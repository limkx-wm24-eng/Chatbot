import re
import string

def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text

# Rule patterns
rules = {
    "admission": [
        r"\bapply\b", r"\badmission\b", r"\benroll\b", r"\bentry\b",
        r"\bregister\b", r"\bnew student\b", r"\bapplication\b",
        r"\benrollment\b"
    ],
    "fees": [
        r"\bfee\b", r"\bfees\b", r"\btuition\b", r"\bpayment\b", r"\bbalance\b",
        r"\binstallment\b", r"\btransfer\b", r"\bpay\b", r"\breceipt\b"
    ],
    "courses": [
        r"\bcourse\b", r"\bcourses\b", r"\bprogramme\b", r"\bprogram\b",
        r"\bdiploma\b", r"\bdegree\b", r"\bcomputer science\b",
        r"\binformation technology\b", r"\bbusiness\b", r"\bsubjects\b"
    ],
    "timetable": [
        r"\btimetable\b", r"\bschedule\b", r"\bclass schedule\b",
        r"\blecture\b", r"\bclass timing\b", r"\bclass time\b"
    ],
    "contact": [
        r"\bcontact\b", r"\bphone\b", r"\bemail\b", r"\bhotline\b",
        r"\benquiries\b", r"\bstudent services\b", r"\bcall\b",
        r"\breach\b", r"\bget in touch\b"
    ],
    "greeting": [
        r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bgood morning\b",
        r"\bgood afternoon\b", r"\bgood evening\b", r"\bgreetings\b"
    ],
    "thanks": [
        r"\bthank\b", r"\bthanks\b", r"\bappreciate\b", r"\bthank you\b"
    ],
    "goodbye": [
        r"\bbye\b", r"\bgoodbye\b", r"\bsee you\b", r"\bcatch you later\b",
        r"\btalk to you later\b", r"\bfarewell\b"
    ]
}

responses = {
    "admission": "You can apply through the university admission portal and submit the required documents.",
    "fees": "You can check tuition fees and payment deadlines through the finance office or student portal.",
    "courses": "The university offers programmes such as IT, Business, and other diploma or degree courses.",
    "timetable": "You can view your class timetable through the student portal.",
    "contact": "You can contact the university through the official email or phone number listed on the website.",
    "greeting": "Hello. How can I help you today?",
    "thanks": "You are welcome.",
    "goodbye": "Goodbye. Have a nice day.",
    "unknown": "Sorry, I can only answer questions about admission, fees, courses, timetable, contact, greetings, thanks, and goodbye."
}

def predict_intent(user_input: str):
    text = clean_text(user_input)
    scores = {}

    for intent, patterns in rules.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, text):
                score += 1
        if score > 0:
            scores[intent] = score

    if not scores:
        return "unknown"

    # Priority order (most important first)
    priority_order = [
        "admission", "fees", "courses",
        "timetable", "contact",
        "greeting", "thanks", "goodbye"
    ]

    # Pick best based on priority + score
    best_intent = None
    best_score = -1

    for intent in priority_order:
        score = scores.get(intent, 0)
    if score > best_score:
        best_score = score
        best_intent = intent

    if best_score == 0:
        return "unknown"

    return best_intent

print("\nUniversity FAQ Chatbot (Rule-Based)")
while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["exit", "quit", "goodbye"]:
        print("Bot: Goodbye.")
        break

    intent = predict_intent(user_input)

    print("Predicted intent:", intent)
    print("Bot:", responses.get(intent, responses["unknown"]))