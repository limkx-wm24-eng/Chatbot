import re
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support
import webbrowser
from PIL import Image, ImageTk

# Class TF-IDF-based RAG Chatbot for TARUMT FAQ dataset
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

    def normalize_text(self, text):
        text = str(text).lower().strip()

        phrase_map = {
            "minimum requirement": "entry requirements",
            "minimal requirement": "entry requirements",
            "requirements to apply": "entry requirements",
            "requirement to apply": "entry requirements",
            "requirements to entry": "entry requirements",
            "requirement to entry": "entry requirements",

            "where is the campus": "campus location",
            "where is campus": "campus location",

            "how to apply": "admission process",
            "how do i apply": "admission process",
            "how can i apply": "admission process",
            "where to apply": "admission process",
            "application process": "admission process",

            "how to make the application": "admission process",
            "how to make application": "admission process",
            "make application": "admission process",
            "make an application": "admission process",

            "when can i enroll": "intake",
            "when can i enrol": "intake",
            "what month can i enroll": "intake",
            "what month can i enrol": "intake",
            "month to enroll": "intake",
            "month to enrol": "intake",
            "what is the month that i can enroll": "intake",
            "what is the month that i can enrol": "intake",

            "register the course": "entry requirements",
            "requirement needed to register the course": "entry requirements",

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

    def load_data(self):
        self.df = pd.read_csv(self.csv_file).fillna("")

        self.df["combined_text"] = (
            self.df["question"].astype(str) + " " +
            self.df["context"].astype(str) + " " +
            self.df["answer"].astype(str)
        )

        self.df["clean_text"] = self.df["combined_text"].apply(self.normalize_text)
        self.build_index()

    def build_index(self):
        self.word_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.word_matrix = self.word_vectorizer.fit_transform(self.df["clean_text"])

        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.char_matrix = self.char_vectorizer.fit_transform(self.df["clean_text"])

    def predict_intent(self, query):
        q = self.normalize_text(query)

        if "intake" in q:
            return "intake_info"
        if "admission process" in q or ("apply" in q and "fee" not in q):
            return "admission_apply"
        if ("campus" in q and "location" in q) or ("where" in q and "campus" in q):
            return "location_info"
        if "course" in q and "requirements" not in q:
            return "course_list"
        if "programme" in q or "diploma" in q or "degree" in q or "postgraduate" in q:
            return "programme_list"
        if "entry" in q or "requirements" in q:
            return "requirements_info"
        if "scholarship" in q or "loan" in q or "financial aid" in q or "ptptn" in q:
            return "scholarship_info"
        if "hostel" in q or "accommodation" in q or "room" in q:
            return "hostel_info"
        if "fee" in q or "payment" in q or "cost" in q or "price" in q or "tuition" in q:
            return "fees_detail"
        if "library" in q or "wifi" in q or "facility" in q or "lab" in q or "canteen" in q or "sports" in q:
            return "facilities_info"
        if "document" in q or "certificate" in q or "transcript" in q:
            return "document_info"
        if "deadline" in q or "closing date" in q or "last date" in q:
            return "deadline_info"
        if "contact" in q or "phone" in q or "email" in q or "number" in q:
            return "contact_info"
        if "transport" in q or "bus" in q or "lrt" in q or "train" in q:
            return "transport_info"

        return "None"

    def rule_based_answer(self, query):
        q = self.normalize_text(query)

        if "intake" in q:
            return "TARUMT intakes are January, May/June, September and November."

        if "admission process" in q or ("apply" in q and "fee" not in q):
            return "You can apply through the TARUMT admission portal via the official website by selecting your programme and submitting the required documents."

        if ("campus" in q and "location" in q) or ("where" in q and "campus" in q):
            return "TARUMT has 6 campuses located in Kuala Lumpur, Penang, Perak, Johor, Pahang and Sabah."

        if "course" in q and "requirements" not in q:
            return "TARUMT offers courses in Engineering, Information Technology, Business and Accounting."

        if "programme" in q or "diploma" in q or "degree" in q or "postgraduate" in q:
            return "TARUMT offers Diploma, Degree and Postgraduate programmes."

        if "entry" in q or "requirements" in q:
            return "Entry requirements depend on the programme level and relevant academic qualifications such as SPM, UEC, STPM or equivalent."

        if "facility" in q or "library" in q or "wifi" in q or "lab" in q or "canteen" in q or "sports" in q:
            return "TARUMT provides facilities such as libraries, computer labs, WiFi access, canteen services and sports facilities."

        if "hostel" in q or "accommodation" in q:
            return "Hostel or accommodation availability may depend on campus. Students should check the official TARUMT website or contact the relevant campus for details."

        if "fee" in q or "payment" in q or "cost" in q or "tuition" in q:
            return "Fees depend on the programme and level of study. Please refer to the official TARUMT website for the latest fee details."

        if "scholarship" in q or "loan" in q or "ptptn" in q or "financial aid" in q:
            return "Students may check available scholarships, financial aid, or PTPTN options through TARUMT's official website or the student support office."

        if "contact" in q or "phone" in q or "email" in q or "number" in q:
            return "You can contact TARUMT through the official website for campus phone numbers, email addresses, and enquiry channels."

        if "transport" in q or "bus" in q or "lrt" in q or "train" in q:
            return "Students can check available public transport options and campus accessibility details through the official TARUMT website or campus office."

        return None

    def tfidf_fallback(self, user_query):
        query = self.normalize_text(user_query)

        word_vec = self.word_vectorizer.transform([query])
        char_vec = self.char_vectorizer.transform([query])

        scores = (
            0.6 * cosine_similarity(word_vec, self.word_matrix).flatten() +
            0.4 * cosine_similarity(char_vec, self.char_matrix).flatten()
        )

        best_idx = scores.argmax()
        best_score = scores[best_idx]

        if best_score < 0.15:
            return "Sorry, I could not find a relevant answer.", float(best_score)

        row = self.df.iloc[best_idx]
        return row["context"] + " " + row["answer"], float(best_score)

    def get_response(self, user_query):
        predicted_intent = self.predict_intent(user_query)
        rule = self.rule_based_answer(user_query)

        if rule:
            return {
                "answer": rule,
                "score": 1.0,
                "predicted_intent": predicted_intent
            }

        answer, score = self.tfidf_fallback(user_query)
        return {
            "answer": answer,
            "score": score,
            "predicted_intent": predicted_intent
        }


class TARUMTChatbotGUI:
    def __init__(self, root, chatbot):
        # Load icons
        try:
            bot_img = Image.open("bot.png").resize((32, 32))
            user_img = Image.open("user.png").resize((32, 32))

            self.bot_icon = ImageTk.PhotoImage(bot_img)
            self.user_icon = ImageTk.PhotoImage(user_img)
        except:
            self.bot_icon = None
            self.user_icon = None

        self.root = root
        self.chatbot = chatbot

        self.root.title("TARUMT FAQ Chatbot")
        self.root.geometry("1250x780")
        self.root.configure(bg="#eef3fb")
        self.root.minsize(1080, 700)

        self.primary = "#0d47a1"
        self.primary2 = "#1565c0"
        self.bg = "#eef3fb"
        self.sidebar_bg = "#dfe8f7"
        self.card_bg = "#ffffff"
        self.chat_bg = "#f7f9fc"
        self.user_bubble = "#dbeafe"
        self.bot_bubble = "#eef2f7"
        self.text_dark = "#1f2d3d"
        self.text_muted = "#6b7280"
        self.success = "#2e7d32"
        self.danger = "#e05a63"
        self.exit_bg = "#6b7280"
        self.input_bg = "#dce3ec"

        self.logo = None
        self.placeholder = "Type your question here..."
        self.typing_bubble = None
        self.typing_job = None
        self.typing_step = 0

        self.build_gui()

    def build_gui(self):
        self.build_header()
        self.build_main()
        self.build_footer()

        self.add_bot_message(
            "Hello. Welcome to the TARUMT FAQ Chatbot.\n"
            "Ask me about intake, programmes, courses, campus, admission, fees, hostel, and requirements."
        )

    def add_hover(self, button, normal_color, hover_color):
        def on_enter(event):
            button.config(bg=hover_color)

        def on_leave(event):
            button.config(bg=normal_color)

        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)

    def build_header(self):
        header = tk.Frame(self.root, bg=self.primary, height=110)
        header.pack(fill="x")
        header.pack_propagate(False)

        left = tk.Frame(header, bg=self.primary)
        left.pack(side="left", padx=20, pady=16)

        try:
            logo_img = Image.open("tarumt_logo.jpg")
            logo_img.thumbnail((420, 80))
            self.logo = ImageTk.PhotoImage(logo_img)

            tk.Label(
                left,
                image=self.logo,
                bg=self.primary
            ).pack(side="left", padx=(0, 18))
        except Exception:
            title_wrap = tk.Frame(left, bg=self.primary)
            title_wrap.pack(side="left")
            tk.Label(
                title_wrap,
                text="TARUMT FAQ Chatbot",
                font=("Arial", 28, "bold"),
                bg=self.primary,
                fg="white"
            ).pack(anchor="w")
            tk.Label(
                title_wrap,
                text="University Information Assistant",
                font=("Arial", 12),
                bg=self.primary,
                fg="#dbeafe"
            ).pack(anchor="w", pady=(4, 0))
        else:
            text_wrap = tk.Frame(left, bg=self.primary)
            text_wrap.pack(side="left", padx=(0, 8))
            tk.Label(
                text_wrap,
                text="TARUMT FAQ Chatbot",
                font=("Arial", 24, "bold"),
                bg=self.primary,
                fg="white"
            ).pack(anchor="w")
            tk.Label(
                text_wrap,
                text="University Information Assistant",
                font=("Arial", 12),
                bg=self.primary,
                fg="#dbeafe"
            ).pack(anchor="w", pady=(4, 0))

        website_btn = tk.Button(
            header,
            text="Open TARUMT Website",
            font=("Arial", 11, "bold"),
            bg="white",
            fg=self.primary,
            relief="flat",
            bd=0,
            padx=18,
            pady=12,
            cursor="hand2",
            command=lambda: webbrowser.open("https://www.tarc.edu.my/")
        )
        website_btn.pack(side="right", padx=26)
        self.add_hover(website_btn, "white", "#dbeafe")

    def build_main(self):
        main = tk.Frame(self.root, bg=self.bg)
        main.pack(fill="both", expand=True, padx=18, pady=18)

        self.build_sidebar(main)
        self.build_chat_panel(main)

    def build_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=self.sidebar_bg, width=320)
        sidebar.pack(side="left", fill="y", padx=(0, 14))
        sidebar.pack_propagate(False)

        # Title
        tk.Label(
            sidebar,
            text="Quick Questions",
            font=("Arial", 18, "bold"),
            bg=self.sidebar_bg,
            fg=self.primary
        ).pack(anchor="w", padx=18, pady=(20, 12))

        # ===== Top scroll area container =====
        scroll_area = tk.Frame(sidebar, bg=self.sidebar_bg)
        scroll_area.pack(fill="both", expand=True, padx=10)

        canvas = tk.Canvas(scroll_area, bg=self.sidebar_bg, highlightthickness=0)
        scrollbar = tk.Scrollbar(scroll_area, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=self.sidebar_bg)

        canvas_window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)

        scroll_frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        quick_questions = [
            "What are the intake months?",
            "What courses are offered?",
            "What programmes are offered?",
            "Where are the campuses located?",
            "How do I apply?",
            "What are the entry requirements?",
            "What facilities are available?",
            "Is there hostel accommodation?"
        ]

        for q in quick_questions:
            btn = tk.Button(
                scroll_frame,
                text=q,
                font=("Arial", 11),
                bg="white",
                fg="#334155",
                wraplength=260,
                justify="left",
                anchor="w",
                relief="flat",
                bd=0,
                padx=16,
                pady=12,
                cursor="hand2",
                command=lambda question=q: self.ask_quick_question(question)
            )
            btn.pack(fill="x", pady=6)
            self.add_hover(btn, "white", "#e0e7ff")

        # Mouse wheel only for sidebar canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        # ===== Bottom action area =====
        bottom_area = tk.Frame(sidebar, bg=self.sidebar_bg)
        bottom_area.pack(fill="x", padx=16, pady=(10, 12))

        tk.Frame(bottom_area, bg="#cbd5e1", height=1).pack(fill="x", pady=(0, 10))

        eval_btn = tk.Button(
            bottom_area,
            text="Run Evaluation",
            font=("Arial", 10, "bold"),
            bg="#0f766e",
            fg="white",
            relief="flat",
            bd=0,
            padx=10,
            pady=8,
            cursor="hand2",
            command=self.run_evaluation_popup
        )
        eval_btn.pack(fill="x", pady=(0, 8))
        self.add_hover(eval_btn, "#0f766e", "#0d9488")

        clear_btn = tk.Button(
            bottom_area,
            text="Clear Chat",
            font=("Arial", 10, "bold"),
            bg=self.danger,
            fg="white",
            relief="flat",
            bd=0,
            padx=10,
            pady=8,
            cursor="hand2",
            command=self.clear_chat
        )
        clear_btn.pack(fill="x")
        self.add_hover(clear_btn, self.danger, "#c9444e")

    def build_chat_panel(self, parent):
        panel = tk.Frame(parent, bg=self.card_bg)
        panel.pack(side="left", fill="both", expand=True)

        top_bar = tk.Frame(panel, bg=self.card_bg, height=58)
        top_bar.pack(fill="x")
        top_bar.pack_propagate(False)

        tk.Label(
            top_bar,
            text="Conversation",
            font=("Arial", 18, "bold"),
            bg=self.card_bg,
            fg=self.text_dark
        ).pack(side="left", padx=22, pady=14)

        self.status_label = tk.Label(
            top_bar,
            text="Ready",
            font=("Arial", 11),
            bg=self.card_bg,
            fg=self.success
        )
        self.status_label.pack(side="right", padx=18)

        divider = tk.Frame(panel, bg="#e5e7eb", height=1)
        divider.pack(fill="x")

        self.canvas = tk.Canvas(panel, bg=self.chat_bg, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(panel, orient="vertical", command=self.canvas.yview)
        self.chat_container = tk.Frame(self.canvas, bg=self.chat_bg)

        self.chat_container.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.chat_container, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind("<Configure>", self.resize_chat_width)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self.on_mousewheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def build_footer(self):
        footer = tk.Frame(self.root, bg=self.bg)
        footer.pack(fill="x", padx=18, pady=(0, 18))

        input_wrap = tk.Frame(footer, bg=self.input_bg, bd=0, relief="flat")
        input_wrap.pack(side="left", fill="x", expand=True, padx=(0, 12))

        tk.Label(
            input_wrap,
            text="⌕",
            font=("Arial", 22),
            bg=self.input_bg,
            fg="#475569"
        ).pack(side="left", padx=(14, 8))

        self.user_input = tk.Entry(
            input_wrap,
            font=("Arial", 13),
            relief="flat",
            bd=0,
            bg=self.input_bg,
            fg="#6b7280",
            insertbackground="#111827"
        )
        self.user_input.pack(side="left", fill="x", expand=True, ipady=16, padx=(0, 10))
        self.user_input.insert(0, self.placeholder)

        self.user_input.bind("<Return>", lambda event: self.send_message())
        self.user_input.bind("<FocusIn>", self.on_entry_focus_in)
        self.user_input.bind("<FocusOut>", self.on_entry_focus_out)

        send_btn = tk.Button(
            footer,
            text="Send",
            font=("Arial", 11, "bold"),
            bg=self.primary2,
            fg="white",
            relief="flat",
            bd=0,
            padx=24,
            pady=16,
            cursor="hand2",
            command=self.send_message
        )
        send_btn.pack(side="left", padx=(0, 10))
        self.add_hover(send_btn, self.primary2, "#1e40af")

        exit_btn = tk.Button(
            footer,
            text="Exit",
            font=("Arial", 11, "bold"),
            bg=self.exit_bg,
            fg="white",
            relief="flat",
            bd=0,
            padx=24,
            pady=16,
            cursor="hand2",
            command=self.root.quit
        )
        exit_btn.pack(side="left")
        self.add_hover(exit_btn, self.exit_bg, "#374151")

    def on_entry_focus_in(self, event):
        if self.user_input.get() == self.placeholder:
            self.user_input.delete(0, tk.END)
            self.user_input.config(fg="#111827")

    def on_entry_focus_out(self, event):
        if self.user_input.get().strip() == "":
            self.user_input.delete(0, tk.END)
            self.user_input.insert(0, self.placeholder)
            self.user_input.config(fg="#6b7280")

    def resize_chat_width(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def on_mousewheel(self, event):
        try:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass

    def rounded_rectangle_points(self, x1, y1, x2, y2, r=20):
        return [
            x1 + r, y1,
            x2 - r, y1,
            x2, y1,
            x2, y1 + r,
            x2, y2 - r,
            x2, y2,
            x2 - r, y2,
            x1 + r, y2,
            x1, y2,
            x1, y2 - r,
            x1, y1 + r,
            x1, y1
        ]

    def create_message_card(self, sender, message, meta=None, is_user=False):
        outer = tk.Frame(self.chat_container, bg=self.chat_bg)
        outer.pack(fill="x", padx=18, pady=8)

        if is_user:
            bubble_color = self.user_bubble
            text_color = "#1e3a8a"
            anchor = "e"
            padx = (220, 20)
        else:
            bubble_color = self.bot_bubble
            text_color = "#065f46"
            anchor = "w"
            padx = (20, 220)

        bubble_canvas = tk.Canvas(
            outer,
            bg=self.chat_bg,
            highlightthickness=0,
            bd=0
        )

        bubble_frame = tk.Frame(bubble_canvas, bg=bubble_color, bd=0)
        window_id = bubble_canvas.create_window(12, 10, window=bubble_frame, anchor="nw")

        header_row = tk.Frame(bubble_frame, bg=bubble_color)

        header_row.pack(anchor="w", padx=16, pady=(12, 4))

        icon = self.user_icon if is_user else self.bot_icon

        if icon is not None:
            tk.Label(
                header_row,
                image=icon,
                bg=bubble_color
            ).pack(side="left", padx=(0, 6))

        tk.Label(
            header_row,
            text=sender,
            font=("Arial", 11, "bold"),
            bg=bubble_color,
            fg=text_color
        ).pack(side="left")

        tk.Label(
            bubble_frame,
            text=message,
            font=("Arial", 11),
            bg=bubble_color,
            fg="#111827",
            wraplength=480,
            justify="left"
        ).pack(anchor="w", padx=16, pady=(0, 6))

        if meta:
            tk.Label(
                bubble_frame,
                text=meta,
                font=("Arial", 9, "italic"),
                bg=bubble_color,
                fg="#6b7280",
                wraplength=520,
                justify="left"
            ).pack(anchor="w", padx=16, pady=(0, 12))
        else:
            tk.Label(
                bubble_frame,
                text="",
                bg=bubble_color
            ).pack(anchor="w", padx=16, pady=(0, 10))

        bubble_frame.update_idletasks()

        width = bubble_frame.winfo_reqwidth() + 20
        height = bubble_frame.winfo_reqheight() + 20

        points = self.rounded_rectangle_points(2, 2, width, height, 20)

        bubble_canvas.create_polygon(
            points,
            smooth=True,
            fill=bubble_color,
            outline=bubble_color
        )

        bubble_canvas.coords(window_id, 12, 10)
        bubble_canvas.config(width=width + 4, height=height + 4)

        bubble_canvas.pack(anchor=anchor, padx=padx)

        self.root.after(100, lambda: self.canvas.yview_moveto(1.0))

    def show_typing_indicator(self):
        self.hide_typing_indicator()

        outer = tk.Frame(self.chat_container, bg=self.chat_bg)
        outer.pack(fill="x", padx=18, pady=8)

        bubble_color = self.bot_bubble
        bubble_canvas = tk.Canvas(
            outer,
            bg=self.chat_bg,
            highlightthickness=0,
            bd=0
        )

        bubble_frame = tk.Frame(bubble_canvas, bg=bubble_color, bd=0)
        window_id = bubble_canvas.create_window(12, 10, window=bubble_frame, anchor="nw")

        tk.Label(
            bubble_frame,
            text="Bot",
            font=("Arial", 11, "bold"),
            bg=bubble_color,
            fg="#065f46"
        ).pack(anchor="w", padx=16, pady=(12, 4))

        typing_label = tk.Label(
            bubble_frame,
            text="Typing.",
            font=("Arial", 11, "italic"),
            bg=bubble_color,
            fg="#6b7280"
        )
        typing_label.pack(anchor="w", padx=16, pady=(0, 12))

        bubble_frame.update_idletasks()

        width = bubble_frame.winfo_reqwidth() + 20
        height = bubble_frame.winfo_reqheight() + 20

        points = self.rounded_rectangle_points(2, 2, width, height, 20)

        bubble_canvas.create_polygon(
            points,
            smooth=True,
            fill=bubble_color,
            outline=bubble_color
        )

        bubble_canvas.coords(window_id, 12, 10)
        bubble_canvas.config(width=width + 4, height=height + 4)
        bubble_canvas.pack(anchor="w", padx=(20, 220))

        self.typing_bubble = {
            "outer": outer,
            "label": typing_label
        }
        self.typing_step = 0
        self.animate_typing()
        self.root.after(100, lambda: self.canvas.yview_moveto(1.0))

    def animate_typing(self):
        if not self.typing_bubble:
            return

        states = ["Typing.", "Typing..", "Typing..."]
        self.typing_bubble["label"].config(text=states[self.typing_step % len(states)])
        self.typing_step += 1
        self.typing_job = self.root.after(400, self.animate_typing)

    def hide_typing_indicator(self):
        if self.typing_job:
            self.root.after_cancel(self.typing_job)
            self.typing_job = None

        if self.typing_bubble:
            self.typing_bubble["outer"].destroy()
            self.typing_bubble = None

    def add_user_message(self, message):
        self.create_message_card("You", message, is_user=True)

    def add_bot_message(self, message, intent=None, score=None):
        meta = None
        if intent is not None and score is not None:
            meta = f"Predicted Intent: {intent} | Score: {score}"
        self.create_message_card("Bot", message, meta=meta, is_user=False)

    def add_system_message(self, message):
        self.create_message_card("System", message, is_user=False)

    def process_bot_response(self, user_text):
        result = self.chatbot.get_response(user_text)
        answer = result["answer"]
        intent = result["predicted_intent"]
        score = round(result["score"], 4)

        self.hide_typing_indicator()
        self.add_bot_message(answer, intent, score)
        self.status_label.config(text="Answered")

        self.user_input.delete(0, tk.END)
        self.user_input.insert(0, self.placeholder)
        self.user_input.config(fg="#6b7280")

    def send_message(self):
        user_text = self.user_input.get().strip()

        if user_text == self.placeholder:
            user_text = ""

        if not user_text:
            messagebox.showwarning("Warning", "Please enter a question.")
            return

        self.add_user_message(user_text)
        self.status_label.config(text="Bot is typing...")
        self.show_typing_indicator()

        self.root.after(900, lambda: self.process_bot_response(user_text))

    def ask_quick_question(self, question):
        self.user_input.delete(0, tk.END)
        self.user_input.insert(0, question)
        self.user_input.config(fg="#111827")
        self.send_message()

    def clear_chat(self):
        self.hide_typing_indicator()

        for widget in self.chat_container.winfo_children():
            widget.destroy()

        self.status_label.config(text="Ready")

        self.add_system_message("Chat has been cleared.")


def main():
    try:
        chatbot = TFIDFRAGChatbot("\dataset\tarumt_faq_dataset.csv")
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", f"Failed to load dataset:\n{e}")
        return

    root = tk.Tk()
    app = TARUMTChatbotGUI(root, chatbot)
    root.mainloop()


if __name__ == "__main__":
    main()