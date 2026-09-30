"""
FAQ Chatbot
-----------
Pipeline: collect FAQs -> preprocess (NLTK) -> TF-IDF vectors -> cosine similarity -> best answer.

Install:  pip install nltk scikit-learn flask
Run:      python faq_chatbot.py          (terminal chat)
          python faq_chatbot.py --web    (browser chat UI at http://127.0.0.1:5000)
"""
import re
import sys

import nltk
from nltk.tokenize import wordpunct_tokenize          # needs no downloaded data
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------- 1. FAQ data
# Topic: a fictional note-taking app. Replace with your own product's FAQs.
# Each FAQ has several ways of asking the same question, which improves matching.
FAQS = [
    {"questions": ["How do I create an account?", "How can I sign up?", "Register a new account"],
     "answer": "Tap Sign Up on the home screen, enter your email and a password, then confirm the link we email you."},
    {"questions": ["I forgot my password. How do I reset it?", "Reset password", "Can't log in to my account"],
     "answer": "Choose Forgot Password on the login screen. We'll email you a reset link that works for 30 minutes."},
    {"questions": ["Is CloudNotes free?", "How much does it cost?", "What are the pricing plans?"],
     "answer": "The Free plan includes 500 notes and 1 GB of storage. Pro costs $4 per month and adds unlimited notes and 50 GB."},
    {"questions": ["How do I cancel my subscription?", "Cancel my Pro plan", "Stop billing"],
     "answer": "Go to Settings > Billing > Cancel plan. You keep Pro features until the end of the billing period."},
    {"questions": ["Can I use CloudNotes offline?", "Does it work without internet?"],
     "answer": "Yes. Notes you opened recently are cached and sync automatically when you reconnect."},
    {"questions": ["Which devices are supported?", "Is there a mobile app?", "Does it work on Android and iPhone?"],
     "answer": "CloudNotes runs on Android, iOS, Windows, macOS and in any modern web browser."},
    {"questions": ["How do I export my notes?", "Download all my data", "Can I back up my notes?"],
     "answer": "Open Settings > Data > Export. You can download your notes as PDF, Markdown or a full ZIP backup."},
    {"questions": ["Are my notes secure?", "Is my data encrypted?", "How do you protect my privacy?"],
     "answer": "All notes are encrypted in transit and at rest. Enable two-factor authentication under Settings > Security for extra protection."},
    {"questions": ["How do I share a note with someone?", "Collaborate with a friend", "Share a notebook"],
     "answer": "Open the note, tap Share, and enter an email address. You can give view-only or edit access."},
    {"questions": ["How do I delete my account?", "Remove all my data permanently"],
     "answer": "Go to Settings > Account > Delete account. This permanently removes your notes after a 14-day grace period."},
    {"questions": ["How do I contact support?", "Talk to a human", "Customer service email"],
     "answer": "Email support@cloudnotes.example or use the Help chat in the app. We reply within one business day."},
]

CONFIDENCE_THRESHOLD = 0.30

GREETING = re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening))\b", re.I)
GOODBYE = re.compile(r"^\s*(bye|goodbye|see you|quit|exit)\b", re.I)
THANKS = re.compile(r"\b(thanks|thank you|thx)\b", re.I)

# ---------------------------------------------------------------- 2. Preprocessing
def _have(resource, name):
    """Return True if an NLTK resource is available (tries to download once)."""
    try:
        nltk.data.find(resource)
        return True
    except LookupError:
        try:
            nltk.download(name, quiet=True)
            nltk.data.find(resource)
            return True
        except Exception:
            return False


_USE_WORDNET = _have("corpora/wordnet", "wordnet")
_lemmatizer = WordNetLemmatizer() if _USE_WORDNET else None
_stemmer = PorterStemmer()

if _have("corpora/stopwords", "stopwords"):
    from nltk.corpus import stopwords
    STOP = set(stopwords.words("english"))
else:
    STOP = set(ENGLISH_STOP_WORDS)
STOP -= {"not", "no", "cant", "can't"}                  # negations carry meaning


def preprocess(text):
    """lowercase -> tokenize -> drop punctuation/stopwords -> lemmatize (or stem)."""
    tokens = wordpunct_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalnum() and t not in STOP]
    if _lemmatizer:
        try:
            return " ".join(_lemmatizer.lemmatize(t) for t in tokens)
        except LookupError:
            pass
    return " ".join(_stemmer.stem(t) for t in tokens)


# ---------------------------------------------------------------- 3. Matching model
class FAQBot:
    def __init__(self, faqs):
        self.rows = [(q, item["answer"]) for item in faqs for q in item["questions"]]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([preprocess(q) for q, _ in self.rows])

    def reply(self, message):
        if GREETING.match(message):
            return "Hello! Ask me anything about CloudNotes: accounts, billing, syncing, privacy and more."
        if GOODBYE.match(message):
            return "Goodbye! Come back any time."
        if THANKS.search(message):
            return "You're welcome! Anything else I can help with?"

        cleaned = preprocess(message)
        if not cleaned:
            return "I didn't catch that. Could you rephrase your question?"

        scores = cosine_similarity(self.vectorizer.transform([cleaned]), self.matrix)[0]
        best = int(scores.argmax())
        if scores[best] >= CONFIDENCE_THRESHOLD:
            return self.rows[best][1]

        # Low confidence: suggest the closest distinct questions instead of guessing.
        seen, suggestions = set(), []
        for i in scores.argsort()[::-1]:
            answer = self.rows[i][1]
            if scores[i] > 0 and answer not in seen:
                seen.add(answer)
                suggestions.append(self.rows[i][0])
            if len(suggestions) == 3:
                break
        if suggestions:
            return "I'm not sure I understood. Did you mean: " + " / ".join(suggestions)
        return "Sorry, I don't have an answer for that. Try rephrasing, or email support@cloudnotes.example."


# ---------------------------------------------------------------- 4. Chat UIs
def run_cli(bot):
    print("CloudNotes FAQ bot. Type 'quit' to exit.\n")
    while True:
        try:
            msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        print("Bot:", bot.reply(msg), "\n")
        if GOODBYE.match(msg):
            break


PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>FAQ Chatbot</title>
<style>
 body{margin:0;font-family:system-ui,sans-serif;background:#eef2f7;display:flex;justify-content:center}
 #app{width:100%;max-width:640px;height:100vh;display:flex;flex-direction:column;background:#fff}
 header{padding:14px 18px;background:#1f4fd8;color:#fff;font-weight:600}
 #log{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:10px}
 .m{max-width:80%;padding:10px 14px;border-radius:14px;line-height:1.45;white-space:pre-wrap}
 .bot{background:#e9eefb;align-self:flex-start}.me{background:#1f4fd8;color:#fff;align-self:flex-end}
 form{display:flex;gap:8px;padding:12px;border-top:1px solid #d5dcea}
 input{flex:1;padding:12px;border:1px solid #c3cde0;border-radius:8px;font:inherit}
 button{padding:0 18px;border:0;border-radius:8px;background:#1f4fd8;color:#fff;font:inherit;cursor:pointer}
</style></head><body><div id="app"><header>CloudNotes FAQ Assistant</header>
<div id="log"></div>
<form id="f"><input id="q" placeholder="Ask a question..." autocomplete="off" autofocus><button>Send</button></form></div>
<script>
const log=document.getElementById('log'),q=document.getElementById('q');
function add(t,c){const d=document.createElement('div');d.className='m '+c;d.textContent=t;log.appendChild(d);log.scrollTop=log.scrollHeight;}
add("Hi! Ask me anything about CloudNotes.","bot");
document.getElementById('f').onsubmit=async e=>{
  e.preventDefault();const t=q.value.trim();if(!t)return;add(t,'me');q.value='';
  try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:t})});
      add((await r.json()).reply,'bot');}
  catch{add('Connection problem. Please try again.','bot');}
};
</script></body></html>"""


def run_web(bot):
    from flask import Flask, jsonify, request
    app = Flask(__name__)

    @app.route("/")
    def index():
        return PAGE

    @app.route("/chat", methods=["POST"])
    def chat():
        message = (request.get_json(silent=True) or {}).get("message", "")
        return jsonify(reply=bot.reply(message))

    app.run(debug=False)


if __name__ == "__main__":
    bot = FAQBot(FAQS)
    run_web(bot) if "--web" in sys.argv else run_cli(bot)
