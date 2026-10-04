"""
Byte - Rule-Based AI Chatbot (v3.0)
DecodeLabs | Artificial Intelligence Industrial Training | Project 1

Files
    chatbot.py    the engine + the continuous loop (this file)
    knowledge.py  all the data: intents, knowledge base, jokes, riddles, quiz
    skills.py     the modules: calculator, converter, weather, Wikipedia, memory...

The "logic engine":
    user text -> normalise -> score intents (keywords + typo tolerance)
              -> if / elif decision chain -> reply -> back to the loop

Project 1 requirements covered
    * greetings and exit commands
    * if-else logic for every response   (see Chatbot.reply_for)
    * a continuous loop                  (see main)

Run:   python chatbot.py               normal
       python chatbot.py --debug       show how each decision was made
       python chatbot.py --offline     turn off weather / Wikipedia lookups
"""

import difflib
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime

import knowledge as kb
import skills

BOT_NAME = "Byte"
VERSION = "3.0"
AUTHOR = "Asma"                # shown when the user asks "who made you?"
TYPING_DELAY = 0.008           # seconds per character; 0 = instant replies
FUZZY_CUTOFF = 0.8             # how close a misspelled word must be to a keyword
GUESS_MAX_TRIES = 8
MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "byte_memory.json")

NET_MSG = ("I couldn't reach the internet just now. Weather and Wikipedia lookups "
           "need a connection.")
OFFLINE_MSG = "Online lookups are switched off (offline mode)."

NAME_RE = re.compile(r"\b(?:my name is|you can call me|call me)\s+([a-zA-Z]{2,20})\b",
                     re.IGNORECASE)


def normalize(text):
    """Lowercase, drop punctuation, fix slang: '  Wat's UP?? ' -> 'what s up'."""
    text = text.lower().replace("'", "").replace("\u2019", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(kb.SLANG.get(w, w) for w in text.split())


def can_combine(a, b):
    """May two intents be answered in the same reply?"""
    if a in kb.NO_COMBINE or b in kb.NO_COMBINE:
        return False
    if a in kb.SOCIAL or b in kb.SOCIAL:
        return True
    return a in kb.COMBINABLE and b in kb.COMBINABLE


class Chatbot:
    def __init__(self, name=None, debug=False, memory_path=None, online=True, notify=None):
        self.memory = skills.Memory(memory_path)   # saved between sessions
        if name:
            self.memory.data["name"] = name
        self.debug = debug
        self.online = online
        self.notify = notify         # optional function(text) for "Looking that up..."
        self.running = True
        self.last_intent = None      # context: what we answered last
        self.last_repeatable = None  # for "another one"
        self.pending = None          # multi-turn state (games, questions to the user)
        self.ctx = {}                # values parsed from the current message
        self.misses = 0              # consecutive fallbacks
        self.decks = {}              # shuffled decks so jokes/facts don't repeat
        self.history = []            # (time, speaker, text)
        self.mood = []               # sentiment score of each message
        self.stats = Counter()
        self.intent_counts = Counter()
        self.started = datetime.now()
        self.last_debug = ""

    # ---- small utilities --------------------------------------------------
    @property
    def name(self):
        return self.memory.data.get("name")

    @name.setter
    def name(self, value):
        self.memory.data["name"] = value
        self.memory.save()

    def n(self):
        return f", {self.name}" if self.name else ""

    def draw(self, key, options):
        """Pick from a shuffled deck: no repeats until everything has been used."""
        deck = self.decks.setdefault(key, [])
        if not deck:
            deck.extend(random.sample(options, len(options)))
        return deck.pop()

    def log(self, speaker, text):
        self.history.append((datetime.now().strftime("%H:%M:%S"), speaker, text))
        del self.history[:-500]

    def find_topic(self, clean):
        padded = f" {clean} "
        for topic, (keywords, _answer) in kb.KNOWLEDGE.items():
            if any(f" {kw} " in padded for kw in keywords):
                return topic
        return None

    def is_exit(self, clean):
        padded = f" {clean} "
        return len(clean.split()) <= 4 and any(f" {w} " in padded for w in kb.EXIT_WORDS)

    # ---- step 1: understand (score every intent) -------------------------
    def score_intents(self, clean, raw):
        words = clean.split()
        padded = f" {clean} "
        scores = {}
        ctx = self.ctx = {}

        # keyword matching, with typo tolerance for single words
        for intent, keywords in kb.INTENTS.items():
            best = 0
            for kw in keywords:
                if f" {kw} " in padded:
                    best = max(best, 3 * len(kw.split()))
                elif " " not in kw and len(kw) >= 4 and intent not in kb.NO_FUZZY:
                    for w in words:
                        if len(w) >= 5 and difflib.SequenceMatcher(None, w, kw).ratio() >= FUZZY_CUTOFF:
                            best = max(best, 2)
            if best and len(words) <= kb.SHORT_ONLY.get(intent, 99):
                scores[intent] = best

        # special detectors (each parses the message with a skill)
        detectors = [
            ("name_set", 20, lambda: NAME_RE.search(raw)),
            ("set_city", 18, lambda: skills.find_set_city(raw)),
            ("todo", 16, lambda: skills.parse_todo(raw)),
            ("math", 15, lambda: skills.extract_math(raw)),
            ("convert", 14, lambda: skills.convert_units(raw)),
            ("text_tool", 14, lambda: skills.text_tool(raw)),
            ("acronym", 13, lambda: skills.acronym(clean)),
            ("days_until", 13, lambda: skills.days_until(raw)),
            ("pick_random", 13, lambda: skills.random_tool(raw)),
            ("world_time", 13, lambda: skills.world_time(clean)),
            ("hello_in", 13, lambda: skills.hello_in(clean)),
        ]
        for intent, weight, detect in detectors:
            found = detect()
            if found:
                scores[intent] = weight
                ctx[intent] = found
        if "name_set" in scores:
            scores.pop("user_name", None)      # "my name is X" is not a question
        if "world_time" in scores:
            scores.pop("time", None)           # "time in Tokyo" is not "time"
        if "todo" in scores or "set_city" in scores:
            scores.pop("weather", None)
        if "text_tool" in scores or "hello_in" in scores:
            scores.pop("greeting", None)       # "reverse hello" / "hello in Urdu"

        if self.find_topic(clean) and set(words[:3]) & kb.EXPLAIN_TRIGGERS:
            scores["explain"] = 8

        # mood answers only make sense right after "how are you" (context!)
        in_context = self.last_intent == "how_are_you" or re.search(
            r"\b(i am|im|i feel|feeling)\b", clean)
        if len(words) <= 5 and not {"greeting", "how_are_you"} & scores.keys():
            if (in_context and any(f" {p} " in padded for p in kb.MOOD_BAD)) or (
                    len(words) <= 3 and kb.STRONG_MOOD_BAD & set(words)):
                scores["mood_bad"] = 7
            elif in_context and any(f" {p} " in padded for p in kb.MOOD_GOOD):
                scores["mood_good"] = 7

        return sorted(scores.items(), key=lambda pair: -pair[1])

    # ---- step 2: decide and reply (the if / elif chain) ------------------
    def reply_for(self, intent, clean, raw):
        if intent in kb.REPEATABLE:
            self.last_repeatable = intent
        ctx = self.ctx

        if intent == "greeting":
            if any(w in clean.split() for w in ("salam", "assalamualaikum", "aoa")):
                return f"Walaikum assalam{self.n()}! How can I help you?"
            elif "good morning" in clean:
                return f"Good morning{self.n()}! Hope your day starts well. How can I help?"
            elif "good afternoon" in clean:
                return f"Good afternoon{self.n()}! What can I do for you?"
            elif "good evening" in clean:
                return f"Good evening{self.n()}! What can I do for you?"
            return random.choice(kb.GREETINGS).format(n=self.n())

        elif intent == "how_are_you":
            if "kya haal" in clean or "kaise ho" in clean:
                return "Main theek hoon, shukriya! Aap kaise hain?"
            return random.choice(kb.HOW_ARE_YOU)

        elif intent == "mood_good":
            return "Glad to hear it! Want a joke, a fun fact, or a quick game?"

        elif intent == "mood_bad":
            return ("Sorry to hear that. Want a joke or a motivating line? And if "
                    "things feel heavy, talking to someone you trust really helps.")

        elif intent == "bot_name":
            if "human" in clean or "real" in clean:
                return (f"Nope, I'm {BOT_NAME}, a program. No machine learning "
                        "inside, just careful if-else logic.")
            return (f"I'm {BOT_NAME}, a rule-based chatbot built at DecodeLabs. "
                    "Type 'help' to see my skills.")

        elif intent == "creator":
            return (f"I was created by {AUTHOR} during the DecodeLabs AI internship "
                    f"(Batch 2026). I'm version {VERSION}.")

        elif intent == "name_set":
            candidate = NAME_RE.search(raw).group(1).title()
            if candidate.lower() in kb.NOT_NAMES:
                return "Hmm, I'd love to know your real name. Try: my name is ..."
            self.name = candidate
            return f"Nice to meet you, {candidate}! I'll remember your name."

        elif intent == "user_name":
            if self.name:
                return f"You're {self.name}, as far as I know."
            return "I don't know your name yet. Tell me: my name is ..."

        elif intent == "set_city":
            self.memory.data["city"] = ctx["set_city"]
            self.memory.save()
            return f"Got it, you live in {ctx['set_city']}. Ask me 'weather' any time."

        elif intent == "recall_me":
            return self.memory_text()

        elif intent == "forget_me":
            self.pending = {"type": "confirm_forget"}
            return ("This will erase your name, city and to-do list from this "
                    "computer. Are you sure? (yes/no)")

        elif intent == "help":
            for section, text in kb.HELP_SECTIONS.items():
                if f" {section} " in f" {clean} ":
                    return text
            return kb.HELP_MAIN

        elif intent == "time":
            return "It's " + datetime.now().strftime("%I:%M %p").lstrip("0") + " right now."

        elif intent == "world_time":
            return ctx["world_time"]

        elif intent == "date":
            return "Today is " + datetime.now().strftime("%A, %d %B %Y") + "."

        elif intent == "days_until":
            return ctx["days_until"]

        elif intent == "weather":
            city = skills.extract_city(raw) or self.memory.data.get("city")
            if not city:
                self.pending = {"type": "ask_city"}
                return "Which city? (say the city name, or 'stop')"
            return self.weather_reply(city)

        elif intent == "joke":
            return self.draw("joke", kb.JOKES)

        elif intent == "fact":
            return self.draw("fact", kb.FACTS)

        elif intent == "motivation":
            return self.draw("motivation", kb.MOTIVATION)

        elif intent == "career":
            return self.draw("career", kb.CAREER_TIPS)

        elif intent == "study_tips":
            return self.draw("study", kb.STUDY_TIPS)

        elif intent == "debug_help":
            return self.draw("debug", kb.DEBUG_TIPS)

        elif intent == "coin":
            return random.choice(["Heads!", "Tails!"])

        elif intent == "dice":
            return f"You rolled a {random.randint(1, 6)}."

        elif intent == "pick_random":
            return ctx["pick_random"]

        elif intent == "password":
            digits = re.search(r"\d+", raw)
            password = skills.make_password(int(digits.group()) if digits else 14)
            return (f"Here's a strong password: {password}\n"
                    "Don't reuse it, and note it will appear in the log if you /save.")

        elif intent == "math":
            expression = ctx["math"]
            try:
                return f"{expression} = {skills.fmt_number(skills.safe_eval(expression))}"
            except ZeroDivisionError:
                return "I can't divide by zero, and neither can any computer."
            except (ValueError, SyntaxError, OverflowError):
                return "I couldn't work that one out. Try something like: calculate 12 * 7"

        elif intent == "convert":
            return ctx["convert"]

        elif intent == "text_tool":
            return ctx["text_tool"]

        elif intent == "acronym":
            return ctx["acronym"]

        elif intent == "hello_in":
            return ctx["hello_in"]

        elif intent == "explain":
            return kb.KNOWLEDGE[self.find_topic(clean)][1]

        elif intent == "todo":
            return self.todo_reply(*ctx["todo"])

        elif intent == "games":
            return ("Let's play! Try: guess the number, riddle, quiz me, "
                    "rock paper scissors, flip a coin or roll a dice.")

        elif intent == "guess":
            return self.start_guess()

        elif intent == "riddle":
            return self.start_riddle()

        elif intent == "quiz":
            return self.start_quiz()

        elif intent == "rps":
            self.pending = {"type": "rps", "win": 0, "lose": 0, "draw": 0}
            return "Rock, paper or scissors? (say 'stop' to finish)"

        elif intent == "thanks":
            if "shukriya" in clean or "jazakallah" in clean:
                return "Koi baat nahi! You're welcome."
            return random.choice(kb.THANKS)

        elif intent == "compliment":
            return random.choice(kb.COMPLIMENTS)

        elif intent == "rude":
            return random.choice(kb.RUDE_REPLIES)

        elif intent == "repeat":
            if self.last_repeatable:
                return self.reply_for(self.last_repeatable, clean, raw)
            return "Another what? Ask me for a joke, a fact or a riddle first."

        elif intent == "ack":
            if self.last_intent in kb.REPEATABLE:
                return "Glad you liked it! Say 'another' for one more."
            return "Anything else I can help with? Type 'help' to see my skills."

        else:
            return self.fallback(clean)

    # ---- skills that need the bot's state ---------------------------------
    def memory_text(self):
        data = self.memory.data
        lines = []
        if data.get("name"):
            lines.append(f"Your name: {data['name']}")
        if data.get("city"):
            lines.append(f"Your city: {data['city']}")
        open_tasks = sum(1 for t in data["todos"] if not t["done"])
        if data["todos"]:
            lines.append(f"To-do list: {open_tasks} open, {len(data['todos']) - open_tasks} done")
        if data.get("visits"):
            lines.append(f"Sessions together: {data['visits']}")
        if not lines:
            return "I don't have anything saved about you yet. Try: my name is ..."
        return "What I have saved on this computer:\n" + "\n".join(lines)

    def todo_reply(self, action, arg):
        todos = self.memory.data["todos"]
        if action == "add":
            text = arg.strip().rstrip(".!")[:120]
            todos.append({"text": text, "done": False})
            self.memory.save()
            return f"Added: {text}. You have {sum(1 for t in todos if not t['done'])} open task(s)."
        elif action == "list":
            if not todos:
                return "Your to-do list is empty. Try: add buy milk to my todo list"
            return "\n".join(f"{i}. [{'x' if t['done'] else ' '}] {t['text']}"
                             for i, t in enumerate(todos, 1))
        elif action == "clear":
            count = len(todos)
            todos.clear()
            self.memory.save()
            return f"Cleared {count} task(s)."
        index = int(arg) - 1
        if not 0 <= index < len(todos):
            return f"There's no task number {arg}. Say 'show my todo list' to see them."
        if action == "done":
            todos[index]["done"] = True
            self.memory.save()
            return f"Marked done: {todos[index]['text']}. Nice work!"
        removed = todos.pop(index)
        self.memory.save()
        return f"Removed: {removed['text']}"

    def weather_reply(self, city):
        if not self.online:
            return OFFLINE_MSG
        if self.notify:
            self.notify("Checking the forecast...")
        try:
            report = skills.weather_report(city)
        except skills.NET_ERRORS:
            self.stats["fallbacks"] += 1
            return NET_MSG
        if report is None:
            return f"I couldn't find a place called '{city}'. Check the spelling?"
        self.stats["lookups"] += 1
        return report

    def wiki_reply(self, topic):
        self.stats["fallbacks"] += 1
        if not self.online:
            return "I don't have that in my rules, and online lookups are switched off."
        if self.notify:
            self.notify("Let me look that up...")
        try:
            result = skills.wiki_summary(topic)
        except skills.NET_ERRORS:
            return "I don't have that in my rules, and I couldn't reach the internet to look it up."
        if result is None:
            return f"I couldn't find anything about '{topic}'. Try different words?"
        self.stats["fallbacks"] -= 1      # a successful lookup is not a miss
        self.stats["lookups"] += 1
        title, text = result
        return f"{title}: {text}\n(Source: Wikipedia)"

    # ---- games and multi-turn conversations -------------------------------
    def start_guess(self):
        self.pending = {"type": "guess", "target": random.randint(1, 100), "tries": 0}
        return (f"I'm thinking of a number between 1 and 100. You have "
                f"{GUESS_MAX_TRIES} tries. What's your guess?")

    def start_riddle(self):
        question, answers, shown = self.draw("riddle", kb.RIDDLES)
        self.pending = {"type": "riddle", "answers": answers, "shown": shown, "tries": 0}
        return f"Riddle: {question}"

    def start_quiz(self):
        questions = []
        for item in random.sample(kb.QUIZ, 5):
            options = random.sample(item["options"], len(item["options"]))
            questions.append({"q": item["q"], "options": options, "answer": item["answer"]})
        self.pending = {"type": "quiz", "qs": questions, "i": 0, "score": 0}
        return "Quiz time! 5 questions. Answer with A, B, C or D (say 'stop' to quit).\n\n" \
               + self.format_question()

    def format_question(self):
        p = self.pending
        q = p["qs"][p["i"]]
        lines = [f"Q{p['i'] + 1}/{len(p['qs'])}: {q['q']}"]
        lines += [f"{'ABCD'[i]}) {option}" for i, option in enumerate(q["options"])]
        return "\n".join(lines)

    def end_pending(self):
        p, self.pending = self.pending, None
        if p and p["type"] == "rps":
            return (f"Final score: you {p['win']}, me {p['lose']}, draws {p['draw']}. "
                    "Good game!")
        return "Okay, stopped. What next?"

    def step_pending(self, raw, clean):
        """Continue a multi-turn conversation (a game or a question we asked)."""
        p = self.pending
        kind = p["type"]

        if kind == "guess":
            if clean in kb.GIVE_UP_WORDS:
                self.pending = None
                return f"The number was {p['target']}. Say 'guess the number' to play again."
            match = re.search(r"-?\d+", raw)
            if not match:
                return "Send me a number between 1 and 100 (or say 'give up')."
            guess = int(match.group())
            p["tries"] += 1
            if guess == p["target"]:
                self.pending = None
                return f"Correct! It was {guess}, and you got it in {p['tries']} tries."
            if p["tries"] >= GUESS_MAX_TRIES:
                self.pending = None
                return f"Out of tries! The number was {p['target']}."
            hint = "higher" if guess < p["target"] else "lower"
            return f"Go {hint}. Tries left: {GUESS_MAX_TRIES - p['tries']}."

        elif kind == "riddle":
            if clean in kb.GIVE_UP_WORDS:
                self.pending = None
                return f"The answer: {p['shown']} Say 'another' for a new riddle."
            if any(a in clean.split() for a in p["answers"]):
                self.pending = None
                return f"Correct! {p['shown']} Say 'another' for one more."
            p["tries"] += 1
            if p["tries"] >= 3:
                self.pending = None
                return f"Not quite. The answer: {p['shown']}"
            return "Not quite. Try again, or say 'give up'."

        elif kind == "quiz":
            question = p["qs"][p["i"]]
            correct = question["options"].index(question["answer"])
            choice = None
            match = re.fullmatch(r"(?:option |answer )?([abcd])", clean)
            if match:
                choice = "abcd".index(match.group(1))
            elif clean not in kb.GIVE_UP_WORDS:
                for i, option in enumerate(question["options"]):
                    if normalize(option) == clean:
                        choice = i
                if choice is None:
                    return "Answer with A, B, C or D (or say 'stop')."
            if choice == correct:
                p["score"] += 1
                feedback = "Correct!"
            else:
                feedback = f"Not quite. The answer: {'ABCD'[correct]}) {question['answer']}"
            p["i"] += 1
            if p["i"] >= len(p["qs"]):
                score, total = p["score"], len(p["qs"])
                self.pending = None
                verdict = ("Excellent!" if score >= 4 else "Good effort!" if score >= 2
                           else "Keep practising, you'll get there!")
                return f"{feedback}\n\nQuiz over: {score}/{total}. {verdict} Say 'quiz me' to try again."
            return f"{feedback}\n\n{self.format_question()}"

        elif kind == "rps":
            moves = ("rock", "paper", "scissors")
            words = clean.split()
            user = next((m for m in moves if m in words), "scissors" if "scissor" in words else None)
            if user is None:
                return "Say rock, paper or scissors (or 'stop')."
            bot = random.choice(moves)
            beats = {"rock": "scissors", "scissors": "paper", "paper": "rock"}
            if user == bot:
                p["draw"] += 1
                result = "It's a draw."
            elif beats[user] == bot:
                p["win"] += 1
                result = "You win!"
            else:
                p["lose"] += 1
                result = "I win!"
            return f"You: {user} | Me: {bot}. {result} Score: you {p['win']} - me {p['lose']}. Again?"

        elif kind == "confirm_forget":
            if clean in kb.YES_WORDS:
                self.memory.forget()
                self.pending = None
                return "Done. I've erased everything I had saved about you."
            if clean in kb.NO_WORDS:
                self.pending = None
                return "Okay, I'll keep everything."
            return "Please answer yes or no."

        elif kind == "ask_city":
            city = re.sub(r"^(in|at|for)\s+", "", raw.strip(), flags=re.IGNORECASE)
            if not re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,40}", city):
                return "That doesn't look like a city name. Try again, or say 'stop'."
            self.pending = None
            return self.weather_reply(city.title())

        self.pending = None
        return "Okay, let's start fresh. What would you like to do?"

    # ---- fallback and the main decision step ------------------------------
    def fallback(self, clean):
        """Nothing matched: be honest, suggest what might have been meant."""
        self.stats["fallbacks"] += 1
        self.misses += 1
        close = difflib.get_close_matches(clean, kb.SUGGESTIONS, n=2, cutoff=0.45)
        if close:
            options = " or ".join(f"'{c}'" for c in close)
            return f"I'm not sure I got that. Did you mean {options}?"
        if self.misses >= 2:
            return "I'm still learning! Type 'help' to see what I can do."
        return random.choice(kb.FALLBACKS) + " Type 'help' to see what I can do."

    def chat(self, raw, clean):
        scored = self.score_intents(clean, raw)
        self.last_debug = ", ".join(f"{i}({s})" for i, s in scored) or "no match"

        if not scored:
            self.last_intent = None
            topic = skills.wiki_topic(raw)
            if topic:                              # "who is Alan Turing?" -> Wikipedia
                return self.wiki_reply(topic)
            feeling = skills.sentiment(clean)      # otherwise read the mood
            if feeling < 0:
                return random.choice(kb.EMPATHY)
            if feeling > 0:
                return random.choice(kb.CHEER)
            return self.fallback(clean)

        chosen = [scored[0][0]]
        # answer two things at once: "hi, what time is it?"
        if len(scored) > 1 and scored[1][1] >= 3 and can_combine(scored[0][0], scored[1][0]):
            chosen.append(scored[1][0])
        chosen.sort(key=lambda i: 0 if i in kb.SOCIAL else 1)

        replies = []
        for intent in chosen:
            self.intent_counts[intent] += 1
            replies.append(self.reply_for(intent, clean, raw))
        self.misses = 0
        self.last_intent = chosen[-1]
        return " ".join(replies)

    # ---- commands ---------------------------------------------------------
    def stats_text(self):
        total = self.stats["messages"]
        misses = self.stats["fallbacks"]
        rate = f"{100 * misses / total:.0f}%" if total else "0%"
        top = ", ".join(f"{k} x{v}" for k, v in self.intent_counts.most_common(3)) or "none yet"
        minutes = (datetime.now() - self.started).seconds // 60
        overall = sum(self.mood)
        mood = "positive" if overall > 1 else "negative" if overall < -1 else "neutral"
        return (f"Messages: {total} | Not understood: {misses} ({rate})\n"
                f"Top topics: {top}\nOnline lookups: {self.stats['lookups']}\n"
                f"Conversation mood: {mood} | Session: {minutes} min")

    def save_log(self):
        filename = "chat_log_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                for stamp, speaker, text in self.history:
                    f.write(f"[{stamp}] {speaker}: {text}\n")
            return f"Saved this conversation to {filename}"
        except OSError:
            return "Sorry, I couldn't save the file here."

    def command(self, raw):
        cmd = raw.lower().split()[0]
        if cmd in ("/help", "/menu"):
            return kb.HELP_MAIN
        elif cmd == "/history":
            lines = [f"{who}: {text}" for _t, who, text in self.history[-11:-1]]
            return "\n".join(lines) if lines else "No history yet."
        elif cmd == "/stats":
            return self.stats_text()
        elif cmd == "/memory":
            return self.memory_text()
        elif cmd == "/forget":
            return self.reply_for("forget_me", "", raw)
        elif cmd == "/save":
            return self.save_log()
        elif cmd == "/clear":
            self.history.clear()
            return "History cleared."
        elif cmd in ("/exit", "/quit"):
            return self.goodbye()
        return "Unknown command. Try /help."

    def goodbye(self):
        self.running = False
        self.pending = None
        message = random.choice(kb.GOODBYES).format(n=self.n())
        return f"{message} We exchanged {self.stats['messages']} messages."

    # ---- public entry point -----------------------------------------------
    def respond(self, text):
        raw = text.strip()
        self.stats["messages"] += 1
        self.last_debug = ""
        self.log("You", raw)
        if raw == "":
            reply = random.choice(kb.EMPTY_REPLIES)
        elif raw.startswith("/"):
            reply = self.command(raw)
        else:
            clean = normalize(raw)
            self.mood.append(skills.sentiment(clean))
            if self.pending and clean in kb.STOP_WORDS:
                reply = self.end_pending()
            elif self.is_exit(clean):
                reply = self.goodbye()
            elif self.pending:
                reply = self.step_pending(raw, clean)
            else:
                reply = self.chat(raw, clean)
        self.log(BOT_NAME, reply)
        return reply


# --------------------------------------------------------------------------
# TERMINAL INTERFACE + THE CONTINUOUS LOOP
# --------------------------------------------------------------------------

USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
if USE_COLOR and os.name == "nt":
    os.system("")   # enables ANSI colours in Windows terminals


def paint(text, code):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def say(text):
    sys.stdout.write(paint(f"{BOT_NAME}: ", "36;1"))
    if TYPING_DELAY and sys.stdout.isatty() and len(text) < 200:
        for ch in text:
            sys.stdout.write(ch)
            sys.stdout.flush()
            time.sleep(TYPING_DELAY)
        sys.stdout.write("\n")
    else:
        print(text)


def parse_name(text):
    text = re.sub(r"^(my name is|i'?m|i am|call me|its|it is)\s+", "", text.strip(), flags=re.I)
    return text.title() if re.fullmatch(r"[A-Za-z]{2,20}", text) else None


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    bot = Chatbot(debug="--debug" in sys.argv, memory_path=MEMORY_FILE,
                  online="--offline" not in sys.argv, notify=say)
    print(paint(f"=== {BOT_NAME} | Rule-Based AI Chatbot v{VERSION} | DecodeLabs Project 1 ===", "1"))
    print("Type 'help' for my skills, 'bye' to exit.\n")

    data = bot.memory.data
    returning = bool(bot.name)
    data["visits"] += 1
    previous_visit = data.get("last_visit")
    data["last_visit"] = datetime.now().strftime("%Y-%m-%d")
    bot.memory.save()

    if returning:
        say(f"Welcome back{bot.n()}!" + (f" Last time we chatted was {previous_visit}." if previous_visit else ""))
        open_tasks = sum(1 for t in data["todos"] if not t["done"])
        if open_tasks:
            say(f"You have {open_tasks} open task(s). Say 'show my todo list' to see them.")
    else:
        say(f"Hi! I'm {BOT_NAME}. What should I call you?")
        try:
            bot.name = parse_name(input(paint("You: ", "32;1")))
        except (EOFError, KeyboardInterrupt):
            print()
            return
        say(f"Nice to meet you{bot.n()}! Ask me anything.")

    # The continuous loop: runs until the user says an exit word
    while bot.running:
        try:
            user_text = input(paint("You: ", "32;1"))
        except (EOFError, KeyboardInterrupt):
            print()
            user_text = "bye"
        reply = bot.respond(user_text)
        if bot.debug and bot.last_debug:
            print(paint(f"   [debug] intents: {bot.last_debug}", "2"))
        say(reply)


if __name__ == "__main__":
    main()
