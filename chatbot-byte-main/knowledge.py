"""
knowledge.py - everything Byte "knows" (pure data, no logic).

Edit this file to teach the bot new things: add keywords to INTENTS, add
entries to KNOWLEDGE, or add jokes/facts/riddles/quiz questions.
"""

# --------------------------------------------------------------------------
# Intents: name -> keywords/phrases. Longer phrases are more specific and
# score higher. Single words also match with typos (fuzzy matching).
# --------------------------------------------------------------------------

EXIT_WORDS = ["bye", "exit", "quit", "goodbye", "see you", "see ya",
              "good night", "take care", "khuda hafiz", "allah hafiz"]
STOP_WORDS = {"stop", "quit", "exit", "cancel", "end game", "stop game", "enough"}
GIVE_UP_WORDS = {"give up", "i give up", "idk", "i dont know", "skip", "answer",
                 "show answer", "reveal"}
YES_WORDS = {"yes", "y", "yeah", "yep", "sure", "confirm", "ok", "okay"}
NO_WORDS = {"no", "n", "nope", "cancel", "dont", "never mind", "nevermind"}

INTENTS = {
    "greeting": ["hi", "hello", "hey", "hii", "heya", "howdy", "yo", "salam",
                 "assalamualaikum", "aoa", "namaste", "good morning",
                 "good afternoon", "good evening"],
    "how_are_you": ["how are you", "hows it going", "how is it going", "whats up",
                    "sup", "kya haal", "kaise ho", "how do you do",
                    "how have you been"],
    "bot_name": ["your name", "who are you", "what are you", "introduce yourself",
                 "are you a bot", "are you human", "are you real"],
    "creator": ["who made you", "who created you", "who built you",
                "who programmed you", "your creator", "your developer"],
    "user_name": ["my name", "who am i", "know my name", "remember me"],
    "recall_me": ["what do you remember", "what do you know about me"],
    "forget_me": ["forget me", "forget everything", "delete my data",
                  "erase my data", "clear my data"],
    "help": ["help", "what can you do", "menu", "commands", "features"],
    "time": ["time", "what time", "clock"],
    "date": ["date", "todays date", "what day", "which day", "day is it",
             "what is today"],
    "weather": ["weather", "forecast", "is it raining", "how hot is it",
                "how cold is it"],
    "joke": ["joke", "funny", "make me laugh", "humor", "humour"],
    "fact": ["fact", "facts", "tell me something", "did you know", "trivia",
             "something interesting"],
    "motivation": ["motivate", "motivation", "inspire", "inspiration",
                   "encourage me", "cheer me up"],
    "career": ["internship tips", "career advice", "career tips", "job tips",
               "resume tips", "interview tips", "portfolio", "how to get a job",
               "job hunt"],
    "study_tips": ["study tips", "how to study", "how to learn programming",
                   "learn to code", "learn coding", "how to learn python"],
    "debug_help": ["bug", "bugs", "error", "errors", "stuck", "not working",
                   "doesnt work", "traceback", "exception", "crashed"],
    "coin": ["flip a coin", "coin flip", "toss a coin", "heads or tails", "coin"],
    "dice": ["roll a dice", "roll a die", "roll dice", "dice"],
    "password": ["generate a password", "strong password", "random password",
                 "make a password", "password generator"],
    "games": ["play a game", "games", "lets play", "bored", "game time"],
    "guess": ["guess the number", "number guessing", "guessing game"],
    "riddle": ["riddle", "riddles", "brain teaser"],
    "quiz": ["quiz", "quiz me", "test me", "test my knowledge", "mcq"],
    "rps": ["rock paper scissors", "rps"],
    "thanks": ["thanks", "thank you", "thx", "shukriya", "jazakallah",
               "appreciate it"],
    "compliment": ["you are smart", "you are great", "you are awesome",
                   "you are cool", "you are amazing", "good bot", "nice bot",
                   "well done", "love you"],
    "rude": ["stupid bot", "you are stupid", "idiot", "shut up", "useless bot",
             "dumb bot", "hate you", "worst bot"],
    "repeat": ["another", "again", "one more", "more", "next", "another one"],
    "ack": ["ok", "okay", "alright", "cool", "nice", "hmm", "wow", "lol",
            "haha", "hehe"],
}

# some intents only make sense in short messages ("more" vs "tell me more about X")
SHORT_ONLY = {"repeat": 3, "ack": 3}
# things "another one" can repeat
REPEATABLE = {"joke", "fact", "motivation", "career", "study_tips",
              "debug_help", "coin", "dice", "riddle"}
# never combined with another intent in the same reply
NO_COMBINE = {"help", "repeat", "mood_good", "mood_bad", "games", "guess",
              "riddle", "quiz", "rps", "forget_me", "rude"}
# social intents can be combined with almost anything ("hi, what time is it?")
SOCIAL = {"greeting", "thanks", "ack"}
# these can be combined with each other ("what is your name and what time is it")
COMBINABLE = {"time", "date", "world_time", "joke", "fact", "motivation", "coin",
              "dice", "bot_name", "creator", "user_name", "math", "convert",
              "how_are_you", "weather", "password"}
# short words where "times" must not be treated as a typo of "time"
NO_FUZZY = {"time", "date", "coin"}

MOOD_GOOD = ["good", "fine", "great", "okay", "ok", "well", "awesome", "happy",
             "excellent", "better", "alhamdulillah"]
MOOD_BAD = ["bad", "sad", "tired", "stressed", "upset", "angry", "anxious",
            "worried", "terrible", "awful", "down", "not good", "not great",
            "not fine", "not well", "not okay"]
STRONG_MOOD_BAD = {"sad", "tired", "stressed", "upset", "angry", "anxious",
                   "worried", "terrible", "awful"}

# small lexicons for sentiment (used when no intent matches)
POSITIVE = {"love", "great", "awesome", "amazing", "happy", "excited", "good",
            "nice", "wonderful", "fantastic", "proud", "glad", "enjoy", "best",
            "perfect", "excellent", "brilliant", "fun", "yay", "thrilled"}
NEGATIVE = {"hate", "angry", "sad", "terrible", "awful", "bad", "worst",
            "frustrated", "frustrating", "annoying", "annoyed", "confused",
            "tired", "stressed", "anxious", "worried", "boring", "difficult",
            "hard", "failed", "broken", "lonely", "overwhelmed", "hopeless"}

# typing shortcuts fixed before matching
SLANG = {"u": "you", "r": "are", "ur": "your", "wat": "what", "wht": "what",
         "hw": "how", "pls": "please", "plz": "please", "abt": "about",
         "tiem": "time", "tym": "time", "tel": "tell", "helo": "hello",
         "hlo": "hello"}

NOT_NAMES = {"later", "back", "maybe", "please", "when", "if", "now", "soon",
             "tomorrow", "crazy", "stupid"}

EXPLAIN_TRIGGERS = {"what", "explain", "define", "tell", "meaning", "about",
                    "describe", "how", "who"}

# --------------------------------------------------------------------------
# Knowledge base: topic -> (keywords, short answer). Order matters: more
# specific topics come first (find_topic returns the first match).
# --------------------------------------------------------------------------

KNOWLEDGE = {
    "rule based": (["rule based", "rulebased"],
                   "Rule-based means every reply comes from explicit rules a "
                   "developer wrote: if this, then that. It is predictable and "
                   "easy to debug, but it can't handle what it has no rule for."),
    "chatbot": (["chatbot", "chatbots", "chat bot"],
                "A chatbot is a program that holds a conversation. Rule-based "
                "ones (like me) follow if-else rules; AI-based ones learn "
                "language patterns from data."),
    "intent": (["intent", "intents"],
               "An intent is what the user wants to do, like greeting or asking "
               "the time. I detect intents from keywords, then run the matching "
               "rule."),
    "sentiment analysis": (["sentiment analysis", "sentiment"],
                           "Sentiment analysis detects whether text is positive, "
                           "negative or neutral. I use a tiny word-list version."),
    "fuzzy matching": (["fuzzy matching", "fuzzy"],
                       "Fuzzy matching finds words that are close but not "
                       "identical, so 'hellooo' can still match 'hello'. I use "
                       "it to forgive typos."),
    "turing test": (["turing test"],
                    "The Turing Test (Alan Turing, 1950) asks whether a machine "
                    "can converse so well that a human can't tell it from a "
                    "person."),
    "hallucination": (["hallucination", "hallucinations"],
                      "In AI, a hallucination is when a model confidently states "
                      "something false. Rule-based bots like me don't invent "
                      "answers; I just say when I don't know."),
    "ai agent": (["ai agent", "ai agents", "agentic ai"],
                 "An AI agent plans and takes actions (like calling tools) to "
                 "finish a task, instead of only answering questions."),
    "llm": (["llm", "llms", "large language model", "chatgpt", "gpt"],
            "An LLM (large language model) is a huge neural network trained on "
            "lots of text to predict and generate language. ChatGPT and Claude "
            "are examples."),
    "machine learning": (["machine learning", "ml"],
                         "Machine learning lets a computer learn patterns from "
                         "data instead of following hand-written rules. Spam "
                         "filters and recommendations are typical examples."),
    "deep learning": (["deep learning", "neural network", "neural networks"],
                      "Deep learning is machine learning with many-layered "
                      "neural networks. It powers image recognition, speech and "
                      "modern language models."),
    "nlp": (["nlp", "natural language processing"],
            "NLP (natural language processing) is how computers work with human "
            "language. I use a tiny bit: cleaning text and fuzzy matching."),
    "if else": (["if else", "ifelse", "conditional", "conditionals",
                 "control flow"],
                "If-else is control flow: the program checks a condition and "
                "runs different code depending on whether it is true. It is the "
                "core of my decision-making."),
    "loop": (["loop", "loops", "while loop"],
             "A loop repeats code until a condition changes. My while loop keeps "
             "reading your messages until you say bye."),
    "variable": (["variable", "variables"],
                 "A variable is a named place in memory that stores a value, "
                 "like age = 20. You can change the value later."),
    "function": (["function", "functions"],
                 "A function is a reusable block of code with a name. Define it "
                 "once, call it whenever you need it."),
    "list": (["list", "lists", "array", "arrays"],
             "A list (or array) is an ordered collection of items, like "
             "[1, 2, 3]. You can add, remove and change items."),
    "dictionary": (["dictionary", "dictionaries", "dict"],
                   "In Python, a dictionary stores key-value pairs, like "
                   "{'city': 'Lahore'}, so you can look things up by key."),
    "string": (["string", "strings"],
               "A string is text in quotes, like 'hello'. You can join, slice "
               "and search strings."),
    "class": (["class", "classes"],
              "A class is a blueprint for creating objects. Each object made "
              "from it has its own data."),
    "oop": (["oop", "object oriented programming", "object oriented"],
            "OOP (object-oriented programming) organises code into classes and "
            "objects that bundle data and behaviour together."),
    "recursion": (["recursion", "recursive"],
                  "Recursion is when a function calls itself to solve a smaller "
                  "version of the same problem, until it reaches a base case."),
    "algorithm": (["algorithm", "algorithms"],
                  "An algorithm is a step-by-step method for solving a problem. "
                  "A cooking recipe is a good everyday example."),
    "data structure": (["data structure", "data structures"],
                       "A data structure organises data so it can be used "
                       "efficiently, like lists, stacks, queues, trees and hash "
                       "tables."),
    "regex": (["regex", "regular expression", "regular expressions"],
              "A regular expression (regex) is a pattern for finding text, like "
              "every email address in a document. I use one to spot math."),
    "database": (["database", "databases"],
                 "A database stores organised data so it can be searched and "
                 "updated efficiently, e.g. MySQL or PostgreSQL."),
    "sql": (["sql"],
            "SQL is the language for querying and managing data in relational "
            "databases, e.g. SELECT * FROM students;"),
    "html": (["html"],
             "HTML structures a web page: headings, paragraphs, links and "
             "images."),
    "css": (["css"], "CSS styles web pages: colours, fonts and layout."),
    "javascript": (["javascript", "js"],
                   "JavaScript makes web pages interactive, and also runs on "
                   "servers with Node.js."),
    "react": (["react", "reactjs"],
              "React is a JavaScript library for building user interfaces from "
              "reusable components."),
    "json": (["json"],
             "JSON is a lightweight text format for data, like "
             "{\"city\": \"Lahore\"}. APIs commonly use it."),
    "api": (["api", "apis"],
            "An API is a set of rules that lets one program talk to another, "
            "for example a chatbot asking a weather service for data. I use "
            "two of them for weather and Wikipedia."),
    "git": (["git", "github"],
            "Git tracks changes to your code; GitHub hosts repositories online "
            "so you can share work and build a public portfolio."),
    "framework": (["framework", "frameworks"],
                  "A framework is a ready-made structure for building apps "
                  "faster, like React or Django. You add your own logic."),
    "library": (["library", "libraries"],
                "A library is reusable code you import so you don't rewrite "
                "common tasks, like NumPy for maths."),
    "compiler": (["compiler", "interpreter"],
                 "A compiler translates code to machine code before it runs; an "
                 "interpreter (like Python's) runs it line by line."),
    "cache": (["cache", "caching"],
              "A cache keeps frequently used data in a fast place so it can be "
              "reused quickly."),
    "encryption": (["encryption", "encrypt"],
                   "Encryption scrambles data so only someone with the right key "
                   "can read it."),
    "cybersecurity": (["cybersecurity", "cyber security"],
                      "Cybersecurity protects systems and data from attacks. "
                      "Good basics: strong passwords, updates and two-factor "
                      "login."),
    "binary": (["binary"],
               "Binary is a number system with only 0 and 1, which is how "
               "computers store all data."),
    "debugging": (["debugging"],
                  "Debugging is finding and fixing errors in code. Reading the "
                  "error message and printing values are the first steps."),
    "internet": (["internet"],
                 "The internet is a global network of connected computers. The "
                 "web (websites) is just one service running on it."),
    "operating system": (["operating system", "operating systems"],
                         "An operating system manages hardware and software, "
                         "e.g. Windows, Linux, macOS and Android."),
    "open source": (["open source", "opensource"],
                    "Open-source software has publicly available code that "
                    "anyone can read, use and improve."),
    "cloud": (["cloud computing", "cloud"],
              "Cloud computing means renting computing power and storage over "
              "the internet instead of owning servers."),
    "big data": (["big data"],
                 "Big data means datasets too large or complex for ordinary "
                 "tools, needing special systems to store and analyse."),
    "iot": (["iot", "internet of things"],
            "IoT (Internet of Things) connects everyday devices like sensors "
            "and appliances to the internet."),
    "decodelabs": (["decodelabs", "decode labs"],
                   "DecodeLabs runs the training programme I was built for. "
                   "Project 1 is this rule-based chatbot."),
    "python": (["python"],
               "Python is a beginner-friendly, widely used language, popular in "
               "AI, data science and automation. I'm written in it."),
    "ai": (["ai", "artificial intelligence"],
           "AI is making machines do tasks that normally need human "
           "intelligence, like understanding language. I'm the simplest kind: "
           "hand-written rules."),
}

ACRONYMS = {
    "html": "HyperText Markup Language", "css": "Cascading Style Sheets",
    "http": "HyperText Transfer Protocol", "https": "HyperText Transfer Protocol Secure",
    "url": "Uniform Resource Locator", "api": "Application Programming Interface",
    "sql": "Structured Query Language", "cpu": "Central Processing Unit",
    "gpu": "Graphics Processing Unit", "ram": "Random Access Memory",
    "rom": "Read-Only Memory", "usb": "Universal Serial Bus",
    "ip": "Internet Protocol", "dns": "Domain Name System",
    "json": "JavaScript Object Notation", "xml": "eXtensible Markup Language",
    "ide": "Integrated Development Environment", "oop": "Object-Oriented Programming",
    "ai": "Artificial Intelligence", "ml": "Machine Learning",
    "nlp": "Natural Language Processing", "llm": "Large Language Model",
    "ui": "User Interface", "ux": "User Experience",
    "pwa": "Progressive Web App", "cli": "Command-Line Interface",
    "gui": "Graphical User Interface", "sdk": "Software Development Kit",
    "iot": "Internet of Things", "vpn": "Virtual Private Network",
    "lan": "Local Area Network", "wan": "Wide Area Network",
    "pdf": "Portable Document Format", "ascii": "American Standard Code for Information Interchange",
    "bios": "Basic Input/Output System", "crud": "Create, Read, Update, Delete",
    "seo": "Search Engine Optimization", "tcp": "Transmission Control Protocol",
    "ssd": "Solid-State Drive", "hdd": "Hard Disk Drive",
    "os": "Operating System", "dbms": "Database Management System",
    "faq": "Frequently Asked Questions", "regex": "Regular Expression",
}

LANG_HELLO = {
    "urdu": "Assalam o Alaikum (or just 'Salam')", "arabic": "As-salamu alaykum / Marhaba",
    "hindi": "Namaste", "spanish": "Hola", "french": "Bonjour", "german": "Hallo",
    "italian": "Ciao", "turkish": "Merhaba", "japanese": "Konnichiwa",
    "chinese": "Ni hao", "korean": "Annyeong", "portuguese": "Ola",
    "russian": "Privet", "bengali": "Nomoskar", "persian": "Salam",
    "swahili": "Jambo", "dutch": "Hallo",
}

# cities that do NOT use daylight saving time -> fixed offsets are always right
CITY_OFFSETS = {
    "karachi": 5, "lahore": 5, "islamabad": 5, "peshawar": 5, "quetta": 5,
    "dubai": 4, "abu dhabi": 4, "doha": 3, "riyadh": 3, "makkah": 3,
    "mecca": 3, "madinah": 3, "jeddah": 3, "istanbul": 3, "moscow": 3,
    "tehran": 3.5, "kabul": 4.5, "delhi": 5.5, "mumbai": 5.5, "dhaka": 6,
    "bangkok": 7, "jakarta": 7, "singapore": 8, "kuala lumpur": 8,
    "beijing": 8, "shanghai": 8, "hong kong": 8, "tokyo": 9, "seoul": 9,
    "utc": 0, "gmt": 0,
}

# WMO weather codes used by the Open-Meteo API
WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Freezing fog", 51: "Light drizzle", 53: "Moderate drizzle",
    55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain", 71: "Slight snow",
    73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers",
    82: "Violent rain showers", 85: "Slight snow showers",
    86: "Heavy snow showers", 95: "Thunderstorm",
    96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}

# --------------------------------------------------------------------------
# Content decks (drawn in shuffled order so nothing repeats too soon)
# --------------------------------------------------------------------------

JOKES = [
    "Why do programmers prefer dark mode? Because light attracts bugs!",
    "I would tell you a UDP joke, but you might not get it.",
    "There are 10 types of people: those who understand binary and those who don't.",
    "Why did the developer go broke? He used up all his cache.",
    "A SQL query walks into a bar, sees two tables and asks: can I join you?",
    "Debugging: being the detective in a crime movie where you are also the murderer.",
    "Why was the JavaScript developer sad? Because he didn't know how to null his feelings.",
    "How many programmers does it take to change a light bulb? None, that's a hardware problem.",
]

FACTS = [
    "ELIZA, built at MIT in 1966 by Joseph Weizenbaum, was one of the first "
    "chatbots. It worked by pattern matching, just like me.",
    "The term 'artificial intelligence' comes from the 1956 Dartmouth workshop.",
    "Alan Turing proposed his famous 'imitation game', now called the Turing "
    "Test, in 1950.",
    "Python is named after the comedy group Monty Python, not the snake.",
    "ALICE, a famous rule-based chatbot, was written in a markup language "
    "called AIML and won the Loebner Prize several times.",
    "In 1947 a real moth was taped into the Harvard Mark II logbook, a famous "
    "early 'bug'.",
    "Ada Lovelace is often called the first computer programmer, thanks to her "
    "notes on Charles Babbage's Analytical Engine in the 1840s.",
    "Ray Tomlinson sent the first networked email in 1971 and chose the @ "
    "symbol to separate the user from the computer.",
    "Google's name comes from 'googol', the number 1 followed by 100 zeros.",
    "The first computer mouse was invented by Douglas Engelbart in the 1960s "
    "and had a wooden shell.",
    "Tim Berners-Lee created the first website at CERN, and it went public in "
    "1991.",
]

MOTIVATION = [
    "Every expert once wrote a program that didn't run. Keep going.",
    "Small daily progress beats occasional big bursts. One more commit today.",
    "Bugs are not failures, they're the program teaching you how it really works.",
    "You don't need to know everything. You need to keep learning the next thing.",
    "Ship the imperfect version, then improve it. Done beats perfect.",
    "Comparing your chapter 1 to someone else's chapter 10 helps no one. Build yours.",
]

CAREER_TIPS = [
    "Put every project on GitHub with a clear README. Recruiters read those.",
    "Understand every line you submit. Interviewers always ask 'why did you do it this way?'.",
    "Show growth: v1 works, v2 is cleaner, v3 has tests. That story impresses.",
    "Write about your projects (a short post or README). Explaining is a skill employers notice.",
    "Apply consistently and tailor each application. Volume alone rarely works.",
    "Keep your GitHub tidy: pin your best 3-6 projects and add a short profile bio.",
    "Practice explaining your project in 60 seconds. You'll need that pitch in interviews.",
]

STUDY_TIPS = [
    "Build small projects instead of only watching tutorials. You learn most when things break.",
    "Study in 25-minute focus blocks with 5-minute breaks (the Pomodoro technique).",
    "Type the code yourself instead of copy-pasting, then change it and see what happens.",
    "Review yesterday's topic for 5 minutes before starting something new. Spaced review builds memory.",
    "Explain a concept in simple words, to a friend or to me. If you can't, revisit it.",
    "Keep a notes file of mistakes and fixes. It becomes your personal cheat sheet.",
]

DEBUG_TIPS = [
    "Read the error from the bottom up: the last line names the problem, the lines above show where.",
    "Print the variable right before the line that fails and check what it really holds.",
    "Explain the code out loud, line by line (rubber duck debugging). You'll often spot the bug yourself.",
    "Change one thing at a time, then re-run. Changing many things hides the real fix.",
    "Search the exact error message plus the language name. Someone has hit it before.",
    "Take a short break. Fresh eyes spot typos, indentation and missing colons fast.",
]

RIDDLES = [
    ("What has keys but can't open locks?", ["keyboard", "piano"], "A keyboard (or a piano)."),
    ("What has a head and a tail but no body?", ["coin"], "A coin."),
    ("What gets wetter the more it dries?", ["towel"], "A towel."),
    ("I speak without a mouth and hear without ears. What am I?", ["echo"], "An echo."),
    ("What has to be broken before you can use it?", ["egg"], "An egg."),
    ("What runs but never walks and has a mouth but never talks?", ["river"], "A river."),
    ("What can travel around the world while staying in one corner?", ["stamp"], "A stamp."),
]

QUIZ = [
    {"q": "Which Python keyword repeats code while a condition stays true?",
     "options": ["while", "loop", "repeat", "cycle"], "answer": "while"},
    {"q": "What does HTML stand for?",
     "options": ["HyperText Markup Language", "High Tech Modern Language",
                 "Home Tool Markup Language", "HyperText Machine Learning"],
     "answer": "HyperText Markup Language"},
    {"q": "Which data structure works 'last in, first out'?",
     "options": ["Stack", "Queue", "Array", "Graph"], "answer": "Stack"},
    {"q": "What does CPU stand for?",
     "options": ["Central Processing Unit", "Computer Power Unit",
                 "Central Program Utility", "Core Processing Update"],
     "answer": "Central Processing Unit"},
    {"q": "Which symbol starts a comment in Python?",
     "options": ["#", "//", "--", "**"], "answer": "#"},
    {"q": "What is the binary number 1010 in decimal?",
     "options": ["10", "8", "12", "5"], "answer": "10"},
    {"q": "Which of these is a version control tool?",
     "options": ["Git", "Excel", "Photoshop", "Chrome"], "answer": "Git"},
    {"q": "Who proposed the famous 'Turing Test' in 1950?",
     "options": ["Alan Turing", "Ada Lovelace", "John McCarthy", "Tim Berners-Lee"],
     "answer": "Alan Turing"},
    {"q": "Which language is used to query relational databases?",
     "options": ["SQL", "CSS", "HTML", "Bash"], "answer": "SQL"},
    {"q": "What does an if-else statement do?",
     "options": ["Chooses what to run based on a condition", "Repeats code forever",
                 "Stores data on disk", "Draws graphics"],
     "answer": "Chooses what to run based on a condition"},
]

# --------------------------------------------------------------------------
# Reply variations
# --------------------------------------------------------------------------

GREETINGS = ["Hello{n}! How can I help you today?",
             "Hey{n}! What would you like to do?",
             "Hi{n}! Nice to see you. Ask me anything or type 'help'."]
HOW_ARE_YOU = ["I'm doing great, thanks for asking! How about you?",
               "All systems running smoothly. How are you feeling?",
               "Doing well, no bugs today! And you?"]
THANKS = ["You're welcome!", "Anytime!", "Happy to help!"]
COMPLIMENTS = ["Thank you! I'm just well-organised if-else statements, but I'll take it.",
               "That made my day. Well, my loop iteration.",
               "Aw, thanks! I was built to be helpful."]
RUDE_REPLIES = ["Sorry I let you down. Tell me what you were trying to do and I'll try again, or type 'help'.",
                "I'm still a simple rule-based bot, so I miss things. Try rephrasing, or type 'help'."]
EMPATHY = ["That sounds frustrating. Want a joke, a study tip, or some debugging tips?",
           "Sorry to hear that. I'm here if you want a distraction or a tip."]
CHEER = ["That's great to hear!", "Love that energy. What shall we do next?"]
EMPTY_REPLIES = ["Say something and I'll respond!", "I'm listening. Type a message."]
FALLBACKS = ["Hmm, I don't have a rule for that yet.",
             "I didn't quite get that. Could you rephrase?",
             "That one is outside my rules for now."]
GOODBYES = ["Goodbye{n}! Have a great day!", "See you soon{n}! Keep coding.",
            "Bye{n}! It was nice chatting."]

SUGGESTIONS = ["what time is it", "tell me a joke", "what is ai",
               "what is a chatbot", "calculate 12 * 7", "flip a coin",
               "give me a fun fact", "motivate me", "internship tips",
               "weather in lahore", "convert 5 km to miles", "quiz me",
               "play a game", "generate a password", "show my todo list", "help"]

HELP_MAIN = (
    "I'm Byte! Ask me for help on a topic:\n"
    "- help chat\n- help tools\n- help games\n- help learn\n"
    "- help memory\n- help commands\n"
    "Or just start talking. Type bye to exit."
)
HELP_SECTIONS = {
    "chat": "Chat: hello, how are you, my name is ..., who made you, thanks. "
            "I understand typos, slang and some Roman Urdu (salam, kya haal).",
    "tools": "Tools:\n- Math: calculate 12 * 7 (or 5 plus 3)\n"
             "- Convert: 5 km to miles, 100 f to c, 2 gb to mb\n"
             "- Time: what time is it, time in Tokyo\n"
             "- Days: days until 2026-12-31\n"
             "- Weather: weather in Lahore (needs internet)\n"
             "- Password: generate a password 16\n"
             "- Text: reverse hello, spell python, is level a palindrome\n"
             "- Random: pick a number between 1 and 50, choose between tea, coffee",
    "games": "Games (say 'stop' to leave one):\n- guess the number\n- riddle\n"
             "- quiz me\n- rock paper scissors\n- flip a coin, roll a dice",
    "learn": "Learn:\n- what is AI / API / recursion / SQL ...\n"
             "- what does HTML stand for\n"
             "- who is Alan Turing (Wikipedia, needs internet)\n"
             "- fun fact, study tips, internship tips\n"
             "- I'm stuck with a bug\n- say hello in Spanish",
    "memory": "Memory (saved on this computer):\n- my name is ...\n"
              "- I live in Lahore\n- add buy milk to my todo list\n"
              "- show my todo list, done 1, remove 2, clear my todo list\n"
              "- what do you remember\n- forget me",
    "commands": "Commands: /help /history /stats /memory /save /clear /forget /exit",
}
