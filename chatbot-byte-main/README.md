# Byte - Rule-Based AI Chatbot

Byte is a **rule-based chatbot** written in Python. Every reply comes from explicit `if / elif` logic, so you can trace exactly why the bot said what it said. It needs **no API keys and no third-party packages** (standard library only).

Besides chatting, Byte can remember you between sessions, keep a to-do list, run games, do calculations and unit conversions, and look up live weather and Wikipedia summaries.

## Project requirements

| Requirement | Where it lives |
|---|---|
| Handle greetings and exit commands | `greeting` intent, `is_exit()`, `goodbye()` |
| Use if-else logic for responses | `Chatbot.reply_for()` in `chatbot.py` |
| Run in a continuous loop | `while bot.running:` in `main()` |

## Features

**Conversation**
- Understands typos ("hellooo", "jokess"), slang ("wat tiem is it") and some Roman Urdu ("salam", "kya haal hai")
- Answers two things at once: "hi, what time is it?"
- Context-aware: "another one" repeats the last joke or fact; mood replies only make sense after "how are you"
- Simple sentiment detection: kind replies when you sound frustrated
- Honest fallback with "Did you mean...?" suggestions

**Memory (saved locally in `byte_memory.json`)**
- Remembers your name and city between sessions ("Welcome back!")
- To-do list: `add buy milk to my todo list`, `show my todo list`, `done 1`, `remove 2`, `clear my todo list`
- `what do you remember` shows what is saved; `forget me` erases it

**Tools**
- Safe calculator (built with `ast`, never `eval`): `calculate (2+3)*4`, `5 plus 3`
- Unit converter: `5 km to miles`, `100 f to c`, `2 gb to mb`
- World time: `time in Tokyo` (cities without daylight saving)
- Countdown: `days until 2027-01-01`
- Password generator, random picker, text tools (`reverse`, `spell`, palindrome check)
- **Weather** for any city (Open-Meteo API, no key needed)
- **Wikipedia** summaries: `who is Alan Turing`

**Learn and play**
- About 45 tech topics (`what is recursion`), 45 acronyms (`what does HTML stand for`)
- Jokes, fun facts, study tips, debugging tips, internship tips
- Games: guess the number, riddles, a 5-question tech quiz, rock paper scissors

**Developer features**
- `--debug` mode prints the intent scores behind every reply
- Commands: `/help /history /stats /memory /save /clear /forget /exit`
- 43 unit tests (network calls are mocked, so tests run offline)

## How it works

```
user text
   |
normalise  (lowercase, strip punctuation, fix slang)
   |
score every intent  (keywords + fuzzy matching + special detectors)
   |
if / elif decision chain  ->  reply
   |
back to the loop until the user says "bye"
```

Each message is scored against all intents. Longer, more specific phrases score higher, and single words are matched with typo tolerance (`difflib`). Special detectors (math, unit conversion, to-do commands, etc.) parse the message with a small skill function. The winning intent goes through an explicit `if / elif` chain in `reply_for()`. Multi-turn features such as games use a small state object (`self.pending`).

## Project structure

```
chatbot.py        the engine and the continuous loop
knowledge.py      all data: intents, knowledge base, jokes, riddles, quiz
skills.py         modules: calculator, converter, weather, Wikipedia, memory...
test_chatbot.py   unit tests
```

To teach Byte something new, edit `knowledge.py` (no engine changes needed).

## Run it

Requires Python 3.8 or newer.

```bash
python chatbot.py             # normal
python chatbot.py --debug     # show how each decision is made
python chatbot.py --offline   # turn off weather and Wikipedia lookups
python -m unittest test_chatbot -v   # run the tests
```

## Example conversation

```
Byte: Hi! I'm Byte. What should I call you?
You: asma
Byte: Nice to meet you, Asma! Ask me anything.
You: calculate 12 * 7
Byte: 12 * 7 = 84
You: convert 5 km to miles
Byte: 5 km = 3.1069 miles
You: add finish project 1 to my todo list
Byte: Added: finish project 1. You have 1 open task(s).
You: weather
Byte: Which city? (say the city name, or 'stop')
You: karachi
Byte: Karachi, Pakistan: Clear sky, 32.8°C (feels like 37.7°C). Humidity 57%, wind 7.0 km/h. Today: high 38.0°C, low 26.2°C.
You: wikipedia computer
Byte: Computer: A computer is a machine that can be programmed to automatically carry out sequences of arithmetic or logical operations (computation). ...
      (Source: Wikipedia)
```

## Privacy

- Your name, city and to-do list are stored only in `byte_memory.json` on your own computer. `forget me` deletes it.
- Weather and Wikipedia lookups send only the city or topic you typed to the public Open-Meteo and Wikipedia APIs. No personal data is sent, and no keys are used.

## Limitations (by design)

Byte only understands what its rules describe. It does not learn from conversations and cannot handle open-ended chat. That is the trade-off of rule-based systems: fully predictable and easy to debug, but limited. Machine learning and LLMs address this, and are natural next steps.

## Ideas for next versions

- Load intents from an `intents.json` file so new rules need no code changes
- A web interface (HTML/JS or Streamlit)
- Hybrid mode: rules first, an LLM API as a fallback for unknown questions

## Author

Built by Asma ([github.com/Asma-Amin](https://github.com/Asma-Amin)) during the DecodeLabs AI internship, Batch sep 2026.
