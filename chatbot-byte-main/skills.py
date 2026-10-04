"""
skills.py - Byte's "modules": small, independent tools the chatbot can use.

Everything here is plain Python from the standard library. Offline skills
(math, converter, time, text tools...) always work; online skills (weather,
Wikipedia) fail gracefully when there is no internet.
"""

import ast
import http.client
import json
import operator
import os
import random
import re
import secrets
import string
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone

import knowledge as kb

# errors that mean "the online lookup did not work"
NET_ERRORS = (OSError, ValueError, KeyError, IndexError, TypeError,
              http.client.HTTPException)


def fmt_number(value):
    """5.0 -> '5', 3.14159265 -> '3.14159'."""
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.6g}"
    return str(value)


# ==========================================================================
# 1. Calculator (safe: uses ast, never eval)
# ==========================================================================

_MATH_WORDS = [
    (r"\bmultiplied by\b|\btimes\b", "*"),
    (r"\bdivided by\b", "/"),
    (r"\bto the power of\b", "**"),
    (r"\bplus\b", "+"),
    (r"\bminus\b", "-"),
    (r"\bmodulo\b|\bmod\b", "%"),
]
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def extract_math(raw):
    """'what is 5 plus 3?' -> '5 + 3'. Returns None if there is no maths."""
    text = raw.lower().replace(",", "")
    text = re.sub(r"\d{4}-\d{2}-\d{2}", " ", text)   # ignore dates like 2026-12-31
    for pattern, symbol in _MATH_WORDS:
        text = re.sub(pattern, symbol, text)
    text = text.replace("^", "**").replace("\u00d7", "*").replace("\u00f7", "/")
    text = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", text)
    candidates = re.findall(r"[\d.()\s+\-*/%]+", text)
    candidates = [c.strip() for c in candidates
                  if re.search(r"\d\s*[+\-*/%]+\s*[\d(]", c)]
    return max(candidates, key=len) if candidates else None


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


def safe_eval(expression):
    return _eval_node(ast.parse(expression, mode="eval"))


# ==========================================================================
# 2. Unit converter
# ==========================================================================

_UNITS = {}


def _add(category, factor, *names):
    for name in names:
        _UNITS[name] = (category, factor)


_add("length", 1, "m", "meter", "meters", "metre", "metres")
_add("length", 1000, "km", "kilometer", "kilometers", "kilometre", "kilometres")
_add("length", 0.01, "cm", "centimeter", "centimeters")
_add("length", 0.001, "mm", "millimeter", "millimeters")
_add("length", 1609.344, "mile", "miles", "mi")
_add("length", 0.3048, "ft", "foot", "feet")
_add("length", 0.0254, "inch", "inches")
_add("length", 0.9144, "yard", "yards", "yd")
_add("mass", 1, "kg", "kilogram", "kilograms", "kilo", "kilos")
_add("mass", 0.001, "g", "gram", "grams")
_add("mass", 0.45359237, "lb", "lbs", "pound", "pounds")
_add("mass", 0.028349523125, "oz", "ounce", "ounces")
_add("volume", 1, "l", "liter", "liters", "litre", "litres")
_add("volume", 0.001, "ml", "milliliter", "milliliters")
_add("volume", 3.785411784, "gallon", "gallons")
_add("data", 1, "byte", "bytes")
_add("data", 1024, "kb")
_add("data", 1024 ** 2, "mb")
_add("data", 1024 ** 3, "gb")
_add("data", 1024 ** 4, "tb")
_TEMP = {"c": "c", "celsius": "c", "f": "f", "fahrenheit": "f", "k": "k", "kelvin": "k"}
_CONVERT_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*\u00b0?\s*([a-zA-Z]+)\s+(?:to|in|into)\s+\u00b0?\s*([a-zA-Z]+)")


def convert_units(raw):
    """'5 km to miles' -> '5 km = 3.10686 miles'. None if it isn't a conversion."""
    match = _CONVERT_RE.search(raw)
    if not match:
        return None
    value = float(match.group(1))
    a, b = match.group(2).lower(), match.group(3).lower()
    known_a, known_b = a in _TEMP or a in _UNITS, b in _TEMP or b in _UNITS
    if not (known_a and known_b):
        return None
    if a in _TEMP and b in _TEMP:
        kind_a, kind_b = _TEMP[a], _TEMP[b]
        celsius = {"c": value, "f": (value - 32) * 5 / 9, "k": value - 273.15}[kind_a]
        result = {"c": celsius, "f": celsius * 9 / 5 + 32, "k": celsius + 273.15}[kind_b]
        return f"{fmt_number(value)} {a} = {fmt_number(round(result, 2))} {b}"
    if a in _UNITS and b in _UNITS and _UNITS[a][0] == _UNITS[b][0]:
        result = value * _UNITS[a][1] / _UNITS[b][1]
        note = " (1024-based)" if _UNITS[a][0] == "data" else ""
        return f"{fmt_number(value)} {a} = {fmt_number(round(result, 4))} {b}{note}"
    return f"I can't convert {a} to {b}: they measure different things."


# ==========================================================================
# 3. World time (cities without daylight saving, so fixed offsets are exact)
# ==========================================================================

def world_time(clean):
    """'what time is it in tokyo' -> "It's 6:30 PM in Tokyo (UTC+9)."""
    if "time" not in clean.split():
        return None
    padded = f" {clean} "
    for city in sorted(kb.CITY_OFFSETS, key=len, reverse=True):
        if f" {city} " in padded:
            offset = kb.CITY_OFFSETS[city]
            now = datetime.now(timezone.utc) + timedelta(hours=offset)
            hours, minutes = int(abs(offset)), int(round((abs(offset) % 1) * 60))
            label = f"UTC{'+' if offset >= 0 else '-'}{hours}" + (f":{minutes:02d}" if minutes else "")
            return f"It's {now.strftime('%I:%M %p').lstrip('0')} in {city.title()} ({label})."
    return None


# ==========================================================================
# 4. Text tools
# ==========================================================================

_TEXT_RE = re.compile(
    r"^\s*(reverse|spell|uppercase|lowercase|shout|count words in|palindrome|is)\s+(.+?)\s*\??\s*$",
    re.IGNORECASE)


def _is_palindrome(text):
    letters = re.sub(r"[^a-z0-9]", "", text.lower())
    return bool(letters) and letters == letters[::-1]


def text_tool(raw):
    """'reverse hello' -> 'olleh'. None if the message is not a text command."""
    match = _TEXT_RE.match(raw)
    if not match:
        return None
    command, arg = match.group(1).lower(), match.group(2).strip().strip("\"'")
    if command == "reverse":
        return arg[::-1]
    if command == "spell":
        return "-".join(ch.upper() for ch in arg if not ch.isspace())
    if command in ("uppercase", "shout"):
        return arg.upper()
    if command == "lowercase":
        return arg.lower()
    if command == "count words in":
        return f"{len(arg.split())} word(s), {len(arg)} characters."
    if command == "palindrome":
        word = arg
    else:  # "is <word> a palindrome"
        inner = re.match(r"(.+?)\s+a\s+palindrome$", arg, re.IGNORECASE)
        if not inner:
            return None
        word = inner.group(1)
    if _is_palindrome(word):
        return f"Yes, '{word}' is a palindrome."
    return f"No, '{word}' is not a palindrome."


# ==========================================================================
# 5. Password generator, random picker, date maths, greetings, acronyms
# ==========================================================================

def make_password(length=14):
    length = max(8, min(64, length))
    pools = [string.ascii_lowercase, string.ascii_uppercase, string.digits, "!@#$%^&*-_"]
    chars = [secrets.choice(pool) for pool in pools]
    everything = "".join(pools)
    chars += [secrets.choice(everything) for _ in range(length - len(chars))]
    secrets.SystemRandom().shuffle(chars)
    return "".join(chars)


def random_tool(raw):
    """'pick a number between 1 and 50' / 'choose between tea, coffee or juice'."""
    match = re.search(r"between\s+(-?\d+)\s+(?:and|to)\s+(-?\d+)", raw, re.IGNORECASE)
    if match and re.search(r"number|random|pick", raw, re.IGNORECASE):
        low, high = sorted((int(match.group(1)), int(match.group(2))))
        return f"Your number: {random.randint(low, high)}"
    match = re.search(r"(?:choose|pick|select|decide)(?:\s+one)?\s*(?:between|from|of|:)\s*(.+)",
                      raw, re.IGNORECASE)
    if match:
        options = [o.strip(" ?.!") for o in re.split(r",|\bor\b|\band\b", match.group(1))]
        options = [o for o in options if o]
        if len(options) >= 2:
            return f"I choose: {random.choice(options)}"
    return None


def days_until(raw):
    """'days until 2026-12-31' -> '102 days until 2026-12-31.'"""
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if not match or not re.search(r"\b(until|till|left|countdown|days)\b", raw, re.IGNORECASE):
        return None
    try:
        target = date(*(int(g) for g in match.groups()))
    except ValueError:
        return "That doesn't look like a valid date. Use the format YYYY-MM-DD."
    diff = (target - date.today()).days
    if diff > 1:
        return f"{diff} days until {target.isoformat()}."
    if diff == 1:
        return f"Tomorrow! ({target.isoformat()})"
    if diff == 0:
        return "That's today!"
    return f"{target.isoformat()} was {abs(diff)} days ago."


_HELLO_RE = re.compile(r"(?:hello|hi|greetings?|good morning)\s+in\s+([a-z]+)")


def hello_in(clean):
    match = _HELLO_RE.search(clean)
    if not match:
        return None
    language = match.group(1)
    if language in kb.LANG_HELLO:
        return f"In {language.title()}: {kb.LANG_HELLO[language]}"
    known = ", ".join(sorted(k.title() for k in kb.LANG_HELLO)[:8])
    return f"I don't know {language.title()} yet. I can greet in: {known}, and more."


def acronym(clean):
    """'what does html stand for' -> 'HTML stands for HyperText Markup Language.'"""
    if not re.search(r"stands? for|full form|abbreviation|acronym|expand", clean):
        return None
    for word in clean.split():
        if word in kb.ACRONYMS:
            return f"{word.upper()} stands for {kb.ACRONYMS[word]}."
    return None


# ==========================================================================
# 6. To-do parsing and city detection
# ==========================================================================

_TODO = r"(?:to-?do|to do|tasks?)"


def parse_todo(raw):
    """Returns (action, argument) or None. Actions: add, list, done, remove, clear."""
    text = raw.strip()
    if re.search(rf"\bclear\s+(?:my\s+|all\s+)?{_TODO}", text, re.IGNORECASE):
        return ("clear", None)
    match = re.search(rf"\b(?:add|put|write)\s+(.+?)\s+(?:to|on|in|into)\s+(?:my\s+)?{_TODO}(?:\s+list)?\b",
                      text, re.IGNORECASE)
    if match:
        return ("add", match.group(1))
    match = re.match(rf"\s*{_TODO}\s*(?:add|:)\s*(.+)", text, re.IGNORECASE)
    if match:
        return ("add", match.group(1))
    match = re.match(r"\s*remind me to\s+(.+)", text, re.IGNORECASE)
    if match:
        return ("add", match.group(1))
    match = re.match(r"\s*mark\s+(?:task\s+)?(\d+)\s+(?:as\s+)?done\s*$", text, re.IGNORECASE)
    if match:
        return ("done", match.group(1))
    match = re.match(r"\s*(done|complete|completed|finished|finish|remove|delete)\s+(?:with\s+)?"
                     r"(?:task\s+|to-?do\s+|number\s+|#)?(\d+)\s*$", text, re.IGNORECASE)
    if match:
        action = "remove" if match.group(1).lower() in ("remove", "delete") else "done"
        return (action, match.group(2))
    if re.search(rf"\b(?:show|list|view|see|display|what(?:'s| is)?)\b.*\b{_TODO}\b", text, re.IGNORECASE) \
            or re.match(rf"\s*(?:my\s+){_TODO}(?:\s+list)?\s*[?.!]*\s*$", text, re.IGNORECASE) \
            or re.match(rf"\s*{_TODO}(?:\s+list)?\s*$", text, re.IGNORECASE):
        return ("list", None)
    return None


_SET_CITY_RE = re.compile(r"\b(?:i live in|i stay in|my city is|set my city to)\s+([A-Za-z][A-Za-z .'-]{1,40}?)\s*[.!]*$",
                          re.IGNORECASE)
_WEATHER_CITY_RE = re.compile(r"\b(?:in|at|for|of)\s+([A-Za-z][A-Za-z .'-]{1,40}?)\s*[?.!]*$")
_TRAILING = {"today", "tomorrow", "now", "tonight", "please", "currently", "right", "outside"}


def find_set_city(raw):
    match = _SET_CITY_RE.search(raw.strip())
    return match.group(1).strip().title() if match else None


def extract_city(raw):
    """'weather in Lahore today' -> 'Lahore'."""
    match = _WEATHER_CITY_RE.search(raw.strip())
    if not match:
        return None
    words = match.group(1).split()
    while words and words[-1].lower() in _TRAILING:
        words.pop()
    if words and words[-1].lower() == "right":
        words.pop()
    return " ".join(words).title() if words else None


# ==========================================================================
# 7. Sentiment (tiny word-list model)
# ==========================================================================

def sentiment(clean):
    """> 0 positive, < 0 negative, 0 neutral. Handles 'not good'."""
    words = clean.split()
    score = 0
    for i, word in enumerate(words):
        value = 1 if word in kb.POSITIVE else -1 if word in kb.NEGATIVE else 0
        if value and i > 0 and words[i - 1] in ("not", "no", "never", "dont"):
            value = -value
        score += value
    return score


# ==========================================================================
# 8. Memory (saved to a small JSON file on the user's computer)
# ==========================================================================

class Memory:
    """Remembers the user's name, city and to-do list between sessions."""

    def __init__(self, path=None):
        self.path = path
        self.data = self._empty()
        self.load()

    @staticmethod
    def _empty():
        return {"name": None, "city": None, "todos": [], "visits": 0, "last_visit": None}

    def load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                stored = json.load(f)
        except (OSError, ValueError):
            return
        if not isinstance(stored, dict):
            return
        for key in ("name", "city", "last_visit"):
            if isinstance(stored.get(key), str):
                self.data[key] = stored[key]
        if isinstance(stored.get("visits"), int):
            self.data["visits"] = stored["visits"]
        if isinstance(stored.get("todos"), list):
            self.data["todos"] = [
                {"text": str(t.get("text", ""))[:120], "done": bool(t.get("done"))}
                for t in stored["todos"] if isinstance(t, dict) and t.get("text")]

    def save(self):
        if not self.path:
            return
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except OSError:
            pass

    def forget(self):
        self.data = self._empty()
        if self.path and os.path.exists(self.path):
            try:
                os.remove(self.path)
            except OSError:
                pass


# ==========================================================================
# 9. Online skills (weather + Wikipedia). Both fail gracefully offline.
# ==========================================================================

USER_AGENT = "ByteChatbot/3.0 (student project; DecodeLabs internship)"


def http_get_json(url, timeout=6):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def weather_report(city):
    """Current weather from the free Open-Meteo API (no key). None if city unknown."""
    query = urllib.parse.quote(city)
    geo = http_get_json("https://geocoding-api.open-meteo.com/v1/search"
                        f"?name={query}&count=1&language=en&format=json")
    results = geo.get("results")
    if not results:
        return None
    place = results[0]
    data = http_get_json(
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={place['latitude']}&longitude={place['longitude']}"
        "&current=temperature_2m,apparent_temperature,relative_humidity_2m,"
        "weather_code,wind_speed_10m"
        "&daily=temperature_2m_max,temperature_2m_min&timezone=auto&forecast_days=1")
    now, day = data["current"], data["daily"]
    where = place.get("name", city) + (f", {place['country']}" if place.get("country") else "")
    condition = kb.WMO_CODES.get(now.get("weather_code"), "Unknown conditions")
    return (f"{where}: {condition}, {now['temperature_2m']}\u00b0C "
            f"(feels like {now['apparent_temperature']}\u00b0C). "
            f"Humidity {now['relative_humidity_2m']}%, wind {now['wind_speed_10m']} km/h. "
            f"Today: high {day['temperature_2m_max'][0]}\u00b0C, low {day['temperature_2m_min'][0]}\u00b0C.")


_WIKI_RE = re.compile(
    r"^\s*(?:who\s+(?:is|was|were|are)|what\s+(?:is|are|was|were)|tell me about|"
    r"wiki(?:pedia)?|search(?:\s+for)?|define|explain)\s+(?:an?\s+|the\s+)?(.+?)\s*[?.!]*\s*$",
    re.IGNORECASE)


def wiki_topic(raw):
    """'who is Alan Turing?' -> 'Alan Turing'. None if it isn't a lookup question."""
    match = _WIKI_RE.match(raw)
    if not match:
        return None
    topic = match.group(1).strip()
    if not 1 <= len(topic) <= 60 or re.search(r"\b(you|your|me|my|i)\b", topic, re.IGNORECASE):
        return None
    return topic


def shorten(text, limit=320):
    """Keep whole sentences up to about `limit` characters."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out = ""
    for sentence in sentences:
        if out and len(out) + len(sentence) > limit:
            break
        out = f"{out} {sentence}".strip()
    return out if len(out) <= limit + 80 else out[:limit].rstrip() + "..."


def wiki_summary(topic):
    """(title, short summary) from Wikipedia, or None if nothing was found."""
    query = urllib.parse.quote(topic)
    found = http_get_json("https://en.wikipedia.org/w/api.php?action=opensearch"
                          f"&search={query}&limit=1&namespace=0&format=json")
    titles = found[1] if len(found) > 1 else []
    if not titles:
        return None
    title = titles[0]
    page = http_get_json("https://en.wikipedia.org/api/rest_v1/page/summary/"
                         + urllib.parse.quote(title.replace(" ", "_"), safe=""))
    extract = (page.get("extract") or "").strip()
    return (title, shorten(extract)) if extract else None
