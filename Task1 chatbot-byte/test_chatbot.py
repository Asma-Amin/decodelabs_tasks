"""Unit tests for Byte v3  -  run with:  python -m unittest test_chatbot -v"""

import os
import tempfile
import unittest
from unittest import mock

import knowledge as kb
import skills
from chatbot import Chatbot, normalize


class TestSkills(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize("  WHAT'S UP?? "), "whats up")
        self.assertEqual(normalize("how r u"), "how are you")

    def test_math(self):
        self.assertEqual(skills.extract_math("what is 5 plus 3?"), "5 + 3")
        self.assertIsNone(skills.extract_math("hello there"))
        self.assertIsNone(skills.extract_math("days until 2026-12-31"))
        self.assertEqual(skills.safe_eval("2 + 3 * 4"), 14)
        with self.assertRaises(ValueError):
            skills.safe_eval("__import__('os').system('echo hi')")
        with self.assertRaises(ValueError):
            skills.safe_eval("9 ** 9 ** 9")

    def test_units(self):
        self.assertEqual(skills.convert_units("5 km to miles"), "5 km = 3.1069 miles")
        self.assertEqual(skills.convert_units("100 f to c"), "100 f = 37.78 c")
        self.assertEqual(skills.convert_units("0 c to k"), "0 c = 273.15 k")
        self.assertIn("different things", skills.convert_units("5 kg to km"))
        self.assertIsNone(skills.convert_units("5 apples to pears"))

    def test_world_time(self):
        self.assertIn("(UTC+9)", skills.world_time("what time is it in tokyo"))
        self.assertIn("(UTC+5:30)", skills.world_time("time in delhi"))
        self.assertIsNone(skills.world_time("what time is it"))

    def test_text_tools(self):
        self.assertEqual(skills.text_tool("reverse hello"), "olleh")
        self.assertEqual(skills.text_tool("spell cat"), "C-A-T")
        self.assertIn("Yes", skills.text_tool("is Level a palindrome"))
        self.assertIn("No", skills.text_tool("palindrome python"))
        self.assertIsNone(skills.text_tool("is it raining"))

    def test_password(self):
        pw = skills.make_password(20)
        self.assertEqual(len(pw), 20)
        self.assertTrue(any(c.isdigit() for c in pw) and any(c.isupper() for c in pw))
        self.assertEqual(len(skills.make_password(3)), 8)   # minimum length

    def test_random_tool(self):
        n = int(skills.random_tool("pick a number between 5 and 9").split(": ")[1])
        self.assertTrue(5 <= n <= 9)
        self.assertIn(skills.random_tool("choose between tea, coffee or juice").split(": ")[1],
                      ["tea", "coffee", "juice"])

    def test_todo_parsing(self):
        self.assertEqual(skills.parse_todo("add buy milk to my todo list"), ("add", "buy milk"))
        self.assertEqual(skills.parse_todo("remind me to call mom"), ("add", "call mom"))
        self.assertEqual(skills.parse_todo("done 2"), ("done", "2"))
        self.assertEqual(skills.parse_todo("remove 1"), ("remove", "1"))
        self.assertEqual(skills.parse_todo("show my todo list"), ("list", None))
        self.assertEqual(skills.parse_todo("clear my tasks"), ("clear", None))
        self.assertIsNone(skills.parse_todo("hello there"))

    def test_city_helpers(self):
        self.assertEqual(skills.extract_city("weather in Lahore today"), "Lahore")
        self.assertEqual(skills.find_set_city("I live in New York"), "New York")

    def test_sentiment(self):
        self.assertGreater(skills.sentiment("this is awesome"), 0)
        self.assertLess(skills.sentiment("i hate this"), 0)
        self.assertLess(skills.sentiment("not good"), 0)

    def test_misc(self):
        self.assertIn("Hola", skills.hello_in("say hello in spanish"))
        self.assertIn("HyperText", skills.acronym("what does html stand for"))
        self.assertIn("days until", skills.days_until("days until 2999-01-01"))
        self.assertEqual(skills.wiki_topic("who is Alan Turing?"), "Alan Turing")
        self.assertIsNone(skills.wiki_topic("who are you"))


class TestChatbot(unittest.TestCase):
    def setUp(self):
        self.bot = Chatbot()

    # ---- the project requirements ------------------------------------------
    def test_greeting(self):
        reply = self.bot.respond("hello").lower()
        self.assertTrue(any(w in reply for w in ("hello", "hey", "hi")))

    def test_exit_stops_loop(self):
        self.assertTrue(self.bot.running)
        self.bot.respond("bye")
        self.assertFalse(self.bot.running)

    def test_exit_needs_short_message(self):
        self.bot.respond("how do I exit vim without saving my file")
        self.assertTrue(self.bot.running)

    # ---- understanding ------------------------------------------------------
    def test_roman_urdu(self):
        self.assertIn("Walaikum assalam", self.bot.respond("Assalamualaikum"))
        self.assertIn("theek", self.bot.respond("kya haal hai"))

    def test_typos_and_slang(self):
        self.bot.respond("tell me jokess")
        self.assertIn("joke", self.bot.last_debug)
        self.bot.respond("wat tiem is it")
        self.assertIn("time", self.bot.last_debug)

    def test_times_is_not_time(self):
        self.bot.respond("how many times did it fail")
        self.assertNotIn("time(", self.bot.last_debug)

    def test_multi_intent(self):
        self.assertIn("right now", self.bot.respond("hi, what time is it?"))

    def test_followups(self):
        first = self.bot.respond("tell me a joke")
        self.assertNotEqual(first, self.bot.respond("another one"))

    def test_no_repeat_until_deck_used(self):
        seen = {self.bot.respond("joke") for _ in range(len(kb.JOKES))}
        self.assertEqual(len(seen), len(kb.JOKES))

    def test_mood_needs_context(self):
        self.bot.respond("how are you")
        self.assertIn("Glad", self.bot.respond("good"))
        self.assertNotIn("Glad", Chatbot().respond("good"))

    def test_sentiment_fallback(self):
        self.assertIn(self.bot.respond("i hate this"), kb.EMPATHY)
        self.assertIn(self.bot.respond("this is awesome"), kb.CHEER)

    def test_fallback_and_stats(self):
        self.bot.respond("xqzv blorp")
        self.assertEqual(self.bot.stats["fallbacks"], 1)
        self.assertIn("Not understood: 1", self.bot.respond("/stats"))

    def test_empty_input(self):
        self.assertTrue(self.bot.respond("   "))
        self.assertTrue(self.bot.running)

    # ---- tools ---------------------------------------------------------------
    def test_tools(self):
        self.assertEqual(self.bot.respond("calculate 12 * 7"), "12 * 7 = 84")
        self.assertIn("zero", self.bot.respond("10 / 0"))
        self.assertIn("3.1069 miles", self.bot.respond("convert 5 km to miles"))
        self.assertEqual(self.bot.respond("reverse hello"), "olleh")
        self.assertIn("HyperText", self.bot.respond("what does HTML stand for"))
        self.assertIn("Hola", self.bot.respond("say hello in spanish"))
        self.assertIn("UTC+9", self.bot.respond("time in tokyo"))

    def test_explain(self):
        self.assertIn("machines", self.bot.respond("what is AI").lower())
        self.assertIn("calls itself", self.bot.respond("explain recursion"))

    # ---- memory ---------------------------------------------------------------
    def test_remembers_name(self):
        self.bot.respond("my name is asma")
        self.assertEqual(self.bot.name, "Asma")
        self.assertIn("Asma", self.bot.respond("what is my name"))

    def test_todo_flow(self):
        self.bot.respond("add buy milk to my todo list")
        self.bot.respond("remind me to call mom")
        self.assertIn("2. [ ] call mom", self.bot.respond("show my todo list"))
        self.bot.respond("done 1")
        self.assertIn("1. [x] buy milk", self.bot.respond("show my todo list"))
        self.bot.respond("remove 2")
        self.assertIn("no task number 9", self.bot.respond("done 9"))
        self.assertIn("Cleared 1", self.bot.respond("clear my todo list"))

    def test_memory_persists_and_forget(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mem.json")
            first = Chatbot(memory_path=path)
            first.respond("my name is sara")
            first.respond("i live in Lahore")
            first.respond("add study python to my todo list")
            second = Chatbot(memory_path=path)             # a "new session"
            self.assertEqual(second.name, "Sara")
            self.assertEqual(second.memory.data["city"], "Lahore")
            self.assertIn("study python", second.respond("show my todo list"))
            second.respond("forget me")
            self.assertIn("erased", second.respond("yes"))
            self.assertFalse(os.path.exists(path))
            self.assertIsNone(second.name)

    def test_forget_can_be_cancelled(self):
        self.bot.respond("my name is sara")
        self.bot.respond("forget me")
        self.assertIn("keep", self.bot.respond("no"))
        self.assertEqual(self.bot.name, "Sara")

    # ---- games ------------------------------------------------------------------
    def test_guess_game(self):
        self.bot.respond("guess the number")
        self.bot.pending["target"] = 42
        self.assertIn("higher", self.bot.respond("10"))
        self.assertIn("lower", self.bot.respond("90"))
        self.assertIn("Correct", self.bot.respond("42"))
        self.assertIsNone(self.bot.pending)

    def test_guess_game_ends_after_max_tries(self):
        self.bot.respond("guess the number")
        self.bot.pending["target"] = 50
        replies = [self.bot.respond("1") for _ in range(8)]
        self.assertIn("Out of tries", replies[-1])

    def test_riddle(self):
        self.bot.respond("riddle")
        self.bot.pending["answers"] = ["coin"]
        self.assertIn("Not quite", self.bot.respond("banana"))
        self.assertIn("Correct", self.bot.respond("a coin"))
        self.bot.respond("riddle")
        self.assertIn("answer", self.bot.respond("give up").lower())

    def test_quiz(self):
        self.assertIn("Q1/5", self.bot.respond("quiz me"))
        for _ in range(5):
            question = self.bot.pending["qs"][self.bot.pending["i"]]
            letter = "abcd"[question["options"].index(question["answer"])]
            reply = self.bot.respond(letter)
        self.assertIn("5/5", reply)
        self.assertIsNone(self.bot.pending)

    def test_quiz_rejects_bad_answer_and_stop(self):
        self.bot.respond("quiz me")
        self.assertIn("Answer with A, B, C or D", self.bot.respond("banana"))
        self.assertIn("stopped", self.bot.respond("stop"))
        self.assertIsNone(self.bot.pending)

    def test_rps(self):
        self.bot.respond("rock paper scissors")
        self.assertIn("Score", self.bot.respond("rock"))
        self.assertIn("Final score", self.bot.respond("stop"))

    def test_bye_exits_even_during_game(self):
        self.bot.respond("quiz me")
        self.bot.respond("bye")
        self.assertFalse(self.bot.running)

    # ---- online skills (network is mocked: tests never touch the internet) -------
    def test_weather_success(self):
        geo = {"results": [{"name": "Lahore", "country": "Pakistan", "latitude": 31.5, "longitude": 74.3}]}
        forecast = {"current": {"temperature_2m": 30.1, "apparent_temperature": 33.0,
                                "relative_humidity_2m": 60, "weather_code": 2, "wind_speed_10m": 9.5},
                    "daily": {"temperature_2m_max": [34.0], "temperature_2m_min": [25.0]}}
        with mock.patch("skills.http_get_json", side_effect=[geo, forecast]):
            reply = self.bot.respond("weather in Lahore")
        self.assertIn("Lahore, Pakistan: Partly cloudy, 30.1", reply)

    def test_weather_asks_city_then_answers(self):
        self.assertIn("Which city", self.bot.respond("weather"))
        with mock.patch("skills.http_get_json", return_value={}):
            self.assertIn("couldn't find a place", self.bot.respond("Atlantis"))

    def test_weather_uses_remembered_city(self):
        self.bot.respond("i live in Lahore")
        with mock.patch("skills.http_get_json", return_value={}) as fake:
            self.bot.respond("weather")
        self.assertTrue(fake.called)

    def test_wikipedia_success(self):
        opensearch = ["Alan Turing", ["Alan Turing"], [""], [""]]
        page = {"extract": "Alan Turing was an English mathematician. He is a father of computer science."}
        with mock.patch("skills.http_get_json", side_effect=[opensearch, page]):
            reply = self.bot.respond("who is Alan Turing?")
        self.assertIn("Alan Turing was an English mathematician", reply)
        self.assertIn("Wikipedia", reply)
        self.assertEqual(self.bot.stats["fallbacks"], 0)

    def test_offline_graceful(self):
        with mock.patch("skills.http_get_json", side_effect=OSError("no network")):
            self.assertIn("couldn't reach the internet", self.bot.respond("weather in Paris"))
            self.assertIn("couldn't reach the internet", self.bot.respond("who is Marie Curie"))
        offline = Chatbot(online=False)
        self.assertIn("offline", offline.respond("weather in Paris"))

    # ---- commands ------------------------------------------------------------------
    def test_commands(self):
        self.bot.respond("hello")
        self.assertIn("You: hello", self.bot.respond("/history"))
        self.assertIn("Unknown command", self.bot.respond("/nope"))
        self.assertIn("help tools", self.bot.respond("/help"))
        self.assertIn("anything saved", self.bot.respond("/memory"))


if __name__ == "__main__":
    unittest.main()
