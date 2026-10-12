"""
Sentence Builder & ASL Gloss-to-English Grammar Engine
Transforms raw streams of ASL signs and fingerspelled letters into coherent English sentences.
"""
import time
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SentenceBuilder")

class SentenceBuilder:
    # Rule-based idiomatic ASL gloss sequence restructuring
    GLOSS_PHRASE_PATTERNS = {
        ("HELLO",): "Hello, greetings!",
        ("HELLO", "HOW_ARE_YOU"): "Hello, how are you doing?",
        ("HOW_ARE_YOU",): "How are you doing?",
        ("THANK_YOU",): "Thank you very much.",
        ("PLEASE",): "Please.",
        ("YES",): "Yes, that is correct.",
        ("NO",): "No, thank you.",
        ("HELP",): "Please, I need help.",
        ("HELP", "PLEASE"): "Please, help me immediately.",
        ("WATER",): "I would like some water.",
        ("WATER", "PLEASE"): "Please give me some water.",
        ("FOOD",): "I need food.",
        ("FOOD", "PLEASE"): "Please give me some food.",
        ("DOCTOR",): "I need to see a doctor.",
        ("DOCTOR", "HELP"): "Medical emergency, please call a doctor!",
        ("FAMILY",): "This is my family.",
        ("I_LOVE_YOU",): "I love you.",
        ("GOOD_MORNING",): "Good morning to you.",
        ("PEACE",): "Peace and harmony.",
        ("GOODBYE",): "Goodbye, see you later."
    }

    # Order-independent multi-sign patterns (frozenset matching)
    GLOSS_SET_PATTERNS = {
        # 3-sign combinations
        frozenset(["HELLO", "HOW_ARE_YOU", "THANK_YOU"]): "Hello, how are you? Thank you very much.",
        frozenset(["PLEASE", "WATER", "FOOD"]): "Please give me some water and food.",
        frozenset(["PLEASE", "HELP", "DOCTOR"]): "Emergency, please help and call a doctor!",
        frozenset(["PLEASE", "HELP", "WATER"]): "Please help me, I need water.",
        frozenset(["PLEASE", "HELP", "FOOD"]): "Please help me, I need food.",
        frozenset(["HELP", "DOCTOR", "PLEASE"]): "Emergency, please call a doctor immediately!",
        
        # 2-sign combinations (order-independent)
        frozenset(["WATER", "PLEASE"]): "Please give me some water.",
        frozenset(["FOOD", "PLEASE"]): "Please give me some food.",
        frozenset(["HELP", "PLEASE"]): "Please, help me immediately.",
        frozenset(["DOCTOR", "HELP"]): "Medical emergency, please call a doctor!",
        frozenset(["DOCTOR", "PLEASE"]): "Please, I need to see a doctor.",
        frozenset(["HELLO", "HOW_ARE_YOU"]): "Hello, how are you doing?",
        frozenset(["HELLO", "THANK_YOU"]): "Hello, thank you very much.",
        frozenset(["HELLO", "PLEASE"]): "Hello, please.",
        frozenset(["YES", "PLEASE"]): "Yes, please.",
        frozenset(["NO", "THANK_YOU"]): "No, thank you.",
        frozenset(["YES", "THANK_YOU"]): "Yes, thank you.",
        frozenset(["NO", "PLEASE"]): "No, please.",
        frozenset(["THANK_YOU", "PLEASE"]): "Thank you, please.",
        frozenset(["WATER", "FOOD"]): "I need water and food.",
        frozenset(["HELP", "WATER"]): "Help, please give me some water.",
        frozenset(["HELP", "FOOD"]): "Help, please give me some food.",
        frozenset(["FAMILY", "I_LOVE_YOU"]): "I love my family.",
        frozenset(["PEACE", "FAMILY"]): "Peace to my family.",
        frozenset(["PEACE", "HELLO"]): "Peace and greetings.",
        frozenset(["GOOD_MORNING", "HOW_ARE_YOU"]): "Good morning, how are you doing?",
        frozenset(["GOODBYE", "THANK_YOU"]): "Goodbye, thank you very much.",
        frozenset(["GOOD_MORNING", "HELLO"]): "Good morning, hello!",

        # 1-sign standalone meanings
        frozenset(["HELLO"]): "Hello, greetings!",
        frozenset(["HOW_ARE_YOU"]): "How are you doing?",
        frozenset(["THANK_YOU"]): "Thank you very much.",
        frozenset(["PLEASE"]): "Please.",
        frozenset(["YES"]): "Yes, that is correct.",
        frozenset(["NO"]): "No, thank you.",
        frozenset(["HELP"]): "Please, I need help.",
        frozenset(["WATER"]): "I would like some water.",
        frozenset(["FOOD"]): "I need food.",
        frozenset(["DOCTOR"]): "I need to see a doctor.",
        frozenset(["FAMILY"]): "This is my family.",
        frozenset(["I_LOVE_YOU"]): "I love you.",
        frozenset(["GOOD_MORNING"]): "Good morning to you.",
        frozenset(["PEACE"]): "Peace and harmony.",
        frozenset(["GOODBYE"]): "Goodbye, see you later."
    }

    def __init__(self, fingerspell_timeout_sec: float = 2.5):
        self.tokens: List[str] = []
        self.spelling_buffer: List[str] = []
        self.last_letter_time: float = 0.0
        self.last_activity_time: float = time.time()
        self.fingerspell_timeout = fingerspell_timeout_sec

    def add_token(self, token: str) -> str:
        """
        Add new recognized sign. Automatically detects if it is a single-letter fingerspelling
        or an entire word/sign. Prevents accidental consecutive word repetition.
        
        Returns the current draft sentence string.
        """
        token = token.strip().upper()
        if not token or token in ["IDLE", "REST", "NONE", "UNKNOWN"]:
            return self.get_draft_sentence()

        now = time.time()
        self.last_activity_time = now

        # If token is a single letter (A-Z)
        if len(token) == 1 and token.isalpha():
            # Prevent accidental double-tap bounce of the same letter within 400ms
            if self.spelling_buffer and self.spelling_buffer[-1] == token and (now - self.last_letter_time < 0.40):
                logger.debug(f"Suppressed rapid duplicate letter bounce: '{token}'")
                return self.get_draft_sentence()
            self.spelling_buffer.append(token)
            self.last_letter_time = now
        else:
            # If we were previously fingerspelling a word, flush it first
            self._flush_spelling_buffer()
            # Anti-repetition guard: prevent adding the exact same whole word consecutively
            if self.tokens and self.tokens[-1] == token:
                logger.info(f"SentenceBuilder: Suppressed consecutive repeated word '{token}'")
                return self.get_draft_sentence()
            self.tokens.append(token)

        return self.get_draft_sentence()

    def _flush_spelling_buffer(self):
        """Combine accumulated fingerspelled letters into a single word token."""
        if self.spelling_buffer:
            word = "".join(self.spelling_buffer).capitalize()
            self.tokens.append(word)
            self.spelling_buffer.clear()

    def check_timeout(self):
        """Called periodically; flushes fingerspelled letters if user paused signing."""
        if self.spelling_buffer and (time.time() - self.last_letter_time > self.fingerspell_timeout):
            self._flush_spelling_buffer()

    def get_tokens(self) -> List[str]:
        tokens_copy = list(self.tokens)
        if self.spelling_buffer:
            tokens_copy.append("".join(self.spelling_buffer))
        return tokens_copy

    def get_draft_sentence(self) -> str:
        """Construct the best English sentence from currently signed tokens."""
        self.check_timeout()
        tokens = self.get_tokens()
        if not tokens:
            return ""

        # 1. Exact tuple match (preserves explicit sequence)
        tuple_tokens = tuple(tokens)
        if tuple_tokens in self.GLOSS_PHRASE_PATTERNS:
            return self.GLOSS_PHRASE_PATTERNS[tuple_tokens]

        # 2. Exact set match (order-independent multi-sign phrases)
        set_tokens = frozenset(tokens)
        if set_tokens in self.GLOSS_SET_PATTERNS:
            return self.GLOSS_SET_PATTERNS[set_tokens]

        # 3. Sub-phrase decomposition (e.g. ["HELLO", "WATER", "PLEASE"])
        if len(tokens) >= 2:
            last_pair = frozenset(tokens[-2:])
            if last_pair in self.GLOSS_SET_PATTERNS:
                lead_words = [t.replace("_", " ").capitalize() for t in tokens[:-2]]
                lead_str = ", ".join(lead_words)
                pair_str = self.GLOSS_SET_PATTERNS[last_pair]
                return f"{lead_str}, {pair_str}" if lead_str else pair_str

            first_pair = frozenset(tokens[:2])
            if first_pair in self.GLOSS_SET_PATTERNS:
                pair_str = self.GLOSS_SET_PATTERNS[first_pair].rstrip(".!?:")
                tail_words = [t.replace("_", " ").capitalize() for t in tokens[2:]]
                tail_str = ", ".join(tail_words)
                return f"{pair_str}, {tail_str}." if tail_str else f"{pair_str}."

        # 4. Smart multi-word synthesis for arbitrary signed combinations
        clean_words = []
        for t in tokens:
            cleaned = t.replace("_", " ").capitalize()
            clean_words.append(cleaned)

        if len(clean_words) == 2:
            w1, w2 = clean_words[0], clean_words[1]
            if w1.upper() in ["PLEASE", "HELLO"] or w2.upper() in ["PLEASE"]:
                sentence = f"{w1}, {w2}."
            else:
                sentence = f"{w1} and {w2}."
        else:
            sentence = " ".join(clean_words)
            if sentence and sentence[-1] not in [".", "?", "!"]:
                sentence += "."

        return sentence

    def backspace(self) -> str:
        """Remove last entered character or token."""
        if self.spelling_buffer:
            self.spelling_buffer.pop()
        elif self.tokens:
            self.tokens.pop()
        return self.get_draft_sentence()

    def clear(self):
        """Reset sentence buffer."""
        self.tokens.clear()
        self.spelling_buffer.clear()
        self.last_activity_time = time.time()

    def get_idle_seconds(self) -> float:
        """Return elapsed seconds since last token or letter activity."""
        return max(0.0, time.time() - self.last_activity_time)

    def check_sentence_timeout(self, timeout_sec: float) -> bool:
        """
        Check if user has been inactive for longer than timeout_sec while having an active sentence.
        If timeout is exceeded, clears buffer and returns True.
        """
        if (self.tokens or self.spelling_buffer) and timeout_sec > 0:
            if self.get_idle_seconds() >= timeout_sec:
                logger.info(f"SentenceBuilder: Auto-cleared sentence buffer after {timeout_sec}s idle timeout.")
                self.clear()
                return True
        return False

    def commit_sentence(self) -> str:
        """Return finalized sentence and clear buffer."""
        self._flush_spelling_buffer()
        final_str = self.get_draft_sentence()
        self.clear()
        return final_str
