"""
N-Atlas Multilingual Translation Engine
Translates English text derived from ASL gloves into Nigerian languages (Yoruba, Igbo, Hausa).
Features a dual-engine architecture:
  1. Fast offline lexicon with high-fidelity tonal diacritics.
  2. N-Atlas / NLLB Deep Seq2Seq Neural Transformer via Hugging Face.
"""
import sys
import logging
from typing import Dict, Optional, Tuple, Any
import threading
import config

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TranslatorEngine")

class NAtlasTranslator:
    # High-accuracy bilingual repository of everyday phrases & words with authentic tone markings
    LEXICON = {
        # --- Complete Idiomatic Sentences ---
        "hello, greetings!": {
            "Yoruba": ("Ẹ n lẹ o, ẹ ku àbọ̀!", "Ehn-leh oh, eh koo ah-boh"),
            "Igbo": ("Ndewo, nnoo!", "N-deh-woh, n-noh"),
            "Hausa": ("Sannu da zuwa!", "Sahn-noo dah zoo-wah")
        },
        "hello, how are you doing?": {
            "Yoruba": ("Ẹ n lẹ o, bawo ni gbogbo nkan?", "Ehn-leh oh, bah-woh nee gbog-boh n-kahn?"),
            "Igbo": ("Ndewo, kedu ka ị mere?", "N-deh-woh, keh-doo kah ee meh-reh?"),
            "Hausa": ("Sannu, yaya kake yau?", "Sahn-noo, yah-yah kah-keh yow?")
        },
        "how are you doing?": {
            "Yoruba": ("Bawo ni o ṣe wa?", "Bah-woh nee oh shay wah?"),
            "Igbo": ("Kedu ka ị mere?", "Keh-doo kah ee meh-reh?"),
            "Hausa": ("Yaya kake?", "Yah-yah kah-keh?")
        },
        "thank you very much.": {
            "Yoruba": ("Ẹ ṣe pupọ.", "Eh shay poo-poh"),
            "Igbo": ("Dalu nke ukwuu.", "Dah-loo n-keh oo-kwoo"),
            "Hausa": ("Nagode sosai.", "Nah-goh-deh soh-sigh")
        },
        "yes, that is correct.": {
            "Yoruba": ("Bẹ́ẹ̀ni, bẹ́ẹ̀ gẹ́gẹ́ ni.", "Beh-eh-nee, beh-eh geh-geh nee"),
            "Igbo": ("Ee, ọ bụ eziokwu.", "Eh, oh boo eh-zee-oh-kwoo"),
            "Hausa": ("I, haka ne.", "Ee, hah-kah neh")
        },
        "no, thank you.": {
            "Yoruba": ("Rárá, ẹ ṣe.", "Rah-rah, eh shay"),
            "Igbo": ("Mba, dalu.", "M-bah, dah-loo"),
            "Hausa": ("A'a, nagode.", "Ah-ah, nah-goh-deh")
        },
        "please, i need help.": {
            "Yoruba": ("Ẹ jọ̀wọ́, ẹ gbà mí o!", "Eh joh-woh, eh gbah mee oh!"),
            "Igbo": ("Biko, nyere m aka!", "Bee-koh, nyeh-reh m ah-kah!"),
            "Hausa": ("Don Allah, taimake ni!", "Dohn Ahl-lah, tie-mah-keh nee!")
        },
        "please, help me immediately.": {
            "Yoruba": ("Ẹ tètè gbà mí o!", "Eh teh-teh gbah mee oh!"),
            "Igbo": ("Biko nyere m aka ozugbo!", "Bee-koh nyeh-reh m ah-kah oh-zoog-boh!"),
            "Hausa": ("Don Allah taimake ni yanzu!", "Dohn Ahl-lah tie-mah-keh nee yahn-zoo!")
        },
        "i would like some water.": {
            "Yoruba": ("Mo fẹ́ mu omi.", "Moh feh moo oh-mee"),
            "Igbo": ("Achọrọ m mmiri.", "Ah-choh-roh m mmee-ree"),
            "Hausa": ("Ina son ruwa.", "Ee-nah sohn roo-wah")
        },
        "please give me some water.": {
            "Yoruba": ("Ẹ jọ̀wọ́, ẹ fún mi ní omi.", "Eh joh-woh, eh foon mee nee oh-mee"),
            "Igbo": ("Biko nye m mmiri.", "Bee-koh nyeh m mmee-ree"),
            "Hausa": ("Don Allah bani ruwa.", "Dohn Ahl-lah bah-nee roo-wah")
        },
        "i need food.": {
            "Yoruba": ("Mo fẹ́ jẹ oúnjẹ.", "Moh feh jeh oh-oon-jeh"),
            "Igbo": ("Achọrọ m nri.", "Ah-choh-roh m n-ree"),
            "Hausa": ("Ina son abinci.", "Ee-nah sohn ah-been-chee")
        },
        "please give me some food.": {
            "Yoruba": ("Ẹ jọ̀wọ́, ẹ fún mi ní oúnjẹ.", "Eh joh-woh, eh foon mee nee oh-oon-jeh"),
            "Igbo": ("Biko nye m nri.", "Bee-koh nyeh m n-ree"),
            "Hausa": ("Don Allah bani abinci.", "Dohn Ahl-lah bah-nee ah-been-chee")
        },
        "i need to see a doctor.": {
            "Yoruba": ("Mo fẹ́ rí dókítà.", "Moh feh ree doh-kee-tah"),
            "Igbo": ("Achọrọ m ịhụ dọkịta.", "Ah-choh-roh m ee-hoo doh-kee-tah"),
            "Hausa": ("Ina buƙatar ganin likita.", "Ee-nah boo-kah-tahr gah-neen lee-kee-tah")
        },
        "please, i need to see a doctor.": {
            "Yoruba": ("Ẹ jọ̀wọ́, mo fẹ́ rí dókítà.", "Eh joh-woh, moh feh ree doh-kee-tah"),
            "Igbo": ("Biko, achọrọ m ịhụ dọkịta.", "Bee-koh, ah-choh-roh m ee-hoo doh-kee-tah"),
            "Hausa": ("Don Allah, ina so in ga likita.", "Dohn ahl-lah, ee-nah soh een gah lee-kee-tah")
        },
        "medical emergency, please call a doctor!": {
            "Yoruba": ("Pàjáwìrì! Ẹ pe dókítà wá!", "Pah-jah-wee-ree! Eh peh doh-kee-tah wah!"),
            "Igbo": ("Ihe mberede ahụike! Biko kpọọ dọkịta!", "Ee-heh m-beh-reh-deh ah-hoo-ee-keh!"),
            "Hausa": ("Gaggawa! Don Allah a kira likita!", "Gahg-gah-wah! Dohn ahl-lah ah kee-rah lee-kee-tah!")
        },
        "emergency, please call a doctor immediately!": {
            "Yoruba": ("Pàjáwìrì! Ẹ tètè pe dókítà wá!", "Pah-jah-wee-ree! Eh teh-teh peh doh-kee-tah wah!"),
            "Igbo": ("Ihe mberede! Biko kpọọ dọkịta ozugbo!", "Ee-heh m-beh-reh-deh! Bee-koh kpoh doh-kee-tah oh-zoog-boh!"),
            "Hausa": ("Gaggawa! Don Allah a kira likita nan da nan!", "Gahg-gah-wah! Dohn ahl-lah ah kee-rah lee-kee-tah nahn dah nahn!")
        },
        "emergency, please help and call a doctor!": {
            "Yoruba": ("Pàjáwìrì! Ẹ gbà mí, ẹ pe dókítà wá!", "Pah-jah-wee-ree! Eh gbah mee, eh peh doh-kee-tah wah!"),
            "Igbo": ("Ihe mberede! Biko nyere m aka, kpọọ dọkịta!", "Ee-heh m-beh-reh-deh! Bee-koh nyeh-reh m ah-kah, kpoh doh-kee-tah!"),
            "Hausa": ("Gaggawa! Don Allah a taimake ni, a kira likita!", "Gahg-gah-wah! Dohn ahl-lah ah tie-mah-keh nee, ah kee-rah lee-kee-tah!")
        },
        "i need water and food.": {
            "Yoruba": ("Mo fẹ́ omi àti oúnjẹ.", "Moh feh oh-mee ah-tee oh-oon-jeh"),
            "Igbo": ("Achọrọ m mmiri na nri.", "Ah-choh-roh m mmee-ree nah n-ree"),
            "Hausa": ("Ina son ruwa da abinci.", "Ee-nah sohn roo-wah dah ah-been-chee")
        },
        "please give me some water and food.": {
            "Yoruba": ("Ẹ jọ̀wọ́, ẹ fún mi ní omi àti oúnjẹ.", "Eh joh-woh, eh foon mee nee oh-mee ah-tee oh-oon-jeh"),
            "Igbo": ("Biko nye m mmiri na nri.", "Bee-koh nyeh m mmee-ree nah n-ree"),
            "Hausa": ("Don Allah bani ruwa da abinci.", "Dohn ahl-lah bah-nee roo-wah dah ah-been-chee")
        },
        "help, please give me some water.": {
            "Yoruba": ("Ẹ gbà mí, mo fẹ́ mu omi o!", "Eh gbah mee, moh feh moo oh-mee oh!"),
            "Igbo": ("Nyere m aka, achọrọ m mmiri!", "Nyeh-reh m ah-kah, ah-choh-roh m mmee-ree!"),
            "Hausa": ("Taimake ni, ina son ruwa!", "Tie-mah-keh nee, ee-nah sohn roo-wah!")
        },
        "please help me, i need water.": {
            "Yoruba": ("Ẹ gbà mí, mo fẹ́ mu omi o!", "Eh gbah mee, moh feh moo oh-mee oh!"),
            "Igbo": ("Nyere m aka, achọrọ m mmiri!", "Nyeh-reh m ah-kah, ah-choh-roh m mmee-ree!"),
            "Hausa": ("Taimake ni, ina son ruwa!", "Tie-mah-keh nee, ee-nah sohn roo-wah!")
        },
        "help, please give me some food.": {
            "Yoruba": ("Ẹ gbà mí, mo fẹ́ jẹ oúnjẹ o!", "Eh gbah mee, moh feh jeh oh-oon-jeh oh!"),
            "Igbo": ("Nyere m aka, achọrọ m nri!", "Nyeh-reh m ah-kah, ah-choh-roh m n-ree!"),
            "Hausa": ("Taimake ni, ina son abinci!", "Tie-mah-keh nee, ee-nah sohn ah-been-chee!")
        },
        "please help me, i need food.": {
            "Yoruba": ("Ẹ gbà mí, mo fẹ́ jẹ oúnjẹ o!", "Eh gbah mee, moh feh jeh oh-oon-jeh oh!"),
            "Igbo": ("Nyere m aka, achọrọ m nri!", "Nyeh-reh m ah-kah, ah-choh-roh m n-ree!"),
            "Hausa": ("Taimake ni, ina son abinci!", "Tie-mah-keh nee, ee-nah sohn ah-been-chee!")
        },
        "hello, thank you very much.": {
            "Yoruba": ("Ẹ n lẹ o, ẹ ṣe pupọ.", "Eh n leh oh, eh shay poo-poh"),
            "Igbo": ("Ndewo, dalu nke ukwuu.", "N-deh-woh, dah-loo n-keh oo-kwoo"),
            "Hausa": ("Sannu, nagode sosai.", "Sahn-noo, nah-goh-deh soh-sigh")
        },
        "hello, please.": {
            "Yoruba": ("Ẹ n lẹ o, ẹ jọ̀wọ́.", "Eh n leh oh, eh joh-woh"),
            "Igbo": ("Ndewo, biko.", "N-deh-woh, bee-koh"),
            "Hausa": ("Sannu, don Allah.", "Sahn-noo, dohn ahl-lah")
        },
        "yes, please.": {
            "Yoruba": ("Bẹ́ẹ̀ni, ẹ jọ̀wọ́.", "Beh-eh-nee, eh joh-woh"),
            "Igbo": ("Ee, biko.", "Eh, bee-koh"),
            "Hausa": ("I, don Allah.", "Ee, dohn ahl-lah")
        },
        "yes, thank you.": {
            "Yoruba": ("Bẹ́ẹ̀ni, ẹ ṣe pupọ.", "Beh-eh-nee, eh shay poo-poh"),
            "Igbo": ("Ee, dalu nke ukwuu.", "Eh, dah-loo n-keh oo-kwoo"),
            "Hausa": ("I, nagode sosai.", "Ee, nah-goh-deh soh-sigh")
        },
        "no, please.": {
            "Yoruba": ("Rárá, ẹ jọ̀wọ́.", "Rah-rah, eh joh-woh"),
            "Igbo": ("Mba, biko.", "M-bah, bee-koh"),
            "Hausa": ("A'a, don Allah.", "Ah-ah, dohn ahl-lah")
        },
        "thank you, please.": {
            "Yoruba": ("Ẹ ṣe, ẹ jọ̀wọ́.", "Eh shay, eh joh-woh"),
            "Igbo": ("Dalu, biko.", "Dah-loo, bee-koh"),
            "Hausa": ("Nagode, don Allah.", "Nah-goh-deh, dohn ahl-lah")
        },
        "this is my family.": {
            "Yoruba": ("Èyí ni ẹbí mi.", "Eh-yee nee eh-bee mee"),
            "Igbo": ("Nke a bụ ezinụlọ m.", "N-keh ah boo eh-zee-noo-loh m"),
            "Hausa": ("Wannan shine iyalina.", "Wahn-nahn shee-neh ee-yah-lee-nah")
        },
        "i love you.": {
            "Yoruba": ("Mo nífẹ̀ẹ́ rẹ.", "Moh nee-feh-eh reh"),
            "Igbo": ("A hụrụ m gị n'anya.", "Ah hoo-roo m gee n-ahn-yah"),
            "Hausa": ("Ina son ku.", "Ee-nah sohn koo")
        },
        "i love my family.": {
            "Yoruba": ("Mo nífẹ̀ẹ́ ẹbí mi.", "Moh nee-feh-eh eh-bee mee"),
            "Igbo": ("A hụrụ m ezinụlọ m n'anya.", "Ah hoo-roo m eh-zee-noo-loh m n-ahn-yah"),
            "Hausa": ("Ina son iyalina.", "Ee-nah sohn ee-yah-lee-nah")
        },
        "peace and harmony.": {
            "Yoruba": ("Àlàáfíà àti ìrẹ́pọ̀.", "Ah-lah-fee-ah ah-tee ee-reh-poh"),
            "Igbo": ("Udo na ịdị n'otu.", "Oo-doh nah ee-dee n-oh-too"),
            "Hausa": ("Zaman lafiya da jituwa.", "Zah-mahn lah-fee-yah dah jee-too-wah")
        },
        "peace to my family.": {
            "Yoruba": ("Àlàáfíà fún ẹbí mi.", "Ah-lah-fee-ah foon eh-bee mee"),
            "Igbo": ("Udo dịrị ezinụlọ m.", "Oo-doh dee-ree eh-zee-noo-loh m"),
            "Hausa": ("Zaman lafiya ga iyalina.", "Zah-mahn lah-fee-yah gah ee-yah-lee-nah")
        },
        "peace and greetings.": {
            "Yoruba": ("Àlàáfíà àti kíkí.", "Ah-lah-fee-ah ah-tee kee-kee"),
            "Igbo": ("Udo na ekele.", "Oo-doh nah eh-keh-leh"),
            "Hausa": ("Zaman lafiya da gaisuwa.", "Zah-mahn lah-fee-yah dah gye-soo-wah")
        },
        "good morning to you.": {
            "Yoruba": ("Ẹ káàárọ̀ o.", "Eh kah-ah-roh oh"),
            "Igbo": ("Ụtụtụ ọma.", "Oo-too-too oh-mah"),
            "Hausa": ("Ina kwana.", "Ee-nah kwah-nah")
        },
        "good morning, how are you doing?": {
            "Yoruba": ("Ẹ káàárọ̀, bawo ni gbogbo nkan?", "Eh kah-ah-roh, bah-woh nee gbog-boh n-kahn?"),
            "Igbo": ("Ụtụtụ ọma, kedu ka ị mere?", "Oo-too-too oh-mah, keh-doo kah ee meh-reh?"),
            "Hausa": ("Ina kwana, yaya kake yau?", "Ee-nah kwah-nah, yah-yah kah-keh yow?")
        },
        "good morning, hello!": {
            "Yoruba": ("Ẹ káàárọ̀, ẹ n lẹ o!", "Eh kah-ah-roh, eh n leh oh!"),
            "Igbo": ("Ụtụtụ ọma, ndewo!", "Oo-too-too oh-mah, n-deh-woh!"),
            "Hausa": ("Ina kwana, sannu!", "Ee-nah kwah-nah, sahn-noo!")
        },
        "goodbye, see you later.": {
            "Yoruba": ("Ó dàbọ̀, a ó ríra nígbà mìíràn.", "Oh dah-boh, ah oh ree-rah"),
            "Igbo": ("Ka ọ dị, ka emesia.", "Kah oh dee, kah eh-meh-see-ah"),
            "Hausa": ("Sai an jima, sai mun sake haɗuwa.", "Sigh ahn jee-mah")
        },
        "goodbye, thank you very much.": {
            "Yoruba": ("Ó dàbọ̀, ẹ ṣe pupọ.", "Oh dah-boh, eh shay poo-poh"),
            "Igbo": ("Ka ọ dị, dalu nke ukwuu.", "Kah oh dee, dah-loo n-keh oo-kwoo"),
            "Hausa": ("Sai an jima, nagode sosai.", "Sigh ahn jee-mah, nah-goh-deh soh-sigh")
        },

        # --- Individual Vocabulary Signs & Short Phrases ---
        "hello": {
            "Yoruba": ("Ẹ n lẹ o", "Ehn-leh oh"),
            "Igbo": ("Ndewo", "N-deh-woh"),
            "Hausa": ("Sannu", "Sahn-noo")
        },
        "please": {
            "Yoruba": ("Ẹ jọ̀wọ́", "Eh joh-woh"),
            "Igbo": ("Biko", "Bee-koh"),
            "Hausa": ("Don Allah", "Dohn Ahl-lah")
        },
        "water": {
            "Yoruba": ("Omi", "Oh-mee"),
            "Igbo": ("Mmiri", "Mmee-ree"),
            "Hausa": ("Ruwa", "Roo-wah")
        },
        "food": {
            "Yoruba": ("Oúnjẹ", "Oh-oon-jeh"),
            "Igbo": ("Nri", "N-ree"),
            "Hausa": ("Abinci", "Ah-been-chee")
        },
        "help": {
            "Yoruba": ("Ìrànwọ́", "Ee-rahn-woh"),
            "Igbo": ("Enyemaka", "Eh-nyeh-mah-kah"),
            "Hausa": ("Taimako", "Tie-mah-koh")
        },
        "doctor": {
            "Yoruba": ("Dókítà", "Doh-kee-tah"),
            "Igbo": ("Dọkịta", "Doh-kee-tah"),
            "Hausa": ("Likita", "Lee-kee-tah")
        },
        "thank you": {
            "Yoruba": ("Ẹ ṣe", "Eh shay"),
            "Igbo": ("Dalu", "Dah-loo"),
            "Hausa": ("Nagode", "Nah-goh-deh")
        },
        "yes": {
            "Yoruba": ("Bẹ́ẹ̀ni", "Beh-eh-nee"),
            "Igbo": ("Ee", "Eh"),
            "Hausa": ("I", "Ee")
        },
        "no": {
            "Yoruba": ("Rárá", "Rah-rah"),
            "Igbo": ("Mba", "M-bah"),
            "Hausa": ("A'a", "Ah-ah")
        },
        "how are you": {
            "Yoruba": ("Bawo ni", "Bah-woh nee"),
            "Igbo": ("Kedu", "Keh-doo"),
            "Hausa": ("Yaya kake", "Yah-yah kah-keh")
        },
        "good morning": {
            "Yoruba": ("Ẹ káàárọ̀", "Eh kah-ah-roh"),
            "Igbo": ("Ụtụtụ ọma", "Oo-too-too oh-mah"),
            "Hausa": ("Ina kwana", "Ee-nah kwah-nah")
        },
        "goodbye": {
            "Yoruba": ("Ó dàbọ̀", "Oh dah-boh"),
            "Igbo": ("Ka ọ dị", "Kah oh dee"),
            "Hausa": ("Sai an jima", "Sigh ahn jee-mah")
        },
        "peace": {
            "Yoruba": ("Àlàáfíà", "Ah-lah-fee-ah"),
            "Igbo": ("Udo", "Oo-doh"),
            "Hausa": ("Zaman lafiya", "Zah-mahn lah-fee-yah")
        },
        "family": {
            "Yoruba": ("Ẹbí", "Eh-bee"),
            "Igbo": ("Ezinụlọ", "Eh-zee-noo-loh"),
            "Hausa": ("Iyali", "Ee-yah-lee")
        },
        "love": {
            "Yoruba": ("Ìfẹ́", "Ee-feh"),
            "Igbo": ("Ịhụnanya", "Ee-hoo-nahn-yah"),
            "Hausa": ("Ƙauna", "Kow-nah")
        },
        "and": {
            "Yoruba": ("àti", "ah-tee"),
            "Igbo": ("na", "nah"),
            "Hausa": ("da", "dah")
        },
        "i need": {
            "Yoruba": ("Mo nílò", "Moh nee-loh"),
            "Igbo": ("Achọrọ m", "Ah-choh-roh m"),
            "Hausa": ("Ina buƙatar", "Ee-nah boo-kah-tahr")
        },
        "i want": {
            "Yoruba": ("Mo fẹ́", "Moh feh"),
            "Igbo": ("Achọrọ m", "Ah-choh-roh m"),
            "Hausa": ("Ina son", "Ee-nah sohn")
        },
        "call a doctor": {
            "Yoruba": ("Ẹ pe dókítà wá", "Eh peh doh-kee-tah wah"),
            "Igbo": ("Kpọọ dọkịta", "Kpoh doh-kee-tah"),
            "Hausa": ("A kira likita", "Ah kee-rah lee-kee-tah")
        },
        "give me": {
            "Yoruba": ("ẹ fún mi ní", "eh foon mee nee"),
            "Igbo": ("nye m", "nyeh m"),
            "Hausa": ("bani", "bah-nee")
        }
    }

    def __init__(self):
        self.tokenizer = None
        self.neural_model = None
        self.neural_loading = False
        self.neural_ready = False
        self.inference_client = None
        self._lock = threading.Lock()

        # Initialize InferenceClient if Hugging Face token is available
        self._init_inference_client()

    def _init_inference_client(self, explicit_token: Optional[str] = None) -> bool:
        """Initialize Hugging Face InferenceClient if token is available."""
        import os
        from pathlib import Path
        token = explicit_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        
        # Check local saved token in data directory
        token_file = config.DATA_DIR / "hf_token.txt"
        if not token and token_file.exists():
            try:
                token = token_file.read_text(encoding="utf-8").strip()
            except Exception:
                pass

        if not token:
            cache_token_path = Path.home() / ".cache" / "huggingface" / "token"
            if cache_token_path.exists():
                try:
                    token = cache_token_path.read_text().strip()
                except Exception:
                    pass

        if token:
            try:
                from huggingface_hub import InferenceClient
                self.inference_client = InferenceClient(model=config.NATLAS_MODEL_NAME, token=token)
                self.neural_ready = True
                logger.info(f"N-Atlas Hugging Face Cloud Inference Client ready ({config.NATLAS_MODEL_NAME}).")
                return True
            except Exception as e:
                logger.debug(f"Could not initialize HF InferenceClient: {e}")
                return False
        return False

    def set_hf_token(self, token: str) -> bool:
        """Save and activate a Hugging Face token for online N-ATLaS inference."""
        token_clean = token.strip()
        if not token_clean:
            return False
        token_file = config.DATA_DIR / "hf_token.txt"
        try:
            token_file.write_text(token_clean, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not persist HF token: {e}")
        return self._init_inference_client(token_clean)

    def load_neural_model_async(self, model_name: str = config.NATLAS_MODEL_NAME):
        """Asynchronously load N-Atlas Llama-3 model (local weights or API) without stalling UI."""
        if self.neural_loading:
            return

        def _loader():
            self.neural_loading = True
            logger.info(f"Loading neural translation model '{model_name}'...")
            
            # Check if weights exist in local directory
            local_dir = getattr(config, "NATLAS_LOCAL_DIR", None)
            load_path = str(local_dir) if (local_dir and (local_dir / "config.json").exists()) else model_name

            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForCausalLM
                
                tokenizer = AutoTokenizer.from_pretrained(load_path)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                
                model = AutoModelForCausalLM.from_pretrained(
                    load_path,
                    torch_dtype=dtype,
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True
                )
                with self._lock:
                    self.tokenizer = tokenizer
                    self.neural_model = model
                    self.neural_ready = True
                logger.info(f"Neural model '{model_name}' loaded successfully into {device.upper()} memory!")
            except Exception as e:
                logger.warning(f"Local neural load note: {e}.")
                self._init_inference_client()
                if self.inference_client:
                    logger.info("Using Hugging Face Serverless Cloud API for N-ATLaS.")
                else:
                    logger.info("Using N-Atlas offline tonal dictionary as fallback engine.")
            finally:
                self.neural_loading = False

        thread = threading.Thread(target=_loader, daemon=True)
        thread.start()

    def get_model_details(self) -> Dict[str, Any]:
        """Return metadata summary about the N-Atlas translation architecture."""
        engine_str = "N-Atlas Authentic Tonal Lexicon"
        if self.neural_model:
            engine_str = "N-Atlas 8B Local Neural LLM"
        elif self.inference_client:
            engine_str = "N-Atlas Hugging Face Cloud Inference (8B LLM)"

        return {
            "model_title": "N-ATLaS Multilingual LLM (NCAIR Nigeria)",
            "hf_repo": config.NATLAS_MODEL_NAME,
            "architecture": "Meta Llama-3-8B Causal LM fine-tuned on 392M Nigerian tokens",
            "supported_languages": ["Yoruba", "Igbo", "Hausa", "English"],
            "lexicon_entries": len(self.LEXICON),
            "neural_checkpoint": config.NATLAS_MODEL_NAME,
            "neural_ready": self.neural_ready,
            "neural_loading": self.neural_loading,
            "active_engine": engine_str,
            "tonal_diacritics": "Full Yoruba/Igbo/Hausa tone marks (´, `, subdots: ẹ, ọ, ṣ, ụ, ị, ṅ)"
        }

    def _prompt_natlas(self, english_text: str, target_lang: str) -> Optional[str]:
        """Query N-ATLaS model with official Llama-3 chat template for translation."""
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are N-ATLaS, an Afro-centric AI language assistant built by NCAIR Nigeria. "
            f"Translate the user's sentence into {target_lang}. Return ONLY the direct translation "
            f"with accurate tonal diacritics, without conversational filler.<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Translate to {target_lang}: \"{english_text}\"<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        # 1. Try local loaded model
        if self.neural_model and self.tokenizer:
            try:
                import torch
                inputs = self.tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                outputs = self.neural_model.generate(
                    **inputs,
                    max_new_tokens=48,
                    temperature=0.1,
                    repetition_penalty=1.12
                )
                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "assistant" in decoded:
                    decoded = decoded.split("assistant")[-1].strip()
                return decoded.strip().replace('"', '')
            except Exception as e:
                logger.debug(f"Local N-ATLaS inference error: {e}")

        # 2. Try Hugging Face InferenceClient
        if self.inference_client:
            try:
                response = self.inference_client.text_generation(
                    prompt,
                    max_new_tokens=48,
                    temperature=0.1,
                    repetition_penalty=1.12,
                    stop=["<|eot_id|>"]
                )
                output = response.strip().replace("<|eot_id|>", "").replace('"', '')
                if output:
                    return output
            except Exception as e:
                logger.debug(f"InferenceClient error: {e}")

        return None

    @staticmethod
    def _normalize_text(s: str) -> str:
        import re
        t = s.lower().strip()
        t = re.sub(r"[^\w\s]", " ", t)
        return " ".join(t.split())

    def translate(self, text: str, target_lang: str) -> Dict[str, str]:
        """
        Translate input English text to chosen target language.
        
        Args:
            text: Input English sentence.
            target_lang: 'English', 'Yoruba', 'Igbo', or 'Hausa'.
            
        Returns:
            Dictionary with {source, target_lang, translation, phonetic, engine}.
        """
        text_clean = text.strip()
        if not text_clean:
            return {"source": "", "target_lang": target_lang, "translation": "", "phonetic": "", "engine": "None"}

        if target_lang.lower() == "english":
            return {
                "source": text_clean,
                "target_lang": "English",
                "translation": text_clean,
                "phonetic": text_clean,
                "engine": "Passthrough"
            }

        norm_input = self._normalize_text(text_clean)

        # 1. Exact Lexicon Match on normalized sentence string
        for key in self.LEXICON:
            if self._normalize_text(key) == norm_input:
                trans_tuple = self.LEXICON[key].get(target_lang)
                if trans_tuple:
                    return {
                        "source": text_clean,
                        "target_lang": target_lang,
                        "translation": trans_tuple[0],
                        "phonetic": trans_tuple[1],
                        "engine": "N-Atlas Lexicon"
                    }

        # 2. Check if deep N-ATLaS neural model or cloud API is ready
        if self.neural_ready:
            try:
                neural_pred = self._prompt_natlas(text_clean, target_lang)
                if neural_pred:
                    engine_label = "N-ATLaS 8B Local" if self.neural_model else "N-ATLaS Cloud API"
                    return {
                        "source": text_clean,
                        "target_lang": target_lang,
                        "translation": neural_pred,
                        "phonetic": neural_pred,
                        "engine": engine_label
                    }
            except Exception as e:
                logger.debug(f"Neural inference note: {e}")

        # 3. Greedy Multi-Word Chunking (Matches sub-phrases and individual words)
        words = norm_input.split()
        if not words:
            return {"source": text_clean, "target_lang": target_lang, "translation": text_clean, "phonetic": text_clean, "engine": "Fallback"}

        i = 0
        translated_chunks = []
        phonetic_chunks = []

        while i < len(words):
            matched = False
            # Try matching longest n-gram first (up to 4 words down to 1)
            for window_size in range(min(4, len(words) - i), 0, -1):
                candidate_phrase = " ".join(words[i : i + window_size])
                for k, v in self.LEXICON.items():
                    if self._normalize_text(k) == candidate_phrase and target_lang in v:
                        trans_text = v[target_lang][0].rstrip(".!?:;,")
                        trans_phon = v[target_lang][1].rstrip(".!?:;,")
                        translated_chunks.append(trans_text)
                        phonetic_chunks.append(trans_phon)
                        i += window_size
                        matched = True
                        break
                if matched:
                    break

            if not matched:
                # Keep word as-is if truly unrecognized
                translated_chunks.append(words[i].capitalize())
                phonetic_chunks.append(words[i])
                i += 1

        if translated_chunks:
            assembled = " ".join(translated_chunks)
            if assembled:
                assembled = assembled[0].upper() + assembled[1:]
                if not assembled.endswith((".", "!", "?")):
                    assembled += "."
            assembled_phon = " ".join(phonetic_chunks)
        else:
            assembled = text_clean
            assembled_phon = text_clean

        return {
            "source": text_clean,
            "target_lang": target_lang,
            "translation": assembled,
            "phonetic": assembled_phon,
            "engine": "N-Atlas Heuristic Lexicon"
        }

if __name__ == "__main__":
    t = NAtlasTranslator()
    test_phrases = ["Hello!", "Please give me some water.", "Thank you very much."]
    for lang in ["Yoruba", "Igbo", "Hausa"]:
        print(f"\n--- Target: {lang} ---")
        for p in test_phrases:
            res = t.translate(p, lang)
            print(f"EN: {p} -> {res['translation']} (Phonetic: {res['phonetic']}) [{res['engine']}]")
