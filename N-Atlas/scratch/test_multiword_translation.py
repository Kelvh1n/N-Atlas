import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "software"))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from sentence_builder import SentenceBuilder
from translator import NAtlasTranslator

sb = SentenceBuilder()
t = NAtlasTranslator()

tests = [
    ['WATER'],
    ['WATER', 'PLEASE'],
    ['PLEASE', 'WATER'],
    ['HELP'],
    ['HELP', 'DOCTOR'],
    ['DOCTOR', 'HELP'],
    ['FOOD'],
    ['FOOD', 'PLEASE'],
    ['WATER', 'FOOD'],
    ['HELLO'],
    ['HELLO', 'HOW_ARE_YOU'],
    ['HELLO', 'THANK_YOU'],
    ['YES', 'PLEASE'],
    ['NO', 'THANK_YOU'],
    ['THANK_YOU', 'PLEASE'],
    ['PLEASE', 'HELP', 'DOCTOR'],
    ['HELLO', 'HOW_ARE_YOU', 'THANK_YOU'],
    ['WATER', 'FOOD', 'DOCTOR']
]

for tokens in tests:
    sb.clear()
    sentence = ""
    for tok in tokens:
        sentence = sb.add_token(tok)
    print(f"\nTOKENS: {tokens} => ENGLISH: \"{sentence}\"")
    for lang in ['Yoruba', 'Igbo', 'Hausa']:
        res = t.translate(sentence, lang)
        print(f"  [{lang:6s}] \"{res['translation']}\"  ({res['phonetic']}) [{res['engine']}]")
