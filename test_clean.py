"""
Test text cleaning and preprocessing edge cases.
"""
from pathlib import Path
import sys

# Ensure project root is available on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.preprocess import clean_text, combine_title_and_text

# 1. Standard text with URL, punctuation, and uppercase
sample1 = "BREAKING NEWS!!! Visit http://abc.com NOW!!!"
res1 = clean_text(sample1)
print(f"Sample 1: '{res1}'")
assert res1 == "breaking news visit"

# 2. None, empty, and whitespace strings
assert clean_text(None) == "", "None input failed!"
assert clean_text("") == "", "Empty string input failed!"
assert clean_text("   ") == "", "Whitespace string input failed!"

# 3. Wire service byline leakage removal
sample3 = "LONDON (Reuters) - Prime Minister announces new bill today."
res3 = clean_text(sample3, strip_wire_byline=True)
print(f"Sample 3: '{res3}'")
assert "reuters" not in res3
assert "prime minister announces new bill today" in res3

# 4. Negation preservation
sample4 = "The suspect was not guilty and had no weapon."
res4 = clean_text(sample4, preserve_negations=True)
print(f"Sample 4: '{res4}'")
assert "not" in res4, "Negation 'not' was stripped!"
assert "no" in res4, "Negation 'no' was stripped!"

# 5. Title-body combination
combined = combine_title_and_text("Shocking Revelation", "Full article body here.")
print(f"Sample 5 (Combined): '{combined}'")
assert combined == "Shocking Revelation - Full article body here."

print("\nAll clean_text tests passed successfully!")