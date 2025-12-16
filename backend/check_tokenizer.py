import os

TOKENIZER_PATH = "/Users/anurag/Desktop/youtube_summarizer/backend/models/t5_summarizer/tokenizer.json"

print("Checking tokenizer at:", TOKENIZER_PATH)

if not os.path.exists(TOKENIZER_PATH):
    print("❌ File does NOT exist!")
    exit()

try:
    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        content = f.read(200)
    print("\nFirst 200 characters of tokenizer.json:\n")
    print(content)
except Exception as e:
    print("\n❌ Error reading file:", e)
