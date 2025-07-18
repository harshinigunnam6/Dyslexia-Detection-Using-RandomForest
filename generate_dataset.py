import pandas as pd
import random

def generate_dyslexic_sentence(base):
    replacements = {
        "ph": "f", "tion": "shun", "oo": "u", "ee": "e", "ie": "ei",
        "read": "reed", "write": "rite", "school": "skool", "friend": "frend",
        "jumped": "jumpt", "ran": "runned", "went": "goed", "they": "thay",
        "because": "becaus", "restaurant": "restrant", "ice cream": "icecrem"
    }
    words = base.split()
    new_words = []
    for word in words:
        if random.random() < 0.3:
            for k, v in replacements.items():
                if k in word:
                    word = word.replace(k, v)
                    break
        new_words.append(word)
    return " ".join(new_words)

base_sentences = [
    "He went to the park and played with the dog.",
    "She likes to read books every day.",
    "There is a cat on the roof and it is looking down.",
    "I know how to write my name and I rarely misspell it.",
    "The sun is shining bright today.",
    "My friends give me toys and sweets.",
    "She ran fast and jumped high.",
    "I go to school every morning.",
    "They were eating in the restaurant.",
    "I want to eat ice cream now because it is hot."
]

data = []
for _ in range(25):  # 25 * 10 = 250 of each
    for sentence in base_sentences:
        data.append((generate_dyslexic_sentence(sentence), "dyslexic"))
        data.append((sentence, "non-dyslexic"))

random.shuffle(data)
df = pd.DataFrame(data, columns=["text", "label"])
df.to_csv("dyslexia_dataset_500.csv", index=False)
print("✅ Dataset saved as dyslexia_dataset_500.csv")
