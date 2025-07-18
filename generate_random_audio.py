from gtts import gTTS
import random

# List of dictation sentences
sentences = [
    "Technology has changed the way we learn and communicate.",
    "Artificial intelligence is shaping the future of healthcare.",
    "Reading books improves vocabulary and imagination.",
    "Children learn best when they are encouraged and praised.",
    "The sky was painted with beautiful shades of orange during sunset.",
    "She walked quickly through the crowded hallway.",
    "Science experiments help students understand complex concepts.",
    "Listening carefully is the first step to learning better.",
    "A curious mind is always eager to explore new ideas.",
    "The classroom was silent except for the sound of pencils scratching paper."
]

# Pick one randomly
sentence = random.choice(sentences)
tts = gTTS(sentence)
tts.save("dictation_audio.mp3")

# Save the sentence to a text file for reference
with open("dictation_text.txt", "w") as f:
    f.write(sentence)

print("✅ Audio & text saved. Sentence:", sentence)
