import spacy
import random

# Load the small English model of spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Model 'en_core_web_sm' not found. Installing now...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Define intents with training phrases and responses
intents = {
    "greeting": {
        "examples": [
            "Hi",
            "Hello",
            "Hey there",
            "Good morning",
            "Good afternoon",
            "Hi there"
        ],
        "responses": [
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
            "Hey! I'm here to chat whenever you like."
        ]
    },
    "goodbye": {
        "examples": [
            "Bye",
            "Goodbye",
            "See you later",
            "Farewell",
            "Catch you later"
        ],
        "responses": [
            "Goodbye! Have a great day!",
            "See you later! Take care!",
            "Farewell! Hope to chat again soon."
        ]
    },
    "thanks": {
        "examples": [
            "Thanks",
            "Thank you",
            "Thanks a lot",
            "Thank you very much"
        ],
        "responses": [
            "You're welcome!",
            "No problem!",
            "Glad I could help!"
        ]
    },
    "how_are_you": {
        "examples": [
            "How are you?",
            "How's it going?",
            "How are you doing?",
            "What's up?"
        ],
        "responses": [
            "I'm just a bot, but I'm doing great! How about you?",
            "All systems normal! How can I assist you?",
            "I'm fine, thank you! What about you?"
        ]
    },
    "name": {
        "examples": [
            "What is your name?",
            "Who are you?",
            "Identify yourself",
            "Tell me your name"
        ],
        "responses": [
            "I'm a chatbot created using spaCy.",
            "I'm your friendly chatbot.",
            "You can call me Chatbot."
        ]
    }
}

FALLBACK_RESPONSES = [
    "I'm not sure I understand. Could you please rephrase?",
    "Interesting, tell me more.",
    "Let's talk about something else. What do you want to discuss?",
    "Can you please clarify that?"
]

def preprocess(text):
    # Lowercase and remove excess whitespace
    return text.strip().lower()

def find_intent(user_input):
    user_doc = nlp(user_input)
    best_intent = None
    best_score = 0.6  # threshold for similarity

    for intent, data in intents.items():
        for example in data["examples"]:
            example_doc = nlp(example)
            similarity = user_doc.similarity(example_doc)
            if similarity > best_score:
                best_score = similarity
                best_intent = intent

    return best_intent

def get_response(intent):
    if intent in intents:
        return random.choice(intents[intent]["responses"])
    else:
        return random.choice(FALLBACK_RESPONSES)

def main():
    print("Chatbot: Hello! You can start chatting with me. Type 'exit' or 'quit' to leave.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye! Have a nice day!")
            break
        if not user_input:
            continue
        intent = find_intent(user_input)
        response = get_response(intent)
        print("Chatbot:", response)

if __name__ == "__main__":
    main()

