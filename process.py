from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Nithiwat/mdeberta-v3-base_claim-detection")
model = AutoModelForSequenceClassification.from_pretrained("Nithiwat/mdeberta-v3-base_claim-detection")

# Function to classify text
def classify_claim(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    # print(probabilities)
    # These might not be right
    is_claim = probabilities[:, 1] > probabilities[:, 0]  # Check if claim probability is higher
    return "Claim" if is_claim else "Non-Claim"

# Test cases
claims = [
    "Climate change is a significant global issue.",
    "Smoking causes lung cancer.",
    "Eating carrots improves your vision at night.",
    "Vaccines can prevent certain diseases.",
    "Regular exercise can reduce the risk of heart disease.",
    "Artificial intelligence will surpass human intelligence in the next decade.",
    "Photosynthesis is a process used by plants to create oxygen.",
    "Einstein won a Nobel Prize for his theory of relativity.",
    "Sleeping less than seven hours a night can harm your health.",
    "The universe is constantly expanding.",
    "The Great Pyramid of Giza was built by aliens.",
    "Chocolate is good for your health.",
    "The world's population will reach 10 billion by 2050.",
    "Coffee is the world's most widely consumed psychoactive drug.",
    "Dogs are better pets than cats."
]
non_claims = [
    "I had pasta for lunch.",
    "The Eiffel Tower is in Paris.",
    "Yesterday was the longest day of the year.",
    "Water boils at 100 degrees Celsius.",
    "Shakespeare wrote 'War and Peace'.",  # False statement but not a claim.
    "The Great Wall of China can be seen from space.",  # A common misconception.
    "The sun revolves around the Earth.",  # Incorrect, but presented as a fact.
    "The capital of Australia is Sydney.",  # Incorrect, but not a claim.
    "The Internet was invented in the 1960s.",
    "Thomas Edison invented the light bulb.",
    "The Titanic was unsinkable.",  # Historical inaccuracy.
    "Humans can survive for weeks without water.",  # Factual inaccuracy.
    "The human brain is the most complex object in the universe.",  # Factual statement.
    "The Mona Lisa was painted by Leonardo da Vinci.",
    "There are more stars in the universe than grains of sand on Earth."
]

test_texts = claims + non_claims
# Classify each test case
for text in test_texts:
    result = classify_claim(text)
    print(f"'{text}' is classified as: {result}")
