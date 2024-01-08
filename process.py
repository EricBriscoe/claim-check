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
    "Today the death toll from coronavirus rose above 28,000, and state governors across the country grappled with the difficult question of when, how and to what extent they could begin trying -- trying to reestablish some kind of normalcy.",
    "What he did not do was provide any evidence that his administration has taken the steps needed for that to actually happen safely in a way that won't trigger new outbreaks and cost more lives.",
    "And that's because once again today, the President spent a large part of what has become a substitute for his political rallies -- which supposed to be the coronavirus task force briefing -- he spent it boasting of his accomplishments -- accomplishments he's yet to actually accomplish -- and deflecting blame.",
    "It's also some of the country's top business executives.",
    "One such step is widespread testing, which the president has both derided and claimed that is currently happening.",
    "They told the President that today.",
]
non_claims = [
    "It is scientists on the coronavirus task force, epidemiologists.",
    "Well today, the state governors also learned, once again, something about the President of the United States.",
    "And when he was asked about it at tonight's briefing, he first tried to claim credit where none is due, and then tried to put the responsibility elsewhere.",
    "And that's not just us saying so.",
    "He's eager, understandably, to reopen the country, as he so often says, and said so again today.",  # False statement but not a claim.
    "Keep in mind(ph) though there is not a widespread testing we will ultimately need to get back to business.",  # A common misconception.
    "There just isn't.",  # Incorrect, but presented as a fact.
    "First, though, the boasting.",  # Incorrect, but not a claim.
    "And good evening, everyone.",
]

test_texts = claims + non_claims
# Classify each test case
for text in test_texts:
    result = classify_claim(text)
    print(f"'{text}' is classified as: {result}")
