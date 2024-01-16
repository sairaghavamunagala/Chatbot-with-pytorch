import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to('cpu')
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

print("I am AI chatbot here to interview you.")

available_roles = ["Python Developer", "Machine Learning Engineer", "Software Engineer", "Data Engineer", "Data Analyst"]
print("Available Roles: ", ", ".join(available_roles))

while True:
    role = input("Enter the Role: ")
    if role.lower() == "quit":
        print("Goodbye! Have a great day!")
        break

    print(f"Great! Let's talk about the {role} role.")

    print("Are you applying for a junior, mid, or senior position?")
    level = input("Level: ")

    if level.lower() == "quit":
        print("Interview session ended. Thank you for your time!")
        break

    print(f"Perfect! Let's discuss the {role} {level} position. Can you introduce yourself?")

  
    asked_tags = set()
    for _ in range(len(intents['intents'])):
        sentence = input("You: ")
        if sentence.lower() == "quit":
            print("Interview session ended. Thank you for your time!")
            break

        if asked_tags:  # Skip the first iteration
            sentence = tokenize(sentence)
            X = bag_of_words(sentence, all_words)
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X).to('cpu')

            output = model(X)
            _, predicted = torch.max(output, dim=1)

            tag = tags[predicted.item()]

            # Ensure asking only one question per tag
            if tag not in asked_tags:
                asked_tags.add(tag)
                for intent in intents['intents']:
                    if tag == intent["tag"]:
                        print(f"{bot_name}: {random.choice(intent['patterns'])}")
                        break

    print(f"{bot_name}: That concludes the interview. Thank you for your time!")