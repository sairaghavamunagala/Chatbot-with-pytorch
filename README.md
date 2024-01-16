# Chatbot-with-pytorch
## Steps to Follow
- Theory + NLP concepts (Stemming, Tokenization, bag of words)
- Create training data
- PyTorch model and training
- Save/load model and implement the chat

## Overview
This is a simple AI interview chatbot implemented in Python using PyTorch. The chatbot engages in a conversation with the user, simulating an interview for various roles, such as Python Developer, Machine Learning Engineer, Software Engineer, Data Engineer, and Data Analyst. The chatbot asks questions based on predefined intents and responds accordingly.

## Getting Started
Clone the repository:
   ```bash
   git clone https://github.com/sairaghavamunagala/Chatbot-with-pytorch.git
   cd Chatbot-with-pytorch
```
## To install project requirements:
- Run the following command
```bash
pip install -r requirements.txt
```

## Start the chat
```bash 
python chat.py
```


# Code Structure
- model.py: Contains the definition of the NeuralNet class used for the chatbot.
- nltk_utils.py: Includes utility functions for tokenization and bag-of-words representation.
- intents.json: Stores predefined intents for different interview questions.
- data.pth: Serialized data file containing information about the neural network model.
# Usage
- The chatbot starts by loading the necessary data and the pre-trained neural network model.
- It prompts the user to choose a role from the list of available roles.
- The chatbot then inquires about the level of the position (junior, mid, or senior).
- The conversation proceeds with the chatbot asking questions based on predefined intents for the selected role and level.
- The user responds to the questions until they decide to quit the interview.
# Interview Process
The chatbot follows a structured interview process, asking questions related to the specified role and level. It ensures that each question is asked only once by keeping track of asked tags.

# Quitting the Interview
At any point during the interview, the user can type "quit" to end the interview session. The chatbot will acknowledge the termination and conclude the conversation.

### Tools Used

* VSCode - IDE of choice
* Pop!_OS - OS of choice

## Compatibility

This project has been tested on ```python 3.10.6```. Make sure you have the compatible Python version installed.

# Note

## Training Data

The chatbot has been trained on a small set of dummy intents to showcase its basic functionality. The training data includes generic responses to common interview questions and is not tailored for any specific domain.

If you are interested in contributing or enhancing the chatbot with real-world data, you are welcome to collaborate! We encourage you to fork the repository, make improvements, and submit pull requests.

## How to Contribute

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes and commit them (`git commit -am 'Add feature/improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a pull request

Thank you for your interest in improving this project!
