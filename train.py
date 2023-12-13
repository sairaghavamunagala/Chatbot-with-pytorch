import json
from nltk_utils import tokenize,stem,bag_of_words
import numpy as np

with open('intents.json','r') as f:
    intents=json.load(f)



all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

ignore_words=['.','?',"!",","]
valid_words=[stem(w) for w in all_words if w not in ignore_words]
all_words=sorted(set(valid_words))
tags=sorted(set(tags))

X_train=list()
Y_train=list()

for (pattern_sentece,tags) in xy:
    bag=bag_of_words(pattern_sentece,all_words
                     )
    X_train.append(bag)
    label=tags.index(tags)
    Y_train.append(label)

X_train=np.array(X_train)
Y_train=np.array(Y_train)

class ChatDataSet:
    def __init_(self):
        self.n_samples=len(X_train)
        self.x_data=len(X_train)
        self.y_data=len(Y_train)

    def __getitem__(self,index):
        return self.x_data[index],self.y_data

    def __len__(self):
        return self.n_samples

# HYPERPARAMETERS
BATCH_SIZE=8
dataset=ChatDataSet()
train_loader = DataLoader(dataset=dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=0)