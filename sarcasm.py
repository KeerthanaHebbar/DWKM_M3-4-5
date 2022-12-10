import re,string

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense,Activation

import numpy
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence



#.....loding twitter text file
f = open('/home/soumya/.config/spyder/SarcasmDetection/train_dataset.txt', 'r')
train_rawtext = f.readlines()
f.close()

f = open('/home/soumya/.config/spyder/SarcasmDetection/test_dataset.txt', 'r')
test_rawtext = f.readlines()
f.close()

#.....removing newlines and tabs
train=[]
for i in range(len(train_rawtext)):
    train.append(re.sub('\s+',' ',train_rawtext[i]))
    
test=[]
for i in range(len(test_rawtext)):
    test.append(re.sub('\s+',' ',test_rawtext[i]))

#.....removing #words, @words, puntuations    
def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text

def strip_all_entities(text):
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

train_text=[]
for t in train:
    train_text.append(strip_all_entities(strip_links(t)))
 
test_text=[]
for t in test:
    test_text.append(strip_all_entities(strip_links(t)))

 
#......tokenizing words
train_lebels=[]
train_lines=[]

for i in range(len(train_text)):
    l=train_text[i].split()
    l1=l[0:2]
    train_lebels.append(l[1])
    l2 = [x for x in l if x not in l1]
    train_lines.append(l2)
    
test_lebels=[]
test_lines=[]

for i in range(len(test_text)):
    l=test_text[i].split()
    l1=l[0:2]
    test_lebels.append(l[1])
    l2 = [x for x in l if x not in l1]
    test_lines.append(l2)
    
    
f = open('emoji.txt', 'r')
rawemoji = f.readlines()
f.close()

#......removing newlines and tab from emoji text
textemoji=[]
for i in range(len(rawemoji)):
    textemoji.append(re.sub('\s+',' ',rawemoji[i]))
    
#......making list of emojis with their meaning 
emoji=[]
for i in range(len(rawemoji)):
    emoji.append(textemoji[i].split())
    
#.......replaceing emojis with their meaning from twitter text
for i in range(len(train_lines)):
    for j in range(len(train_lines[i])):
        for k in range(len(emoji)):
            if train_lines[i][j]==emoji[k][0]:
                train_lines[i][j]=emoji[k][1]
                
                
for i in range(len(test_lines)):
    for j in range(len(test_lines[i])):
        for k in range(len(emoji)):
            if test_lines[i][j]==emoji[k][0]:
                test_lines[i][j]=emoji[k][1]
                
#.......creating dictionary
words=[]
for i in range(len(train_lines)):
    for j in range(len(train_lines[i])):
        words.append(train_lines[i][j])
        
words= list(set(words))

lower_case=map(str.lower,words)

dictionary=sorted(lower_case)


#........one hot encoding and CBOW algorithm
inputs=[]
outputs=[]

for j in range(len(train_lines)):
    for i in range(len(train_lines[j])):
        train_lines[j][i]=train_lines[j][i].lower()
        
for j in range(len(train_lines)):
    for i in range(len(train_lines[j])):
        for k in range(len(dictionary)):
            if dictionary[k]==train_lines[j][i]:
                train_lines[j][i]=k
                
for j in range(len(test_lines)):
    for i in range(len(test_lines[j])):
        test_lines[j][i]=test_lines[j][i].lower()
        
for j in range(len(test_lines)):
    for i in range(len(test_lines[j])):
        for k in range(len(dictionary)):
            if dictionary[k]==test_lines[j][i]:
                test_lines[j][i]=k
            else:
                test_lines[j][i]=0                

                
for j in range(len(train_lines)):
    for i in range(len(train_lines[j])-1):
        inputs.append(train_lines[j][i])
        outputs.append(train_lines[j][i+1])
        outputs.append(train_lines[j][i])
        inputs.append(train_lines[j][i+1])
               
label_encoder = LabelEncoder()
inputs_integer_encoded = label_encoder.fit_transform(inputs)
outputs_integer_encoded = label_encoder.fit_transform(outputs)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = inputs_integer_encoded.reshape(len(inputs_integer_encoded), 1)
inputs_onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
integer_encoded = outputs_integer_encoded.reshape(len(outputs_integer_encoded), 1)
outputs_onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

model=Sequential([
        Dense(500,input_dim=len(words)),
        
        Dense(100),
        
        Dense(output_dim=len(words)),
        Activation('softmax'),
        ])
    
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

model.fit(inputs_integer_encoded,outputs_integer_encoded,epochs=100,batch_size=10)

#........serialize model to JSON
model_json = model.to_json()
with open("model_sar.json", "w") as json_file:
    json_file.write(model_json)
#........serialize weights to HDF5
model.save_weights("model_sar.h5")
print("Saved model to disk")


#......LSTM model

numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
#(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
X_train=train_lines
Y_train=train_lebels

X_test=test_lines
Y_test=test_lebels
# truncate and pad input sequences
max_review_length = 20
label_encoder = LabelEncoder()
inputs_integer = label_encoder.fit_transform(X_train)
X_train= sequence.pad_sequences(X_train, maxlen=max_review_length)
Y_train=numpy.asarray(Y_train)

test_inputs_integer = label_encoder.fit_transform(X_train)
X_test= sequence.pad_sequences(X_test, maxlen=max_review_length)
Y_test=numpy.asarray(Y_test)


embedding_vector_length =100
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, Y_train, epochs=100, batch_size=64)



scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))