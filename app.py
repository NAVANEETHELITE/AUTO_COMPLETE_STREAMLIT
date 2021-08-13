#importing necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image
import os
import base64


file_ = open("IMAGES/rec.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.title("NEXT WORD PREDICTOR >>")

#loading the pre-trained weights and model architecture
model = tf.keras.models.load_model('MODELS/AUTO_COM_model.h5')

#dataset preprocessing
file = open("DATA/Goodwill.txt").read() #opeining the dataset and reading from it

tokenizer = Tokenizer() #tokenizing the dataset
data = file.lower().split("\n") #converting dataset to lowercase

#removing whitespaces from the dataset
corpus = []
for line in data:
    a = line.strip()
    corpus.append(a)

#generating tokens for each sentence in the data
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
# print(tokenizer.word_index)
# print(total_words)

#creating labels for each sentence in dataset
input_sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# create predictors and label
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)



#generating next words given a seed
def next_word(seed):
  seed_text = seed
  next_words = num
  for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
      if index == predicted:
        output_word = word
        break
    seed_text += " " + output_word
  st.subheader('SENTENCE COMPLETION:')
  st.subheader(seed_text)

menu = ["AUTO COMPLETE", "ME"]
choice = st.sidebar.selectbox("Menu",menu)

if choice == "AUTO COMPLETE":
  st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="RNN gif">',
    unsafe_allow_html=True,
)
  num = st.slider("Number of text predictions? (minimum should be 1)",0,5)
  next_word(st.text_input('Enter your sentence','How are'))

else:
  st.header("NAVANEETH")
  me = Image.open(os.path.join('IMAGES/m2.jpeg'))
  st.image(me)
  st.write("REACH ME AT :")
  st.write("[LINKEDIN](https://www.linkedin.com/in/navaneethan-s-a527571b7/)")
  st.write("[EMAIL](mailto:navaneethanselvakumar@gmail.com)")
  st.write("[INSTAGRAM](https://www.instagram.com/_navneeth_/)")

