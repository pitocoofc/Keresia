import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# --- CONFIGURAÇÕES DO NÚCLEO ---
VOCAB_SIZE = 5000  # Tamanho do dicionário
MAX_LEN = 100      # Comprimento máximo da frase
EMBEDDING_DIM = 64 # Profundidade do "significado"

# 1. Carregar Dados Brutos
with open('data/corpus.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

# 2. Arquitetura da Rede Neural (Kernel 3.2)
model = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dense(VOCAB_SIZE, activation='softmax') # Prediz a próxima "ideia"
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 3. Treinar e Salvar
print("[KERNEL] Iniciando treinamento bruto...")
# model.fit(padded_sequences, labels, epochs=50) # Aqui entrariam os labels do seu dataset
model.save('model/kernel_3_2.h5')

# Salva o tokenizer para usar no servidor
with open('model/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("[SUCCESS] Cérebro Kernel 3.2 gerado.")
