import os
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

# Carregar o "Cérebro" e o "Dicionário"
model = tf.keras.models.load_model('model/kernel_3_2.h5')
with open('model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route('/kernel/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("text", "")

    # Converter texto em binário para a IA
    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq, maxlen=100)
    
    # Predição Neuronal
    prediction = model.predict(padded)
    # Lógica de decodificação da resposta (simplificada)
    response_text = "Kernel 3.2: Processamento concluído para " + user_input

    return jsonify({
        "status": "online",
        "engine": "Kernel-Core-3.2",
        "response": response_text
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
  
