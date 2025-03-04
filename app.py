import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image

# Inicializar o Flask
app = Flask(__name__)

# Configuração para upload de arquivos
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Certifique-se de que a pasta de uploads existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Caminho do modelo salvo (modelo TFLite)
model_path = "vgg16_glaucoma_classifier_v2.tflite"

# Carregar o modelo TFLite usando a API do TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Função para verificar se a extensão do arquivo é válida
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Função para carregar e pré-processar a imagem usando PIL
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")  # Garante 3 canais de cor
    img = img.resize(target_size)  
    img_array = np.array(img).astype("float32") / 255.0  # Normaliza
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona batch
    return img_array

# Função para fazer a previsão com o modelo TFLite
def predict_image(img_path):
    img_array = preprocess_image(img_path)

    # Obter os detalhes de entrada e saída do modelo
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Atribuindo a entrada do modelo
    input_data = np.array(img_array, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Executando a inferência
    interpreter.invoke()

    # Obtendo a saída da previsão
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Imprimir as probabilidades da previsão para depuração
    print(f"Probabilidades da previsão: {output_data}")

    # Se a saída for uma previsão binária, pode ser necessário ajustar o índice de acordo com a classe
    predicted_class = "glaucoma" if output_data[0][0] < 0.5 else "normal"
    return f"Imagem classificada como: {predicted_class.upper()}"

# Rota principal para upload e visualização
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Processar a imagem e obter a predição
            result = predict_image(filepath)

            return render_template("index.html", image_path=filepath, prediction=result)

    return render_template("index.html", image_path=None, prediction=None)

# Executar a aplicação
if __name__ == "__main__":
    app.run(debug=False)
