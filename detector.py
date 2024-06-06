import argparse
import pickle
from collections import Counter
from pathlib import Path
import face_recognition
from PIL import Image, ImageDraw
import psycopg2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from bancodedados import conect_db #Função para conectar ao banco
#Configurações do Servidor
app = Flask(__name__)
CORS(app)
app.debug = True


#Configurações de caminho padrão e Exbição do resultado
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"


#Função para conectar com o banco

#Função para treinamento do Modelo
def encode_known_faces(
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    """
    Loads images in the training directory and builds a dictionary of their
    names and encodings.
    """
    names = []
    encodings = []

    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

#Função para Reconhecimento de imagem desconhecida
def recognize_faces(
    image_location: str,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
    """
    Given an unknown image, get the locations and encodings of any faces and
    compares them against the known encodings to find potential matches.
    """
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        _display_face(draw, bounding_box, name)

    del draw
    name = _recognize_face(unknown_encoding, loaded_encodings)
    pillow_image.save(f'results/{name}.jpg')

#Função Auxiliar para encontrar a face que mais se adequa
def _recognize_face(unknown_encoding, loaded_encodings):
    """
    Given an unknown encoding and all known encodings, find the known
    encoding with the most matches.
    """
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]

#Função Auxiliar para Desenhar Resultado na imagem
def _display_face(draw, bounding_box, name):
    """
    Draws bounding boxes around faces, a caption area, and text captions.
    """
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill=BOUNDING_BOX_COLOR,
        outline=BOUNDING_BOX_COLOR,
    )
    draw.text(
        (text_left, text_top),
        name,
        fill=TEXT_COLOR,
    )

#Rotas de app
@app.route("/treinamento", methods=['POST'])
def treinamento():
    #Receber Inputs do Formulario 
    primeiro_nome = request.form.get('primeiro_nome')
    segundo_nome = request.form.get('segundo_nome') 
    cpf = request.form.get('cpf')
    documento = request.files['file-input']
    #Buscar cpf no banco
    conn = conect_db()
    cur = conn.cursor()
    sql_cpf = 'SELECT cpf FROM apresentacao.login WHERE cpf = %s'
    cur.execute(sql_cpf,(cpf,))
    result_cpf = cur.fetchone()

    #Verificar se cpf está cadastrado Caso cpf não esteja cadastrado realizamos o cadastro
    if(result_cpf == None):
        #Salvar Documento com foto em pasta com nome da pessoa
        base='training'
        destino = os.path.join(base,primeiro_nome + ' ' + segundo_nome)
        if not os.path.exists(destino):
            os.makedirs(destino)
        caminho_arquivo = os.path.join(destino, documento.filename)
        documento.save(caminho_arquivo)
        
        #Realizar treinamento do modelo; OBS: Para efeitos de demonstração estamos realizando o treinamento em todas as fotos salvas no servidor, em caso dew aplicação será necessário otimizar essa rotina de treinamento
        encode_known_faces()

        #Fechar Conexão
        cur.close()
        conn.close()

        #Retornar sucesso
        return jsonify({'success':True})
    
    else: #Em caso de aplicação aqui barrariamos a continuação do processo
        #Salvar Documento com foto em pasta com nome da pessoa
        base='training'
        destino = os.path.join(base,primeiro_nome + ' ' + segundo_nome)
        if not os.path.exists(destino):
            os.makedirs(destino)
        caminho_arquivo = os.path.join(destino, documento.filename)
        documento.save(caminho_arquivo)
        
        #Realizar treinamento do modelo; OBS: Para efeitos de demonstração estamos realizando o treinamento em todas as fotos salvas no servidor, em caso dew aplicação será necessário otimizar essa rotina de treinamento
        encode_known_faces()

        #Fechar Conexão
        cur.close()
        conn.close()

        #Retornar sucesso
        return jsonify({'success':True})
    
@app.route("/cadastro", methods=['POST'])
def cadastro():
    pais_residencia = request.form.get('pais_residencia')
    dia = request.form.get('dia')
    mes = request.form.get('mes')
    ano = request.form.get('ano')
    senha = request.form.get('senha')
    email = request.form.get('Email')
    primeiro_nome = request.form.get('primeiro_nome')
    segundo_nome = request.form.get('segundo_nome')
    cpf = request.form.get('cpf')
    conn = conect_db()
    cur = conn.cursor()
    #Inserir dados no banco de Dados
    sql_inserir = 'INSERT INTO apresentacao.login(pais_residencia, primeiro_nome, segundo_nome, dia, mes, ano, cpf, senha, email) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);'
    cur.execute(sql_inserir,(pais_residencia, primeiro_nome, segundo_nome, dia, mes, ano, cpf, senha, email))
    conn.commit()
    #Fechar Conexão
    cur.close()
    conn.close()
    #Retornar Sucesso
    return jsonify({'success':True})

@app.route("/validar", methods=['POST'])
def validar():
    documento = request.files['file-input']
    recognize_faces

        

    

    


    


if __name__ == '__main__':
    ip ='localhost'
    port = 5000
    app.run(host=ip,port=port,debug=True)



