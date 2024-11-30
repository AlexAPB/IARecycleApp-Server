from flask import Flask, request, jsonify, send_file
from PIL import Image
from openai import OpenAI
import scan

client = OpenAI(api_key = "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")
historic = {}

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"})
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
        
    image = Image.open(file.stream)
    output = scan.detect(image)
    processed_image = output['image']
    detection_info = output.get('detection', None)
    if processed_image:
        return jsonify({"detection": detection_info, "image": processed_image})
    else:
        return jsonify({"error": "Detection failed"})

@app.route('/ask', methods=['POST'])
def ask_chatgpt():
    try:
        data = request.get_json()
        identifier = data['identifier']
        question = data['question']
        materials = data['materials']

        if data['materials']:
            question = f"Falando em {materials}... {question}"

        if identifier not in historic:
            historic[identifier] = { 'questions': [], 'answers': [] }
        
        profession = "Você é um especialista em reciclagem do aplicativo RecycleApp. Responda apenas perguntas relacionadas à reciclagem e resíduos ou às mensagens anteriores, mas sem desviar muito o assunto."
        limit = "Se a pergunta não for relacionada à reciclagem, diga 'Desculpe, só respondo perguntas sobre reciclagem.'."
        weight = "Se a pergunta, de alguma forma possível, puder estar relacionada à reciclagem de resíduos, responda-a corretamente."
        relation = "Caso não saiba a que material reciclável a pergunta se refere, ou o usuário diga que quer falar sobre outro tipo de material, ou a mensagem insinue que o usuário quer falar sobre outro material que ele ainda não comentou qual é, diga exatamente '<ask_material>'"
        exception = "Caso o usuário tenha uma dúvida sobre um material não-reciclável, mas com a pergunta sendo sobre reciclagem, responda-a, mas também aponte que não é um material reciclável."
        length = "Dê respostas de no máximo 50 palavras, a não ser que seja um passo a passo."
        style = "Não use símbolos como os que geram negrito ou itálico."
        rules = "Se alguém disser o nome de um material ou objeto sem contexto nenhum na mensagem, presuma que a pessoa está perguntando como reciclá-lo."
        disposable = "Caso o material seja reciclável, mas é possível que ele se torne não reciclável em alguma situação, avise ao usuário, se necessário, desde que já não tenha sido avisado antes."
        
        context = f"{profession} {limit} {weight} {relation} {exception} {length} {style} {rules} {disposable}"
            
        messages = []
        messages.append({"role": "system", "content": context})
        for q, a in zip(historic[identifier]['questions'], historic[identifier]['answers']):
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
            
        messages.append({"role": "user", "content": question})
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
        )

        answer = ''
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                answer += chunk.choices[0].delta.content
        
        historic[identifier]['questions'].append(question)
        historic[identifier]['answers'].append(answer)

        return jsonify({"answer": answer}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)