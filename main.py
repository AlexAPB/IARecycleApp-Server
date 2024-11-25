from flask import Flask, request, jsonify, send_file
from PIL import Image
from openai import OpenAI
import scan

client = OpenAI(api_key = "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")

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect_objects():
    print("Teste")
    if 'image' not in request.files:
        return jsonify({"error": "No image part"})
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    print("Teste")
    image = Image.open(file.stream)
    processed_image = scan.detect(image)
    if processed_image:
        return send_file(processed_image, mimetype='image/jpeg')
    else:
        return jsonify({"error": "Detection failed"})

@app.route('/ask', methods=['POST'])
def ask_chatgpt():
    print("Teste")
    try:
        data = request.get_json()
        question = data['question']
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": question}],
            stream=True,
        )
        
        answer = ''
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                answer += chunk.choices[0].delta.content

        return jsonify({"answer": answer}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)