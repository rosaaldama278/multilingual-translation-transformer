# app.py

from flask import Flask, request, jsonify, render_template, redirect, url_for
from scripts.inference import load_model, translate

app = Flask(__name__)

# Cache models to avoid loading them on each request
model_cache = {}

def get_model(src_lang, tgt_lang):
    lang_pair = f"{src_lang}-{tgt_lang}"
    if lang_pair not in model_cache:
        model, sp = load_model(src_lang, tgt_lang)
        model_cache[lang_pair] = (model, sp)
    return model_cache[lang_pair]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_route():
    source_lang = request.form.get('source_lang')
    target_lang = request.form.get('target_lang')
    input_text = request.form.get('input_text')

    if not all([source_lang, target_lang, input_text]):
        return jsonify({"error": "Missing required parameters"}), 400

    model, sp = get_model(source_lang, target_lang)
    translation = translate(input_text, model, sp)

    return render_template('index.html', translation=translation, source_lang=source_lang, target_lang=target_lang, input_text=input_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
