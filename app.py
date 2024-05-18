from flask import Flask, render_template, request, url_for,jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model/toxicity_model.pkt', 'rb'))
vectorizer = pickle.load(open('model/tf_idf.pkt', 'rb'))
  

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


@app.route('/comment_detector', methods=['POST'])
def comment_detector():
    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        transformed_data = vectorizer.transform(data)
        prediction = model.predict(transformed_data)
        result = 'Toxic' if prediction[0] == 1 else 'Non-Toxic'
        return jsonify({'prediction': result})


if __name__ == '__main__':
    app.run(port=5000, debug=True)