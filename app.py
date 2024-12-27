from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

model_dir = './trained_model'
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model.eval()

def preprocess_input(input_text):
    encoding = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    return encoding['input_ids'], encoding['attention_mask']

def predict_sentiment(input_text):
    input_ids, attention_mask = preprocess_input(input_text)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    probabilities = torch.sigmoid(logits).item()
    return probabilities

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received data:", data)  # Debugging output

        if 'messages' not in data or not data['messages']:
            return jsonify({'error': 'Empty or invalid message list'}), 400

        messages = data['messages']
        total_sentiment = 0

        for message in messages:
            if 'content' not in message:
                return jsonify({'error': 'Message missing content'}), 400
            content = message['content']
            sentiment = predict_sentiment(content)
            total_sentiment += sentiment

        average_sentiment = total_sentiment / len(messages)
        return jsonify({'average_sentiment': average_sentiment})

    except Exception as e:
        print("An error occurred:", str(e))
        return jsonify({'error': 'An error occurred'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)