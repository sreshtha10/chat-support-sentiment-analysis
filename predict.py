import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the trained model and tokenizer
model_dir = './trained_model'
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to predict sentiment of an input text
def predict_sentiment(input_text):
    # Preprocess the input text
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

    # Forward pass through the model
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to compute gradients
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # For regression tasks, you might want to directly use the logits
    # For classification, you typically convert logits to probabilities
    predicted_class = torch.sigmoid(logits).item()

    return predicted_class

# Example usage
input_text = "This is an example sentence to test sentiment analysis."
predicted_sentiment = predict_sentiment(input_text)
print(f"Predicted sentiment: {predicted_sentiment}")