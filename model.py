import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Load data
reddit_data = pd.read_csv('data/Reddit_Data.csv')
twitter_data = pd.read_csv('data/Twitter_Data.csv')

data = pd.concat([reddit_data, twitter_data])

# Clean data: Remove any NaN values in 'clean_comment' and 'category'
data = data.dropna(subset=['clean_comment', 'category'])

# Use a subset of the data for faster training
data = data.sample(frac=0.1, random_state=42)  # Use 10% of the data

# Extract text and sentiment
texts = data['clean_comment'].astype(str).values
sentiments = data['category'].values

# Scale sentiment scores to 0-1 for regression
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_sentiments = scaler.fit_transform(sentiments.reshape(-1, 1))

# Split into train and test sets
texts_train, texts_test, sentiments_train, sentiments_test = train_test_split(
    texts, scaled_sentiments, test_size=0.2, random_state=42
)

print(texts_train.shape, texts_test.shape, sentiments_train.shape, sentiments_test.shape)

# Tokenize the text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class SentimentDataset(Dataset):
    def __init__(self, texts, sentiments, tokenizer, max_length=128):
        self.texts = texts
        self.sentiments = sentiments
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        sentiment = self.sentiments[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(sentiment, dtype=torch.float)
        }

# Prepare datasets
train_dataset = SentimentDataset(texts_train, sentiments_train, tokenizer)
test_dataset = SentimentDataset(texts_test, sentiments_test, tokenizer)

# Directory to save/load the model
model_dir = './trained_model'

# Define training arguments with reduced epochs and increased batch size
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,  # Reduced epochs
    per_device_train_batch_size=32,  # Increased batch size
    per_device_eval_batch_size=128,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    fp16=True  # Enable mixed precision training for faster performance
)

# Check if the model is already trained
if os.path.exists(model_dir):
    # Load the saved model
    model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=1)
else:
    # Train the model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    # Train the model
    trainer.train()

    # Save the trained model
    trainer.save_model(model_dir)

# Evaluate the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

results = trainer.evaluate()
print(results)