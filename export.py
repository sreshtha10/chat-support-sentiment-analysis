import torch
from transformers import BertForSequenceClassification

# Load your trained model
model = BertForSequenceClassification.from_pretrained('./trained_model')

# Define a wrapper to handle model outputs
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Assuming you want to use the logits for your task
        logits = outputs.logits
        return logits

# Create a wrapped model instance
wrapped_model = ModelWrapper(model)

# Dummy input for tracing
dummy_input_ids = torch.zeros(1, 128, dtype=torch.long)  # Adjust dimensions as needed
dummy_attention_mask = torch.ones(1, 128, dtype=torch.long)  # Adjust dimensions as needed

# Convert to TorchScript
wrapped_model.eval()
traced_model = torch.jit.trace(wrapped_model, (dummy_input_ids, dummy_attention_mask), strict=False)
traced_model.save('model.pt')