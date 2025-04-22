from transformers import AutoProcessor, AutoConfig, AutoModelForImageTextToText
import torch
from custom_connector import MyTokenLearnerQFormerConnector

# 1) Load the config & processor for the 2.2B model
model_path = "HuggingFaceTB/SmolVLM-256M-Instruct"
config    = AutoConfig.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(model_path)

# Print original model architecture
print("Original model architecture:")
print(model)

# Replace the connector with our custom connector
vision_hidden_size = model.model.vision_model.config.hidden_size  # 768
text_hidden_size = model.model.text_model.config.hidden_size     # 576

# Create our custom connector
custom_connector = MyTokenLearnerQFormerConnector(
    vision_hidden_size=vision_hidden_size,
    text_hidden_size=text_hidden_size,
    token_learner_tokens=64,  # Number of tokens to extract with TokenLearner
    qformer_query_tokens=32   # Number of query tokens in QFormer
)

# Replace the connector in the model
model.model.connector = custom_connector

# Print the modified model architecture
print("\nModified model architecture:")
print(model)

# Save the modified model
model.save_pretrained("./modified_smolvlm_model")
processor.save_pretrained("./modified_smolvlm_model")
print("Modified model saved to './modified_smolvlm_model'")

# Now prepare for fine-tuning on TextVQA dataset
print("\nSetting up TextVQA dataset for fine-tuning:")
from datasets import load_dataset

# Load the TextVQA dataset
ds = load_dataset("lmms-lab/textvqa")
print(f"TextVQA dataset loaded. Available splits: {list(ds.keys())}")

# Print sample from the dataset
if "train" in ds:
    sample = ds["train"][0]
    print("\nSample from the TextVQA dataset:")
    for key, value in sample.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"{key}: {value[:100]}...")
        else:
            print(f"{key}: {value}")
            
print("\nReady for fine-tuning with the modified model and TextVQA dataset.")
