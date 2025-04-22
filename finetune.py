import os
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, AutoModelForImageTextToText, AutoProcessor
from datasets import load_dataset
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the modified model and processor
model_path = "./modified_smolvlm_model"
model = AutoModelForImageTextToText.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)

# Move model to device
model = model.to(device)

# Load TextVQA dataset
ds = load_dataset("lmms-lab/textvqa")

# Function to load images from URL or local path
def load_image(image_path):
    if image_path.startswith(('http://', 'https://')):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')
    return image

# Data preprocessing function
def preprocess_function(examples):
    images = [load_image(img_path) for img_path in examples["image_path"]]
    questions = examples["question"]
    answers = examples["answers"]  # This is usually a list of possible answers
    
    # For simplicity, we'll take the first answer as the target
    first_answers = [ans_list[0] if ans_list else "" for ans_list in answers]
    
    # Prepare the prompt template
    prompts = [f"<image>\nQuestion: {q}\nAnswer:" for q in questions]
    
    # Process inputs
    inputs = processor(
        text=prompts,
        images=images, 
        return_tensors="pt", 
        padding=True,
        truncation=True
    )
    
    # Process targets
    targets = processor(
        text=first_answers,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).input_ids
    
    # Set the labels
    inputs["labels"] = targets
    
    return inputs

# Apply preprocessing to a small subset for testing
train_ds_small = ds["train"].select(range(100))  # Use 100 examples for testing
valid_ds_small = ds["validation"].select(range(20))  # Use 20 examples for validation

# Preprocess the datasets
train_dataset = train_ds_small.map(preprocess_function, batched=True, batch_size=4)
valid_dataset = valid_ds_small.map(preprocess_function, batched=True, batch_size=4)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),  # Use mixed precision training if available
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=lambda data: {
        'input_ids': torch.stack([x['input_ids'] for x in data]),
        'attention_mask': torch.stack([x['attention_mask'] for x in data]),
        'pixel_values': torch.stack([x['pixel_values'] for x in data]),
        'labels': torch.stack([x['labels'] for x in data]),
    }
)

print("Starting fine-tuning...")

# Start fine-tuning
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./finetuned_smolvlm_tokenlearner_qformer")
processor.save_pretrained("./finetuned_smolvlm_tokenlearner_qformer")

print("Fine-tuning complete! Model saved to './finetuned_smolvlm_tokenlearner_qformer'")

# Example inference with the fine-tuned model
def inference(image_path, question):
    image = load_image(image_path)
    prompt = f"<image>\nQuestion: {question}\nAnswer:"
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    
    # Generate the answer
    outputs = model.generate(
        **inputs,
        max_length=50,
        num_beams=5,
        early_stopping=True
    )
    
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    return answer

# Test with an example from the validation set
if len(valid_ds_small) > 0:
    example = valid_ds_small[0]
    image_path = example["image_path"]
    question = example["question"]
    
    print("\nTesting inference with an example from the validation set:")
    print(f"Question: {question}")
    
    answer = inference(image_path, question)
    print(f"Model Answer: {answer}")
    print(f"Ground Truth: {example['answers'][0] if example['answers'] else 'No answer'}") 