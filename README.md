# Modified SmolVLM with TokenLearner and QFormer

This project replaces the connector module in the SmolVLM model with a custom connector using TokenLearner and QFormer to improve performance on visual question answering tasks.

## Project Structure

- `main.py`: Script to load the original SmolVLM model, replace its connector, and save the modified model
- `custom_connector.py`: Implementation of the custom TokenLearner and QFormer connector modules
- `finetune.py`: Script to fine-tune the modified model on the TextVQA dataset
- `requirements.txt`: List of required packages

## How It Works

1. **TokenLearner**: Learns to select the most important visual tokens from the visual encoder's output
2. **QFormer**: Uses cross-attention to transform these tokens into a format suitable for the language model

This combination provides a more advanced way to connect the vision and language models compared to the original connector which used a simple MLP projection.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Replace the connector and save the modified model

```bash
python main.py
```

This will:
- Load the SmolVLM model
- Replace the connector with our custom TokenLearner+QFormer connector
- Save the modified model

### 2. Fine-tune the modified model on TextVQA

```bash
python finetune.py
```

This will:
- Load the modified model
- Fine-tune it on the TextVQA dataset
- Save the fine-tuned model
- Run an example inference

## Customization

You can modify various parameters in both scripts:
- Number of tokens in TokenLearner (`token_learner_tokens`)
- Number of query tokens in QFormer (`qformer_query_tokens`)
- Training parameters in `finetune.py`

## Requirements

See `requirements.txt` for the list of required packages. 