# Enhanced SmolVLM with TokenLearner-QFormer Connector

This project modifies the SmolVLM-256M-Instruct model by replacing its default connector with a custom connector that combines TokenLearner and LLaMA-style QFormer architectures for better visual-language performance. The model is then fine-tuned on the TextVQA dataset.

## Architecture Modification

Original connector in SmolVLM-256M-Instruct:
```
(connector): Idefics3Connector(
  (modality_projection): Idefics3SimpleMLP(
    (proj): Linear(in_features=12288, out_features=576, bias=False)
  )
)
```

Our custom replacement:
```
(connector): MyTokenLearnerQFormerConnector(
  (token_learner): TokenLearner(...)
  (q_former): LlamaStyleQFormer(...)
)
```

### Components

1. **TokenLearner**: Dynamically extracts tokens from visual features by learning important spatial regions.
   - Based on the paper "[TokenLearner: Adaptive Space-Time Tokenization for Videos](https://arxiv.org/abs/2106.11297)"

2. **LlamaStyleQFormer**: Query-based transformer that processes learned tokens with cross-attention using LLaMA-style architecture.
   - Uses custom LLaMA-style attention and MLP layers
   - Maintains compatibility with the LLaMA architecture in the text model
   - Uses RMSNorm for normalization, matching LLaMA's approach

## Design Choice

We specifically designed the QFormer component to use LLaMA-style attention instead of BERT-style attention to:
1. Maintain architectural consistency with the text model (which is a LLaMA model)
2. Avoid compatibility issues with cross-attention implementation
3. Potentially improve performance by using the same attention mechanism throughout the model

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Code

1. Run the main script:
```bash
python main.py
```

This will:
- Load the SmolVLM-256M-Instruct model
- Replace the connector with our custom TokenLearner-LLaMA QFormer connector
- Fine-tune the model on a subset of the TextVQA dataset
- Save the fine-tuned model to the "modified_model" directory
- Run an example inference on a validation example

## Fine-tuning Details

The fine-tuning process:
- Freezes the vision model to preserve its representations
- Only trains the connector and text model components
- Uses AdamW optimizer with a learning rate of 1e-5
- Uses a linear learning rate scheduler with warmup
- Trains for 3 epochs on a subset of the TextVQA dataset

## Expected Benefits

The combination of TokenLearner and LLaMA-style QFormer should provide several advantages:

1. **Dynamic token selection**: TokenLearner adaptively focuses on the most relevant parts of the image
2. **Better vision-language alignment**: QFormer's cross-attention mechanism helps bridge visual tokens to language representations
3. **Improved text grounding**: Better handling of text within images, which is crucial for TextVQA
4. **Architectural consistency**: Using LLaMA-style attention throughout provides better gradient flow and compatibility

## Dataset

The model is fine-tuned on the [TextVQA dataset](https://textvqa.org/) from Hugging Face, which contains questions about text in images. 