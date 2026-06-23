# Natural Language Understanding: language modelling and intent/slot filling

Code for the Final Project from the NLU course at the University of Trento. The first assignment builds a language model, the second one does intent classification and slot filling on ATIS. Both reports are in the repo, for a longer version with all the settings.

## Part 1: language modeling

The main goal was to push the perplexity of a neural language model as low as possible (lower is better). I started from a plain LSTM and added one change at a time.

Basic improvements first:

| Setup | Test perplexity |
| - | - |
| LSTM baseline | 160.00 |
| + dropout | 102.43 |
| + AdamW | 101.37 |


Then stronger regularisation on top of the best LSTM:

| Setup | Test perplexity |
| - | - |
| + weight tying | 117.98 |
| + variational dropout | 95.79 |
| + NT-AvSGD | 96.02 |


## Part 2: intent classification and slot filling (ATIS)

Here the model has to recognise the intent of a sentence and label each token with its slot at the same time. I report intent accuracy and token-level slot F1.

| Model | Slot F1 | Intent accuracy |
| - | - | - |
| LSTM baseline | 0.926 | 0.929 |
| + bidirectional | 0.939 | 0.953 |
| + dropout (BiLSTM) | 0.943 | 0.950 |
| BERT (bert-base-uncased, joint) | 0.890 | 0.969 |


The BiLSTM with dropout gave the best slot F1, 0.943, while fine-tuning BERT gave the best intent accuracy, 0.969. BERT lost some ground on slots, which surprised me at first. Its WordPiece tokenizer splits words into pieces, and lining those pieces back up with the original slot labels costs F1 on a small dataset like ATIS. The report goes through this in more detail.

## Stack

Python, PyTorch, Hugging Face Transformers, Jupyter. The code for both parts is in this repo, split by assignment.


## References

- Merity et al., Regularizing and Optimizing LSTM Language Models, 2017.

- Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers, 2019.

