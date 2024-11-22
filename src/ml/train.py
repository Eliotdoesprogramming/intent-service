from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import polars as pl
import mlflow
import pandas as pd
import logging
import os
from ml.mlflow import IntentModel
logger = logging.getLogger('intent-service')

def train_intent_classifier(dataframe: pl.DataFrame):
    """
    dataframe:
        A pandas dataframe with the following columns:
        - intent: The intent label of the text
        - text: The text to classify
    """
    intents = dataframe['intent'].unique().to_list()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(intents))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    for i in range(10):
        print(f"Epoch {i}")
        for batch in range(0, len(dataframe), 32):
            batch_df = dataframe.slice(offset=batch, length=min(32, len(dataframe) - batch))
            texts = batch_df['text'].to_list()
            intent_labels = batch_df['intent'].to_list()
            inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
            outputs = model(**inputs, labels=torch.tensor([intents.index(intent) for intent in intent_labels]))
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    model.eval()
    
    # After training is complete:
    with mlflow.start_run() as run:
        # Save model artifacts
        artifact_path = "model"
        os.makedirs(artifact_path, exist_ok=True)
        
        # Save the model
        torch.save(model, os.path.join(artifact_path, "model.pth"))
        
        # Save intent labels
        torch.save(intents, os.path.join(artifact_path, "intent_labels.pth"))
        
        # Save tokenizer
        tokenizer.save_pretrained(os.path.join(artifact_path, "tokenizer"))
        
        # Create an example input
        example_input = pd.DataFrame({
            "text": ["example text for signature"]
        })
        
        # Log the model with its artifacts
        mlflow.pyfunc.log_model(
            artifact_path="intent_model",
            python_model=IntentModel(),
            artifacts={
                "model_path": artifact_path,
                "tokenizer_path": os.path.join(artifact_path, "tokenizer")
            },
            input_example=example_input
        )
        
        return run.info.run_id