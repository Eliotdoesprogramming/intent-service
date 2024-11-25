from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import polars as pl
import mlflow
import pandas as pd
import logging
import os
from ml.mlflow import IntentModel
from schema.schema import TrainingConfig
logger = logging.getLogger('intent-service')

def train_intent_classifier(dataframe: pl.DataFrame, training_config: TrainingConfig):
    """
    Train an intent classifier using the provided dataframe and training configuration.
    
    Args:
        dataframe: A polars dataframe with columns:
            - intent: The intent label of the text
            - text: The text to classify
        training_config: Configuration parameters for training
            
    Returns:
        tuple: (trained model, list of intents, tokenizer)
    """
    intents = dataframe['intent'].unique().to_list()
    
    # Initialize tokenizer and model using config
    tokenizer = DistilBertTokenizer.from_pretrained(training_config.model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        training_config.model_name, 
        num_labels=len(intents)
    )
    
    # Setup training
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(training_config.num_epochs):
        epoch_loss = 0
        batch_count = 0
        
        for batch_start in range(0, len(dataframe), training_config.batch_size):
            # Prepare batch
            batch_df = dataframe.slice(
                offset=batch_start, 
                length=min(training_config.batch_size, len(dataframe) - batch_start)
            )
            texts = batch_df['text'].to_list()
            intent_labels = batch_df['intent'].to_list()
            
            # Tokenize with max length from config
            inputs = tokenizer(
                texts, 
                return_tensors='pt', 
                padding=True, 
                truncation=True,
                max_length=training_config.max_length
            )
            
            # Forward pass
            outputs = model(
                **inputs, 
                labels=torch.tensor([intents.index(intent) for intent in intent_labels])
            )
            loss = outputs.loss
            epoch_loss += loss.item()
            batch_count += 1
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / batch_count
        logger.info(f"Epoch {epoch + 1}/{training_config.num_epochs}, Loss: {avg_epoch_loss:.4f}")
        
        # Early stopping check
        if training_config.early_stopping_patience:
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= training_config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    model.eval()
    return model, intents, tokenizer

def package_model(model, intents, tokenizer):
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