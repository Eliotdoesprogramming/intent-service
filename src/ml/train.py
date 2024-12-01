import logging

import mlflow
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from schema.schema import TrainingConfig

logger = logging.getLogger("intent-service")


def train_intent_classifier(
    dataframe: pl.DataFrame,
    training_config: TrainingConfig = TrainingConfig(),
    progress_queue=None,
    shutdown_event=None,
):
    """
    Train an intent classifier using the provided dataframe and training configuration.

    Args:
        dataframe: A polars dataframe with columns:
            - intent: The intent label of the text
            - text: The text to classify
        training_config: Configuration parameters for training
        progress_queue: Optional queue to report training progress
        shutdown_event: Optional event to signal training should stop

    Returns:
        tuple: (trained model, list of intents, tokenizer)
    """
    intents = dataframe["intent"].unique().to_list()

    if progress_queue:
        progress_queue.put({
            "status": "initializing",
            "message": f"Initializing model with {len(intents)} intents...",
            "total_epochs": training_config.num_epochs,
        })

    # Check for early termination
    if shutdown_event and shutdown_event.is_set():
        return None, None, None

    # Initialize tokenizer and model using AutoClasses
    tokenizer = AutoTokenizer.from_pretrained(training_config.base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        training_config.base_model_name,
        num_labels=len(intents),
        problem_type="single_label_classification",
    )

    if progress_queue:
        progress_queue.put({
            "status": "training",
            "message": "Model initialized, starting training...",
            "current_epoch": 0,
        })

    # Setup training
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    if not mlflow.active_run():
        mlflow.start_run()

    # Log parameters
    mlflow.log_params({
        "model_name": training_config.base_model_name,
        "learning_rate": training_config.learning_rate,
        "weight_decay": training_config.weight_decay,
        "num_epochs": training_config.num_epochs,
        "batch_size": training_config.batch_size,
        "max_length": training_config.max_length,
        "early_stopping_patience": training_config.early_stopping_patience,
        "num_labels": len(intents),
    })

    # Training loop
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(training_config.num_epochs):
        # Check for termination at start of epoch
        if shutdown_event and shutdown_event.is_set():
            return None, None, None

        epoch_loss = 0
        batch_count = 0

        if progress_queue:
            progress_queue.put({
                "status": "training",
                "message": f"Starting epoch {epoch + 1}/{training_config.num_epochs}",
                "current_epoch": epoch + 1,
                "total_epochs": training_config.num_epochs,
            })

        for batch_start in range(0, len(dataframe), training_config.batch_size):
            # Check for termination periodically
            if shutdown_event and shutdown_event.is_set():
                return None, None, None

            # Prepare batch
            batch_df = dataframe.slice(
                offset=batch_start,
                length=min(training_config.batch_size, len(dataframe) - batch_start),
            )
            texts = batch_df["text"].to_list()
            intent_labels = batch_df["intent"].to_list()

            # Tokenize with max length from config
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=training_config.max_length,
            )

            # Forward pass
            outputs = model(
                **inputs,
                labels=torch.tensor([
                    intents.index(intent) for intent in intent_labels
                ]),
            )
            loss = outputs.loss
            epoch_loss += loss.item()
            batch_count += 1

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if progress_queue and batch_count % 10 == 0:
                progress = (batch_start + len(texts)) / len(dataframe)
                progress_queue.put({
                    "status": "training",
                    "message": f"Epoch {epoch + 1}/{training_config.num_epochs} - {progress:.1%} complete",
                    "current_epoch": epoch + 1,
                    "epoch_progress": progress,
                    "current_loss": loss.item(),
                })

        # Calculate and log metrics for the epoch
        avg_epoch_loss = epoch_loss / batch_count
        logger.info(
            f"Epoch {epoch + 1}/{training_config.num_epochs}, "
            f"Loss: {avg_epoch_loss:.4f}"
        )

        mlflow.log_metrics({"loss": avg_epoch_loss, "epoch": epoch + 1}, step=epoch)

        if progress_queue:
            progress_queue.put({
                "status": "training",
                "message": f"Completed epoch {epoch + 1}/{training_config.num_epochs}, Loss: {avg_epoch_loss:.4f}",
                "current_epoch": epoch + 1,
                "epoch_loss": avg_epoch_loss,
            })

        # Early stopping check
        if training_config.early_stopping_patience:
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
                mlflow.log_metric("best_loss", best_loss, step=epoch)
            else:
                patience_counter += 1

            if patience_counter >= training_config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                mlflow.log_param("stopped_epoch", epoch + 1)
                if progress_queue:
                    progress_queue.put({
                        "status": "training",
                        "message": f"Early stopping triggered after {epoch + 1} epochs",
                        "early_stopped": True,
                    })
                break

    model.eval()
    return model, intents, tokenizer
