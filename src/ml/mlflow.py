import os

import mlflow
import torch


class IntentModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from transformers import DistilBertTokenizer

        """Load the model artifacts"""
        self.model = torch.load(
            os.path.join(context.artifacts["model_path"], "model.pth"),
            weights_only=False,
        )
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            context.artifacts["tokenizer_path"]
        )
        self.intent_labels = torch.load(
            os.path.join(context.artifacts["model_path"], "intent_labels.pth"),
            weights_only=False,
        )
        self.model.eval()

    def predict(self, context, model_input):
        # Tokenize the input texts
        inputs = self.tokenizer(
            model_input["text"].tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)

        # Convert to numpy for easier handling
        probs = probabilities.numpy()

        # Create a list of dictionaries containing intent labels and their scores
        results = []
        for prob in probs:
            intent_scores = {
                intent: float(score) for intent, score in zip(self.intent_labels, prob)
            }
            results.append(intent_scores)

        return results
