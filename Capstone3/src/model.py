# src/model.py

from transformers import DistilBertForSequenceClassification


def build_model(
    name: str = "distilbert-base-uncased",
    num_classes: int = 2,
    pretrained: bool = True,
):
    """
    Build and return a text classification model.

    Args:
        name: Transformer model name (HuggingFace)
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights
    """

    if pretrained:
        model = DistilBertForSequenceClassification.from_pretrained(
            name,
            num_labels=num_classes,
        )
    else:
        model = DistilBertForSequenceClassification.from_config(
            name,
            num_labels=num_classes,
        )

    return model
