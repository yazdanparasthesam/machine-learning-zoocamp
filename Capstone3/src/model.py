# src/model.py

from transformers import DistilBertForSequenceClassification

def build_model(name: str, num_classes: int, pretrained: bool = True):
    """
    Build DistilBERT model for sequence classification
    """

    if pretrained:
        model = DistilBertForSequenceClassification.from_pretrained(
            name,
            num_labels=num_classes,
        )
    else:
        model = DistilBertForSequenceClassification.from_pretrained(
            name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    return model
