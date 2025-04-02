from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall


def get_metric_collection(num_classes: int) -> MetricCollection:
    collection = MetricCollection(
        {
            "accuracy-micro": Accuracy(task="multiclass", average="micro", num_classes=num_classes),
            "accuracy-macro": Accuracy(task="multiclass", average="macro", num_classes=num_classes),
            "recall": Recall(task="multiclass", num_classes=num_classes),
            "precision": Precision(task="multiclass", num_classes=num_classes),
            "f1": F1Score(task="multiclass", num_classes=num_classes),
        },
    )
    return collection
