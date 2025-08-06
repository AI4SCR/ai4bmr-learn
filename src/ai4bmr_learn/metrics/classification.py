from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall


def get_metric_collection(num_classes: int) -> MetricCollection:
    if num_classes > 2:
        collection = MetricCollection(
            {
                "accuracy-micro": Accuracy(task="multiclass", average="micro", num_classes=num_classes),
                "accuracy-macro": Accuracy(task="multiclass", average="macro", num_classes=num_classes),
                "recall-micro": Recall(task="multiclass", average="micro", num_classes=num_classes),
                "recall-macro": Recall(task="multiclass", average="macro", num_classes=num_classes),
                "precision-micro": Precision(task="multiclass", average="micro", num_classes=num_classes),
                "precision-macro": Precision(task="multiclass", average="macro", num_classes=num_classes),
                "f1-weighted": F1Score(task="multiclass", average="weighted", num_classes=num_classes),
                "f1-micro": F1Score(task="multiclass", average="micro", num_classes=num_classes),
                "f1-macro": F1Score(task="multiclass", average="macro", num_classes=num_classes),
            },
        )
    else:
        collection = MetricCollection({
            "accuracy": Accuracy(task='binary'),
            "recall": Recall(task='binary'),
            "precision": Precision(task='binary'),
            "f1": F1Score(task='binary'),
        })
    return collection


