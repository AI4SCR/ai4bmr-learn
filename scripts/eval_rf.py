from sklearn.ensemble import RandomForestClassifier
import wandb
from dataclasses import dataclass
from pathlib import Path
from ai4bmr_learn.datamodules.Tabular import TabularDataModule
from dataclasses import dataclass, asdict, field
from sklearn.model_selection import ParameterGrid
from typing import Any
import wandb
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

@dataclass
class ProjectConfig:
    project_name: str = "eval_rf"
    entity: str = None
    run_name: str = None
    tags: list[str] = None
    notes: str = ""
    mode: str = "online"
    wandb_dir: Path = Path("~/.wandb_logs").expanduser().resolve()

@dataclass
class Parameters:
    n_estimators: list[int] = field(default_factory=lambda: [50, 100])
    max_depth: list[int] =  field(default_factory=lambda: [None])
    min_samples_split: list[int] = field(default_factory=lambda: [2])
    max_features: list[str] =  field(default_factory=lambda: ["sqrt"])
    criterion: list[str] = field(default_factory=lambda: ["gini"])

@dataclass
class SweepConfig:
    method: str = "grid"
    num_outer_cv: int = 3
    num_inner_cv: int = 3
    parameters: Parameters = field(default_factory=lambda: Parameters())


def main(project: ProjectConfig, model: Any, sweep: SweepConfig, datamodule: TabularDataModule):

    from wandb.sklearn import plot_precision_recall, plot_feature_importances
    from wandb.sklearn import plot_class_proportions, plot_learning_curve, plot_roc

    wandb.init(**asdict(project))

    # DATA
    datamodule.prepare_data()
    datamodule.setup()

    x_train, y_train = datamodule.train_set.data, datamodule.train_set.targets
    x_test, y_test = datamodule.test_set.data, datamodule.test_set.targets
    x, y = np.concatenate([x_train, x_test]), np.concatenate([y_train, y_test])

    # SWEEP
    sweep = SweepConfig()
    param_grid = list(ParameterGrid(asdict(sweep.parameters)))
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            scoring='accuracy',
            cv=inner_cv,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        plot_class_proportions(y_train, y_test, labels)
        # plot_learning_curve(model, X_train, y_train)
        plot_roc(y_test, y_probas, labels)
        plot_precision_recall(y_test, y_probas, labels)
        plot_feature_importances(model)

    # Log overall stats
    wandb.log({
        "mean_accuracy": np.mean(all_scores),
        "std_accuracy": np.std(all_scores),
    })

    wandb.finish()

if __name__ == "__main__":
    from jsonargparse import auto_cli
    auto_cli(main, as_positional=False)


