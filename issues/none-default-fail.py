# works
from typing import Any


def main(model: Any):
    print(model)


# fails
from sklearn.ensemble import RandomForestClassifier


def main(model: RandomForestClassifier):
    print(model)


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main, as_positional=False)
