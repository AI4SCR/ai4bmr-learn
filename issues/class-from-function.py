from jsonargparse import ArgumentParser, class_from_function
from ai4bmr_learn.datasets.Tabular import TabularDataset

parser = ArgumentParser()
parser.add_argument("--config", action="config")
dynamic_class = class_from_function(TabularDataset.from_paths, TabularDataset)
parser.add_class_arguments(dynamic_class, 'data')

if __name__ == "__main__":
    args = parser.parse_args()