import torch
import torch.nn as nn

from ai4bmr_learn.lit.mil import ClassificationMILLit, RegressionMILLit, SurvivalMILLit
from ai4bmr_learn.models.mil import AttentionAggregation


def make_batch(num_classes: int = 3):
    bag = torch.randn(4, 5, 8)
    mask = torch.ones(4, 5, dtype=torch.bool)
    mask[0, -1] = False
    return {
        "bag": bag,
        "mask": mask,
        "target": torch.tensor([0, 1, 2, num_classes - 1]),
        "time": torch.tensor([4.0, 3.0, 2.0, 1.0]),
        "event": torch.tensor([True, True, False, True]),
    }


def make_aggregator():
    return AttentionAggregation(input_dim=8, hidden_dim=4, gated=True)


def assert_common_step_output(output: dict):
    assert torch.isfinite(output["loss"])
    assert output["embedding"].shape[0] == 4
    assert output["weights"].shape == (4, 5)
    assert output["logits"].shape == (4, 5)
    assert output["prediction"].shape[0] == 4


def test_classification_mil_lit_steps_predict_and_optimizer():
    module = ClassificationMILLit(
        aggregator=make_aggregator(),
        head=nn.Linear(8, 3),
        num_classes=3,
    )
    batch = make_batch(num_classes=3)

    assert_common_step_output(module.training_step(batch, 0))
    assert_common_step_output(module.validation_step(batch, 0))
    assert_common_step_output(module.test_step(batch, 0))

    prediction = module.predict_step(batch, 0)
    assert prediction["prediction"].dtype == torch.long
    assert prediction["weights"].shape == (4, 5)

    optimizer = module.configure_optimizers()
    assert len(optimizer.param_groups) == 4
    assert {group["lr"] for group in optimizer.param_groups} == {module.lr_aggregator, module.lr_head}
    assert any(group["weight_decay"] == module.weight_decay for group in optimizer.param_groups)
    assert any(group["weight_decay"] == 0.0 for group in optimizer.param_groups)
    bias_ids = {id(param) for name, param in module.named_parameters() if name.endswith("bias")}
    decayed_ids = {
        id(param) for group in optimizer.param_groups if group["weight_decay"] > 0 for param in group["params"]
    }
    assert bias_ids.isdisjoint(decayed_ids)


def test_classification_mil_lit_supports_nested_keys():
    module = ClassificationMILLit(
        aggregator=make_aggregator(),
        head=nn.Linear(8, 3),
        num_classes=3,
        bag_key="mil.bag",
        mask_key="mil.mask",
        target_key="label.class",
    )
    batch = make_batch(num_classes=3)
    nested_batch = {
        "mil": {"bag": batch["bag"], "mask": batch["mask"]},
        "label": {"class": batch["target"]},
    }

    output = module.training_step(nested_batch, 0)

    assert torch.isfinite(output["loss"])
    assert output["prediction"].shape == (4,)


def test_regression_mil_lit_steps_and_predict():
    module = RegressionMILLit(
        aggregator=make_aggregator(),
        head=nn.Linear(8, 1),
        num_outputs=1,
    )
    batch = make_batch()
    batch["target"] = torch.randn(4)

    assert_common_step_output(module.training_step(batch, 0))
    assert_common_step_output(module.validation_step(batch, 0))
    assert_common_step_output(module.test_step(batch, 0))

    prediction = module.predict_step(batch, 0)
    assert prediction["prediction"].shape == (4, 1)
    assert prediction["embedding"].shape == (4, 8)


def test_survival_mil_lit_steps_and_predict():
    module = SurvivalMILLit(
        aggregator=make_aggregator(),
        head=nn.Linear(8, 1),
    )
    batch = make_batch()

    assert_common_step_output(module.training_step(batch, 0))
    assert_common_step_output(module.validation_step(batch, 0))
    assert_common_step_output(module.test_step(batch, 0))

    prediction = module.predict_step(batch, 0)
    assert prediction["prediction"].shape == (4,)
    assert prediction["weights"].shape == (4, 5)
