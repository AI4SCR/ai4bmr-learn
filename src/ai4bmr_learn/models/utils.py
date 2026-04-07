from __future__ import annotations


def collect_model_stats(model) -> dict[str, int]:
    from torchinfo import summary

    model_summary = summary(model, verbose=0)
    total_params = model_summary.total_params
    trainable_params = model_summary.trainable_params
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": total_params - trainable_params,
    }
