def collect_model_stats(model):
    from torchinfo import summary

    model_summary = summary(model, verbose=0)

    # Accessing stored values
    total_params = model_summary.total_params
    trainable_params = model_summary.trainable_params
    non_trainable_params = total_params - trainable_params
    return dict(
        total_params=total_params,
        trainable_params=trainable_params,
        non_trainable_params=non_trainable_params,
    )
