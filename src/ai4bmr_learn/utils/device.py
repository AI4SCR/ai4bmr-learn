import torch


def batch_to_device(batch, device):

    if isinstance(batch, torch.Tensor):
        return batch.to(device)

    if isinstance(batch, (list, tuple)):
        return [batch_to_device(batch, device) for batch in batch]

    if isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k,v in batch.items()}


def get_device(logger=None):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        if logger is not None:
            logger.info('CUDA is available. Using GPU.')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        if logger is not None:
            logger.info('MPS is available. Using Apple Silicon GPU.')
    else:
        device = torch.device('cpu')
        if logger is not None:
            logger.info('No GPU available. Using CPU.')
    return device