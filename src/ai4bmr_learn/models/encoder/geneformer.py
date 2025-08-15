import torch
import torch.nn as nn
from loguru import logger

class Geneformer(nn.Module):
    def __init__(
        self,
        model_name: str = 'gf-12L-30M-i2048',
        expr_pooling: str = 'cell',
        mini_batch_size: int = -1,
        device: str = 'cuda',
        nproc: int = 8,
        fill_na: bool = False,
        **kwargs
    ):
        """
        The GeneformerHelical is a wrapper around the Geneformer model.
        Args:
            model_name: The path to the pretrained Geneformer model.
            mini_batch_size: The mini batch size for processing the data, if -1 the full batch is used.
            embed_layer: The layer of the Geneformer model to use for the output.
        """
        super().__init__()

        from helical.models.geneformer import Geneformer, GeneformerConfig

        # TODO: nproc could be a lever for speedup
        self.model_config = model_config = GeneformerConfig(model_name=model_name, device=device, nproc=nproc, **kwargs)
        # TODO set head to identity?
        self.model = Geneformer(model_config).model

        self.embed_dim = self.model.config.hidden_size
        self.mini_batch_size = mini_batch_size

        self.has_special_tokens = model_config.config['special_token']
        self.embed_layer = model_config.config['emb_layer']
        self.expr_pooling = expr_pooling  # model_config.config['emb_mode']

        self.fill_na = fill_na

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """
        Forward pass through the Geneformer model.
        Args:
            expr_tokens: The input expression tokens.
        Returns:
            The output of the Geneformer model.
        """
        bs, num_tokens, num_genes = input_ids.shape

        # explicitly handle the case for patches with no genes
        if num_genes == 0:
            logger.info('No genes in the expression tokens, returning zero embeddings.')
            return torch.zeros(bs, num_tokens, self.embed_dim, device=input_ids.device)

        mini_batch_size = input_ids.shape[0] if self.mini_batch_size == -1 else self.mini_batch_size
        input_ids = input_ids.reshape(-1, num_genes)
        batched_input_ids = input_ids.split(mini_batch_size, dim=0)

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(-1, num_genes)
            batched_attention_mask = attention_mask.split(mini_batch_size, dim=0)
        else:
            batched_attention_mask = [None] * len(batched_input_ids)

        embeddings = []
        for mini_input_ids, mini_attention_mask in zip(batched_input_ids, batched_attention_mask):
            out = self.model(
                input_ids=mini_input_ids,
                attention_mask=mini_attention_mask,
            )

            out = pool_embeddings(
                out.hidden_states[self.embed_layer],
                pooling=self.expr_pooling,
                attention_masks=mini_attention_mask,
                has_special_tokens=self.has_special_tokens,
            )  # (mini_batch_size, geneformer_dim)
            embeddings.append(out)

        embeddings = torch.cat(embeddings, dim=0)  # (bs * num_tokens, geneformer_dim)
        embeddings = torch.nan_to_num(embeddings, nan=0.0) if self.fill_na else embeddings
        return embeddings.reshape(bs, num_tokens, -1)


def pool_embeddings(
    embeddings: torch.Tensor,
    pooling: str = 'cell',
    attention_masks: torch.Tensor | None = None,
    has_special_tokens: bool = False,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Pool the embeddings using the specified pooling method.
    Args:
        embeddings: The input embeddings, shape (batch_size, num_tokens, embedding_dim).
        pooling: The pooling method to use.
        attention_masks: The attention mask to use for pooling, shape (batch_size, num_tokens).
        has_special_tokens: Whether the embeddings have special tokens, if True, the first and last tokens are excluded from pooling.
    Returns:
        The pooled embeddings.
    """
    if pooling == 'cls':
        assert has_special_tokens, f'`{pooling}` is not a valid pooling method for models without special token'
        return embeddings[:, 0]  # batch_size, embedding_dim
    elif pooling == 'cell':
        if attention_masks is None:
            attention_masks = torch.ones(embeddings.shape[0], embeddings.shape[1], device=embeddings.device)
        embeddings = embeddings * attention_masks.unsqueeze(-1)
        if has_special_tokens:
            embeddings = embeddings[:, 1:-1]
            attention_masks = attention_masks[:, 1:-1]
        return embeddings.sum(dim=1) / (attention_masks.sum(dim=1, keepdim=True) + eps)
    else:
        raise NotImplementedError(f'{pooling} is not implemented.')
