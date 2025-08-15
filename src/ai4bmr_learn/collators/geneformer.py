import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
from helical.models.geneformer.geneformer_tokenizer import (  # pants: no-infer-dep # pyright: ignore[reportMissingImports]
    TranscriptomeTokenizer,  # pants: no-infer-dep # pyright: ignore[reportMissingImports]
)
from torch.utils.data._utils.collate import collate, default_collate_fn_map

from ai4bmr_learn.utils.points import compute_points_tokens, to_csr

class BaseCollate:
    def __init__(
        self, kernel_size: int, stride: int, group_by: str = 'ensembl_id', ensembl_ids_path: Path | None = None
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.group_by = group_by
        if ensembl_ids_path is not None:
            assert ensembl_ids_path.exists()
        self.ensembl_ids_path = ensembl_ids_path
        self.ensembl_dict: dict | None = None

        self.collate_fn_map = default_collate_fn_map.copy()
        self.collate_fn_map.update({scipy.sparse.csr_matrix: self.tokenize_batch})

    def tokenize_batch(
        self, expressions: list[scipy.sparse.csr_matrix], collate_fn_map=None
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError()

    def process_batch(self, batch):
        new_batch = []

        for batch_item in batch:
            if 'views' in batch_item:
                views = [self.process_item(item) for item in batch_item['views']]
                batch_item['views'] = views
            else:
                batch_item = self.process_item(batch_item)

            new_batch.append(batch_item)

        return new_batch

    def process_item(self, item: dict) -> dict:

        tokens = self.convert_points_to_tokens(item)
        del item['points']

        csr = to_csr(tokens, gene_dict=self.ensembl_dict)  # pyright: ignore
        assert not np.isnan(csr.data).any(), 'Token-level points data contains NaN values after conversion to CSR.'

        item['expression'] = csr
        return item

    def convert_points_to_tokens(self, item: dict) -> pd.DataFrame:
        points = item['points']
        patch_size = item['patch_size']
        return compute_points_tokens(
            points, patch_size=patch_size, kernel_size=self.kernel_size, stride=self.stride, group_by=self.group_by
        )

    def get_gene_dict(self, batch: list[torch.Tensor] | list[dict] | None = None) -> dict:
        if batch is not None:
            try:
                ensembl_ids = set()
                for batch_item in batch:
                    if 'views' in batch_item:
                        ensembl_ids.update(
                            sum([view['points'][self.group_by].unique().tolist() for view in batch_item['views']], [])  # pyright: ignore
                        )
                    else:
                        ensembl_ids.update(batch_item['points'][self.group_by].unique().tolist())  # pyright: ignore

                ensembl_dict = {k: i for i, k in enumerate(sorted(ensembl_ids))}
            except IndexError:
                ensembl_dict = {}
            return ensembl_dict

        if self.ensembl_dict is not None:
            return self.ensembl_dict

        if self.ensembl_ids_path is not None:
            with open(self.ensembl_ids_path) as f:
                ensembl_ids = json.load(f)

            self.ensembl_dict = {v: k for k, v in enumerate(sorted(ensembl_ids))}
            return self.ensembl_dict
        else:
            raise ValueError(
                'No ensembl_ids_path provided and no batch provided to extract ensembl_ids from.Please provide either.'
            )

    def collate(self, batch):
        batch = self.process_batch(batch)
        result: dict = collate(batch, collate_fn_map=self.collate_fn_map)  # pyright: ignore
        return result

    def __call__(self, batch):
        return batch


class GeneformerCollate(BaseCollate):
    def __init__(
        self,
        kernel_size: int,
        stride: int,
        model_name: str = 'gf-6L-30M-i2048',
        batch_size: int = 256,
        max_len: int | None = None,
    ):
        super().__init__(kernel_size=kernel_size, stride=stride, ensembl_ids_path=None)
        self.tk = get_geneformer_tokenizer(model_name=model_name, batch_size=batch_size)
        self.max_len = max_len

    def tokenize_batch(
        self, input_ids: list[scipy.sparse.csr_matrix], collate_fn_map=None
    ) -> dict[str, torch.Tensor]:
        """
        Tokenize and collate the expressions and gene names.
        Args:
            input_ids: The expression data, (num_tokens, num_ensembl).
        Returns:
            The tokenized and collated expressions and gene names.
        """
        import anndata

        tk = self.tk

        num_tokens = input_ids[0].shape[0]
        x = scipy.sparse.vstack(input_ids)

        adata = anndata.AnnData(X=x)
        adata.var['ensembl_id'] = self.ensembl_dict.keys()  # pyright: ignore
        adata.obs['filter_pass'] = True  # silence `has no column attribute 'filter_pass'`
        input_ids, cell_metadata = tk.tokenize_anndata(adata)  # type: ignore

        # note: largets input_id length but not more than the model input size
        max_len = self.max_len if self.max_len is not None else max(map(len, input_ids))
        max_len = min(tk.model_input_size, max_len)  # type: ignore

        pad_token_id = tk.gene_token_dict.get('<pad>')  # type: ignore
        input_ids = [torch.tensor(i).long() for i in input_ids]
        input_ids = pad_input_ids(input_ids=input_ids, pad_token_id=pad_token_id, max_len=max_len)
        input_ids = torch.stack(input_ids)  # type: ignore
        assert not torch.isnan(input_ids).any(), 'Stacked expression data contains NaN values.'  # pyright: ignore
        attention_mask = get_attention_mask_for_padded_tensors(input_ids=input_ids, pad_token_id=pad_token_id)  # type: ignore

        batch_size = input_ids.shape[0] // num_tokens  # type: ignore
        input_ids = input_ids.view(batch_size, num_tokens, -1)  # type: ignore
        attention_mask = attention_mask.view(batch_size, num_tokens, -1)

        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def __call__(self, batch):
        self.ensembl_dict = self.get_gene_dict(batch)

        result = self.collate(batch)

        # NOTE: since the kaiko.ml_framework concats views we need to enforce a homogenous max_len across views
        #   batch might contain only torch tensors, in this case we do not need to pad
        if 'views' in result and isinstance(result['views'][0], dict) and 'input_ids' in result['views'][0]:
            max_len = max(map(lambda x: x['input_ids'].shape[-1], result['views']))
            for view in result['views']:
                batch_size, num_tokens, num_ids = view['input_ids'].shape

                input_ids = view['input_ids'].view(-1, num_ids)
                input_ids = pad_input_ids(
                    input_ids, pad_token_id=self.tk.gene_token_dict.get('<pad>'), max_len=max_len
                )
                input_ids = torch.stack(input_ids).view(batch_size, num_tokens, max_len)

                attention_mask = view['attention_mask'].view(-1, num_ids)
                attention_mask = pad_input_ids(
                    attention_mask, pad_token_id=self.tk.gene_token_dict.get('<pad>'), max_len=max_len
                )
                attention_mask = torch.stack(attention_mask).view(batch_size, num_tokens, max_len)

                view['input_ids'] = input_ids
                view['attention_mask'] = attention_mask

        return result


def pad_input_ids(input_ids: list[torch.Tensor], pad_token_id: int, max_len: int | None = None) -> list[torch.Tensor]:
    max_len = max_len or max(map(len, input_ids))

    return [
        torch.nn.functional.pad(tensor, pad=(0, max_len - tensor.numel()), mode='constant', value=pad_token_id)
        for tensor in input_ids
    ]


def get_attention_mask_for_padded_tensors(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    attention_mask = np.ones_like(input_ids)
    attention_mask[input_ids == pad_token_id] = 0  # type: ignore
    attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
    return attention_mask


def get_geneformer_tokenizer(model_name: str = 'gf-6L-30M-i2048', batch_size: int = 10):
    from helical.models.geneformer import (  # pants: no-infer-dep # pyright: ignore[reportMissingImports]
        GeneformerConfig,  # pants: no-infer-dep # pyright: ignore[reportMissingImports]
    )

    config = GeneformerConfig(model_name=model_name, batch_size=batch_size)
    tk = TranscriptomeTokenizer(
        custom_attr_name_dict=config.config['custom_attr_name_dict'],
        nproc=config.config['nproc'],
        model_input_size=config.config['input_size'],
        special_token=config.config['special_token'],
        gene_median_file=config.files_config['gene_median_path'],
        token_dictionary_file=config.files_config['token_path'],
        gene_mapping_file=config.files_config['ensembl_dict_path'],
    )
    return tk
