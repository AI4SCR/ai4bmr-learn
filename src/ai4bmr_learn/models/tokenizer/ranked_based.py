# %%
import os
import json
import torch.nn as nn

class RankBasedTokenizer(nn.Module):
    def __init__(self, labels: list[str], label_key: str = 'label', pad_token='[PAD]', unk_token='[UNK]', mask_token='[MASK]', max_length=None):
        super().__init__()

        self.labels = sorted(set(labels))
        self.label_key = label_key

        self.unk_token_id = 0
        self.pad_token_id = 1
        self.mask_token_id = 2

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.mask_token = mask_token

        self.special_tokens_map = {
            self.unk_token: self.unk_token_id,
            self.pad_token: self.pad_token_id,
            self.mask_token: self.mask_token_id
        }
        offset = max(self.special_tokens_map.values()) + 1

        self.token_to_id = {**self.special_tokens_map}

        assert set(labels).intersection(self.token_to_id) == set()

        label_map = { k: v + offset for v, k in enumerate(self.labels) }
        self.token_to_id.update(label_map)

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)
        self.max_length = max_length or len(self.labels)

    def forward(self, item):
        scores = item['points'][self.label_key].value_counts().to_dict()
        tokenized = self.encode(scores)
        return tokenized

    def encode(self, scores):

        input_ids = [(self.token_to_id.get(token, self.unk_token_id), value, token) for token, value in scores.items()]
        input_ids = sorted(input_ids, key=lambda x: (-x[1], x[2]))
        input_ids = [token_id for token_id, _, _ in input_ids[:self.max_length]]

        pad_len = self.max_length - len(input_ids)
        input_ids += [self.pad_token_id] * pad_len
        attention_mask = [1 if i != self.pad_token_id else 0 for i in input_ids]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    def decode(self, token_ids):
        return [self.id_to_token.get(i, self.unk_token) for i in token_ids]

    def pad(self, encoded_inputs, padding=True, max_length=None, return_tensors=None):
        # Assume inputs already padded
        if not isinstance(encoded_inputs, list):
            raise ValueError("pad() expects a list of dicts.")

        if return_tensors is None:
            return encoded_inputs

        import torch
        import numpy as np

        batch = {
            k: [dic[k] for dic in encoded_inputs]
            for k in encoded_inputs[0]
        }

        if return_tensors == "pt":
            return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
        elif return_tensors == "np":
            return {k: np.array(v, dtype=np.int64) for k, v in batch.items()}
        else:
            raise ValueError(f"Unsupported return_tensors: {return_tensors}")

    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, "vocab.json"), "w") as f:
            json.dump(self.token_to_id, f)

        with open(os.path.join(save_dir, "tokenizer_config.json"), "w") as f:
            json.dump({
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "mask_token": self.mask_token,
                "pad_token_id": self.pad_token_id,
                "unk_token_id": self.unk_token_id,
                "mask_token_id": self.mask_token_id,
                "max_length": self.max_length
            }, f)

        with open(os.path.join(save_dir, "special_tokens_map.json"), "w") as f:
            json.dump({
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "mask_token": self.mask_token
            }, f)

    @classmethod
    def from_pretrained(cls, dir):
        with open(os.path.join(dir, "vocab.json")) as f:
            token_to_id = json.load(f)

        with open(os.path.join(dir, "tokenizer_config.json")) as f:
            config = json.load(f)

        # Build from saved vocab (not recomputed from sorted labels)
        instance = cls(
            labels=[],  # will override below
            pad_token=config["pad_token"],
            unk_token=config["unk_token"],
            mask_token=config["mask_token"],
            max_length=config.get("max_length")
        )
        instance.token_to_id = token_to_id
        instance.id_to_token = {v: k for k, v in token_to_id.items()}
        instance.vocab_size = len(token_to_id)

        instance.pad_token_id = config["pad_token_id"]
        instance.unk_token_id = config["unk_token_id"]
        instance.mask_token_id = config["mask_token_id"]

        return instance


# labels = ["A", "B", "C", "D", "E"]
# tokenizer = RankBasedTokenizer(labels=labels)
# scores = {"C": 3.2, "A": 1.5, "E": 2.9, "Z": 0.5}
# tokenizer.encode(scores)
#
# import random
# labels = ["A", "B", "C", "D", "E"]
# random.shuffle(labels)
# scores = {k: 1 for k in labels}
# tokenizer.encode(scores)
