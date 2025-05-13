import torch

class SimpleMLMCollator:
    def __init__(self, tokenizer, mlm: bool = False, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, examples):
        input_ids = torch.stack([x["input_ids"] for x in examples])
        attention_mask = torch.stack([x["attention_mask"] for x in examples])

        if not self.mlm:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        labels = input_ids.clone()

        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
        special_tokens_mask = input_ids == self.pad_token_id
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        input_ids[masked_indices] = self.mask_token_id
        labels[~masked_indices] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
