
class RankBasedTokenizer:
    def __init__(self, gene_list, max_length=128):
        self.gene_to_id = {gene: idx for idx, gene in enumerate(gene_list)}
        self.id_to_gene = {idx: gene for gene, idx in self.gene_to_id.items()}
        self.pad_token = "[PAD]"
        self.pad_token_id = len(gene_list)
        self.unk_token_id = len(gene_list) + 1
        self.max_length = max_length
        self.vocab_size = len(gene_list) + 2

    def __call__(self, expression_dict):
        # Sort genes by descending expression
        sorted_genes = sorted(
            ((g, v) for g, v in expression_dict.items() if g in self.gene_to_id),
            key=lambda x: -x[1]
        )
        token_ids = [self.gene_to_id[g] for g, _ in sorted_genes[:self.max_length]]

        # Padding
        if len(token_ids) < self.max_length:
            token_ids += [self.pad_token_id] * (self.max_length - len(token_ids))

        attention_mask = [1 if id != self.pad_token_id else 0 for id in token_ids]

        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask
        }

    def decode(self, token_ids):
        return [self.id_to_gene.get(i, "[UNK]") for i in token_ids]