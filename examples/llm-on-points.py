# %%
labels = [f'T{i}' for i in range(10)]
scores = [{k: i for i, k in enumerate(labels)}] * 1000 + [{k: i for i, k in enumerate(reversed(labels))}] * 1000

# %% TOKENIZER
from ai4bmr_learn.models.tokenizer.ranked_based import RankBasedTokenizer

tokenizer = RankBasedTokenizer(labels=labels)

tokenizer.encode(scores[0])
tokenizer.encode(scores[-1])

# %%
import torch


class TokenizedDataset(torch.utils.data.Dataset):

    def __init__(self, scores, tokenizer):
        self.tokenizer = tokenizer
        self.scores = scores

    def __getitem__(self, idx):
        item = self.scores[idx]
        tokenized = self.tokenizer.encode(item)
        tokenized = {k: torch.tensor(v).long() for k, v in tokenized.items()}
        return tokenized

    def __len__(self):
        return len(self.scores)


ds_train = TokenizedDataset(scores=scores, tokenizer=tokenizer)
ds_test = TokenizedDataset(scores=scores, tokenizer=tokenizer)
inp = ds_train[0]

# %% MASKING
from ai4bmr_learn.transforms.mlm_collator import SimpleMLMCollator

data_collator = SimpleMLMCollator(tokenizer, mlm=True, mlm_probability=0.25)
batch = data_collator(ds_train)
assert torch.isclose(torch.tensor(0.75), (batch['labels'] == -100).float().mean(), atol=0.1)

# %% MODEL
from transformers import BertConfig, BertForMaskedLM, Trainer, TrainingArguments

config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=4,
    num_hidden_layers=4,
    num_attention_heads=2,
    intermediate_size=4,
    pad_token_id=tokenizer.pad_token_id
)
model = BertForMaskedLM(config)

# %%
from torch.utils.data import DataLoader
from ai4bmr_learn.plotting.umap import plot_umap

data_collator = SimpleMLMCollator(tokenizer)
dl = DataLoader(
    ds_test,
    batch_size=32,
    shuffle=False,
    collate_fn=data_collator
)

from ai4bmr_learn.utils.device import batch_to_device
def get_embedding(model, dataloader):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch_to_device(batch, device=model.device)
            out = model(**batch)
            cls_tokens = out.last_hidden_state[:, 0, :].cpu()
            embeddings.append(cls_tokens)

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings


embeddings = get_embedding(model.bert, dl)
ax = plot_umap(embeddings)
ax.figure.show()

# %% TRAIN
training_args = TrainingArguments(
    output_dir="./mlm-test",
    per_device_train_batch_size=16,
    num_train_epochs=200,
    logging_steps=10,
    report_to="none"
)

data_collator = SimpleMLMCollator(tokenizer, mlm=True, mlm_probability=0.25)
trainer = Trainer(
    model=model.train(),
    args=training_args,
    train_dataset=ds_train,
    data_collator=data_collator,
)

trainer.train()

# %%
embeddings = get_embedding(model.bert, dl)
ax = plot_umap(embeddings)
ax.figure.show()
