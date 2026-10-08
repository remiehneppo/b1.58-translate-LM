import torch
from typing import List, Dict
from torch.utils.data import Dataset
from .model import RuViBitNet10M, RuViConfig


class TranslationPrefixDataset(Dataset):
    """
    Format: [TAG] src_tokens [SEP] tgt_tokens [EOS]
    Labels: [-100 ... -100] [SEP] tgt_tokens [EOS]
    """
    def __init__(self, samples: List[Dict[str, any]], config: RuViConfig):
        self.samples = samples
        self.cfg = config

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        src_ids = item["src_ids"]
        tgt_ids = item["tgt_ids"]
        tag_id = self.cfg.tag_ru_id if item["direction"] == "vi2ru" else self.cfg.tag_vi_id

        # Fit [TAG] src [SEP] tgt [EOS] into max_seq_len, trimming the source before the target.
        budget = self.cfg.max_seq_len - 3
        if budget < 1:
            raise ValueError("max_seq_len must be at least 4 to hold [TAG] src [SEP] tgt [EOS]")
        if len(src_ids) + len(tgt_ids) > budget:
            tgt_ids = tgt_ids[: max(1, budget - len(src_ids))]
            src_ids = src_ids[: budget - len(tgt_ids)]

        # Prefix: [TAG] + src + [SEP]
        prefix = [tag_id] + src_ids + [self.cfg.sep_token_id]
        # Target: tgt + [EOS]
        target = tgt_ids + [self.cfg.eos_token_id]

        input_ids = prefix + target
        prefix_len = len(prefix)

        # Labels: -100 for all prefix tokens
        labels = [-100] * prefix_len + target
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "prefix_len": prefix_len,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_prefix_fn(batch: List[Dict], pad_token_id: int = 0):
    if not batch:
        raise ValueError("batch must not be empty")
    max_len = max(len(x["input_ids"]) for x in batch)
    B = len(batch)

    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    prefix_lens = []

    # Bidirectional prefix + causal target mask
    mask = torch.zeros((B, 1, max_len, max_len), dtype=torch.bool)

    for i, item in enumerate(batch):
        seq_len = len(item["input_ids"])
        p_len = item["prefix_len"]
        prefix_lens.append(p_len)

        input_ids[i, :seq_len] = item["input_ids"]
        labels[i, :seq_len] = item["labels"]

        # Prefix attends to prefix
        mask[i, 0, :p_len, :p_len] = True
        # Target attends to prefix
        mask[i, 0, p_len:seq_len, :p_len] = True
        # Target attends to past target
        target_len = seq_len - p_len
        if target_len > 0:
            mask[i, 0, p_len:seq_len, p_len:seq_len] = torch.tril(torch.ones(target_len, target_len, dtype=torch.bool))

    return {
        "input_ids": input_ids,
        "labels": labels,
        "prefix_lens": prefix_lens,
        "mask": mask,
    }


def train_step(model: RuViBitNet10M, optimizer: torch.optim.Optimizer, batch: Dict) -> float:
    optimizer.zero_grad()
    device = next(model.parameters()).device
    out = model(
        input_ids=batch["input_ids"].to(device),
        mask=batch["mask"].to(device),
        labels=batch["labels"].to(device),
    )
    loss = out["loss"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()
