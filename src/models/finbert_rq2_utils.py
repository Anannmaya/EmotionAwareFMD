"""Shared utilities for the final paired FinBERT RQ2 experiment."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, wilcoxon
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, get_linear_schedule_with_warmup

from src.models.run_rq2_cross_validation import AFFECT_COLUMNS, calculate_metrics


PRIMARY_METRIC = "macro_f1"
SECONDARY_METRIC = "balanced_accuracy"
MODEL_VARIANTS = ["text_only", "emotion_aware"]


@dataclass(frozen=True)
class TrainingConfig:
    checkpoint: str = "ProsusAI/finbert"
    max_length: int = 256
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_epochs: int = 5
    patience: int = 2
    dropout: float = 0.10
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    num_workers: int = 0


class TokenizedDataset(Dataset):
    def __init__(
        self,
        texts: Iterable[str],
        labels: np.ndarray,
        affect: np.ndarray,
        row_ids: Iterable[object],
        pair_ids: Iterable[object],
        tokenizer,
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(
            [str(text) for text in texts],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.affect = torch.as_tensor(affect, dtype=torch.float32)
        self.row_ids = [str(value) for value in row_ids]
        self.pair_ids = [str(value) for value in pair_ids]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        item["affect"] = self.affect[index]
        item["row_id"] = self.row_ids[index]
        item["pair_id"] = self.pair_ids[index]
        return item


class FinBertClassifier(nn.Module):
    """End-to-end FinBERT with optional direct affective feature fusion."""

    def __init__(self, checkpoint: str, emotion_aware: bool, dropout: float):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(checkpoint)
        self.emotion_aware = emotion_aware
        hidden = int(self.encoder.config.hidden_size)
        input_size = hidden + (len(AFFECT_COLUMNS) if emotion_aware else 0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_size, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        affect: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        cls = self.encoder(**kwargs).last_hidden_state[:, 0, :]
        if self.emotion_aware:
            cls = torch.cat([cls, affect], dim=1)
        return self.classifier(self.dropout(cls))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pair_preserving_split(
    df: pd.DataFrame,
    pair_column: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    pairs = df[pair_column].drop_duplicates().to_numpy(copy=True)
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)
    n_test = max(1, round(len(pairs) * 0.10))
    n_validation = max(1, round(len(pairs) * 0.10))
    n_train = len(pairs) - n_validation - n_test
    train_pairs = pairs[:n_train]
    validation_pairs = pairs[n_train : n_train + n_validation]
    test_pairs = pairs[n_train + n_validation :]

    def indices(selected: np.ndarray) -> np.ndarray:
        return df.index[df[pair_column].isin(set(selected.tolist()))].to_numpy()

    assignment = pd.DataFrame(
        {
            "pair_id": np.concatenate(
                [train_pairs, validation_pairs, test_pairs]
            ),
            "split": (
                ["train"] * len(train_pairs)
                + ["validation"] * len(validation_pairs)
                + ["test"] * len(test_pairs)
            ),
        }
    )
    return indices(train_pairs), indices(validation_pairs), indices(test_pairs), assignment


def make_loader(
    dataset: Dataset,
    config: TrainingConfig,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _move(batch: dict[str, object], device: torch.device):
    inputs = {
        key: batch[key].to(device)
        for key in ["input_ids", "attention_mask", "token_type_ids"]
        if key in batch
    }
    return inputs, batch["labels"].to(device), batch["affect"].to(device)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    labels, predictions, probabilities = [], [], []
    row_ids, pair_ids = [], []
    total_loss = 0.0
    for batch in loader:
        inputs, y_true, affect = _move(batch, device)
        logits = model(**inputs, affect=affect)
        total_loss += float(loss_function(logits, y_true).cpu())
        labels.append(y_true.cpu().numpy())
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        row_ids.extend(batch["row_id"])
        pair_ids.extend(batch["pair_id"])
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(predictions)
    metrics = calculate_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(y_true)
    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "probability": np.concatenate(probabilities),
        "row_ids": [str(value) for value in row_ids],
        "pair_ids": [str(value) for value in pair_ids],
    }


def train(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    model_variant: str,
    config: TrainingConfig,
    seed: int,
    device: torch.device,
):
    set_seed(seed)
    model = FinBertClassifier(
        config.checkpoint,
        emotion_aware=model_variant == "emotion_aware",
        dropout=config.dropout,
    ).to(device)
    train_loader = make_loader(train_dataset, config, True, seed)
    validation_loader = make_loader(validation_dataset, config, False, seed)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = math.ceil(
        len(train_loader) / config.gradient_accumulation_steps
    )
    total_steps = max(1, steps_per_epoch * config.max_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(total_steps * config.warmup_ratio),
        total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_function = nn.CrossEntropyLoss()
    best_key, best_state, best_epoch = None, None, 0
    patience_count = 0
    history = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, total_examples = 0.0, 0
        for step, batch in enumerate(train_loader, start=1):
            inputs, labels, affect = _move(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(**inputs, affect=affect)
                loss = loss_function(logits, labels)
                scaled_loss = loss / config.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            if step % config.gradient_accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().cpu()) * len(labels)
            total_examples += len(labels)

        validation = evaluate(model, validation_loader, device)
        validation_metrics = validation["metrics"]
        train_loss = total_loss / total_examples
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_metrics["loss"],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "validation_balanced_accuracy": validation_metrics[
                    "balanced_accuracy"
                ],
            }
        )
        print(
            f"  epoch {epoch}/{config.max_epochs}: "
            f"train loss={train_loss:.4f}, "
            f"validation macro-F1={validation_metrics['macro_f1']:.4f}"
        )
        key = (
            validation_metrics[PRIMARY_METRIC],
            validation_metrics[SECONDARY_METRIC],
            -validation_metrics["loss"],
        )
        if best_key is None or key > best_key:
            best_key, best_epoch, patience_count = key, epoch, 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            patience_count += 1
            if patience_count >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("No valid FinBERT checkpoint was produced.")
    model.load_state_dict(best_state)
    return model.to(device), history, best_epoch


def rank_biserial(differences: np.ndarray) -> float:
    differences = differences[differences != 0]
    if len(differences) == 0:
        return 0.0
    ranks = rankdata(np.abs(differences))
    positive = ranks[differences > 0].sum()
    negative = ranks[differences < 0].sum()
    return float((positive - negative) / (positive + negative))


def paired_test(emotion: np.ndarray, text: np.ndarray, alternative: str):
    differences = emotion - text
    if np.allclose(differences, 0):
        return 0.0, 1.0, 0.0
    result = wilcoxon(
        emotion,
        text,
        alternative=alternative,
        zero_method="wilcox",
    )
    return float(result.statistic), float(result.pvalue), rank_biserial(differences)


def holm(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()
