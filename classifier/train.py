#!/usr/bin/env python3
"""Train script."""

# stdlib
import os
import json
import logging
from dataclasses import dataclass, field
from typing import cast, Dict

# torch and sklearn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# transformers
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import HfArgumentParser, TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
from transformers import EvalPrediction


# local
from data import (
    MulticlassTSVProcessor,
    MulticlassDataset,
)

# setup logging
# handlers=[logging.StreamHandler()],
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# setup labels
LABEL2ID = {
    "RAD": 0,
    "NRAD": 1,
    "O": 2,
}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="roberta-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    data_dir: str = field(
        default="/data/midas3/dataset",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    max_seq_length: int = field(
        default=0,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    use_tqdm: bool = field(
        default=True,
        metadata={
            "help": "Whether to use tqdm for progress bar. "
            "Defaults to False.",
        },
    )


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute metrics."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # general metrics
    metrics = {}

    # micro precision, recall, f1
    micro = precision_recall_fscore_support(
        y_true=labels, y_pred=preds, average="micro"
    )
    metrics["p_micro"] = micro[0]
    metrics["r_micro"] = micro[1]
    metrics["f1_micro"] = micro[2]

    # macro precision, recall, f1
    macro = precision_recall_fscore_support(
        y_true=labels, y_pred=preds, average="macro"
    )
    metrics["p_macro"] = macro[0]
    metrics["r_macro"] = macro[1]
    metrics["f1_macro"] = macro[2]

    # f1 without Other
    background = precision_recall_fscore_support(
        y_true=labels, y_pred=preds, average="micro", labels=[0, 1]
    )
    metrics["p_b"] = background[0]
    metrics["r_b"] = background[1]
    metrics["f1_b"] = background[2]

    # accuracy
    acc = accuracy_score(y_true=labels, y_pred=preds)
    metrics["acc"] = acc

    return metrics


def main():
    # parse args
    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    parser = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments))  # type: ignore
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()

    """Settings"""
    output_dir = training_args.output_dir
    data_dir = data_args.data_dir
    model_name = model_args.model_name_or_path
    max_seq_length = data_args.max_seq_length

    os.makedirs(output_dir, exist_ok=True)
    savepath = output_dir

    """Tokenizer"""
    tkz = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, local_files_only=False
    )
    """Processor and Dataset"""
    logger.info("Creating datasets")
    processor = MulticlassTSVProcessor(data_dir)

    ds_trn = MulticlassDataset(
        processor=processor,
        label2id=LABEL2ID,
        tokenizer=tkz,
        mode="trn",
        max_seq_length=max_seq_length,
    )

    ds_val = MulticlassDataset(
        processor=processor,
        label2id=LABEL2ID,
        tokenizer=tkz,
        mode="val",
        max_seq_length=max_seq_length,
        use_tqdm=False,
    )
    ds_tst = MulticlassDataset(
        processor=processor,
        label2id=LABEL2ID,
        tokenizer=tkz,
        mode="tst",
        max_seq_length=max_seq_length,
        use_tqdm=False,
    )

    num_labels = ds_trn.num_labels
    id2label = ds_trn.id2label
    label2id = ds_trn.label2id

    print(f"Train Dataset num_labels: {num_labels}")
    print(f"Train Dataset num_samples: {len(ds_trn)}")
    print(f"Val Dataset num_samples: {len(ds_val)}")
    print(f"Test Dataset num_samples: {len(ds_tst)}")

    """Model and Training"""
    # print(training_args)
    collator_fn = DataCollatorWithPadding(
        tkz, padding=True, pad_to_multiple_of=8
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
        local_files_only=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator_fn,
        train_dataset=ds_trn,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
    )

    """Training"""
    if training_args.do_train:
        logger.info("Training")
        trainer.train()
        if savepath is not None:
            model.save_pretrained(savepath)
            tkz.save_pretrained(savepath)

    """Evaluation"""
    if training_args.do_eval:
        logger.info("Evaluation: Val")

        # evaluate on dev set
        res = trainer.predict(test_dataset=ds_val, metric_key_prefix="val")
        print(res.metrics)
        with open(os.path.join(savepath, "val_results.txt"), "w") as f:
            f.write(json.dumps(res.metrics))

    # evaluate on gold test set
    if training_args.do_predict:
        logger.info("Evaluation: Test")
        res = trainer.predict(test_dataset=ds_tst, metric_key_prefix="tst")
        print(res.metrics)
        with open(os.path.join(savepath, "tst_results.txt"), "w") as f:
            f.write(json.dumps(res.metrics))


if __name__ == "__main__":
    main()
