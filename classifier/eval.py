#!/usr/bin/env python3
"""Eval script."""


# stdlib
import os
import json
import logging
from dataclasses import dataclass, field
from collections import Counter
from typing import cast, List, Dict, Callable, Optional, Any

# numpy
import numpy as np

# plot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# pytorch
import torch

# transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    HfArgumentParser,
    EvalPrediction,
    Trainer,
    DataCollatorWithPadding,
)

# local
from data import (
    MulticlassTSVFileProcessor,
    MulticlassTestDataset,
    MulticlassTSVProcessor,
    MulticlassDataset,
)


@dataclass
class DataArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    data_dir: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )

    supervised_test_file: str = field(
        metadata={"help": "Path to the test file"},
    )
    max_seq_length: int = field(
        default=0,
        metadata={"help": "Max sequence length"},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    # model path is required and does not have a default value
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model."},
    )


# setup logging
# handlers=[logging.StreamHandler()],
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def make_compute_metrics_fn(
    target_names,
    results_dir=None,
) -> Callable[[EvalPrediction], Dict[str, float]]:
    """Make compute metrics function.

    target_ids and targets_names are corresponding lists of selected ids and names.
    """

    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, Any]:
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

        # f1 micro without Other
        background = precision_recall_fscore_support(
            y_true=labels, y_pred=preds, average="micro", labels=[0, 1]
        )
        metrics["p_b_micro"] = background[0]
        metrics["r_b_micro"] = background[1]
        metrics["f1_b_micro"] = background[2]

        # f1 macro without Other
        background = precision_recall_fscore_support(
            y_true=labels, y_pred=preds, average="macro", labels=[0, 1]
        )
        metrics["p_b_macro"] = background[0]
        metrics["r_b_macro"] = background[1]
        metrics["f1_b_macro"] = background[2]

        # accuracy
        acc = accuracy_score(y_true=labels, y_pred=preds)
        metrics["acc"] = acc

        print("\nMetrics")
        print(metrics)
        if results_dir is not None:
            with open(os.path.join(results_dir, "res.txt"), "w") as fout:
                print(json.dumps(metrics), file=fout)

        # classification report
        report = classification_report(
            y_true=labels,
            y_pred=preds,
            target_names=target_names,
            output_dict=False,
        )
        if results_dir is not None:
            with open(
                os.path.join(results_dir, "full_report.txt"), "w"
            ) as fout:
                print(report, file=fout)

        # confusion matrix
        cm = confusion_matrix(y_true=labels, y_pred=preds, normalize="true")
        cm_list = cm.tolist()
        cm = np.around(cm, decimals=2)
        if results_dir is not None:
            with open(
                os.path.join(results_dir, "confusion_matrix.txt"), "w"
            ) as fout:
                for row in cm_list:
                    print("\t".join([str(x) for x in row]), file=fout)

        # confusion matrix plot
        if results_dir is not None:
            sns.set(font_scale=2)
            plt.figure(figsize=(11, 10))
            g = sns.heatmap(
                cm,
                annot=True,
                fmt="g",
                cmap="Blues",
                xticklabels=["Rare", "Non-Rare", "Other"],
                yticklabels=["Rare", "Non-Rare", "Other"],
            )
            plt.xlabel("Predicted labels")
            plt.ylabel("True labels")
            plt.tight_layout()

            # done
            plt.show(block=False)
            plt.savefig(
                os.path.join(results_dir, "confusion_matrix.png"), dpi=600
            )

        return metrics

    return compute_metrics


def main():
    # parse args
    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    parser = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments))  # type: ignore
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()
    data_dir = data_args.data_dir
    output_dir = training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # tokenizer and model
    tkz = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=True
    )
    collator_fn = DataCollatorWithPadding(
        tkz, padding=True, pad_to_multiple_of=8
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path
    )
    # model = cast(RobertaForSequenceClassification, model)
    label2id = model.config.label2id  # type: ignore
    id2label = model.config.id2label  # type: ignore
    # target names is the list of all labels in order of id
    target_names = [id2label[x] for x in range(len(label2id))]

    # All supervised
    all_sup_file = data_args.supervised_test_file
    sup_processor = MulticlassTSVFileProcessor(
        filepath=all_sup_file, use_tqdm=True
    )
    ds_sup = MulticlassTestDataset(
        processor=sup_processor,
        tokenizer=tkz,
        label2id=label2id,
        max_seq_length=data_args.max_seq_length,
        use_tqdm=True,
    )

    processor = MulticlassTSVProcessor(data_dir)
    ds_val = MulticlassDataset(
        processor=processor,
        label2id=label2id,
        tokenizer=tkz,
        use_tqdm=False,
        mode="val",
    )
    ds_tst = MulticlassDataset(
        processor=processor,
        label2id=label2id,
        tokenizer=tkz,
        use_tqdm=False,
        mode="tst",
    )

    out_sup = os.path.join(output_dir, "supervised")
    os.makedirs(out_sup, exist_ok=True)
    compute_metrics_sup_fn = make_compute_metrics_fn(
        target_names=target_names,
        results_dir=out_sup,
    )

    out_val = os.path.join(output_dir, "val")
    os.makedirs(out_val, exist_ok=True)
    compute_metrics_val_fn = make_compute_metrics_fn(
        target_names=target_names,
        results_dir=out_val,
    )
    out_tst = os.path.join(output_dir, "tst")
    os.makedirs(out_tst, exist_ok=True)
    compute_metrics_tst_fn = make_compute_metrics_fn(
        target_names=target_names,
        results_dir=out_tst,
    )

    # evaluate the model
    model = model.eval()
    # convert to cuda
    cuda = torch.device("cuda")
    model.to(cuda)  # type: ignore

    print("####### News Supervised Test Set #######")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator_fn,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics_sup_fn,
    )
    trainer.evaluate(eval_dataset=ds_sup)

    print("####### Validation Set #######")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator_fn,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics_val_fn,
    )
    trainer.evaluate(eval_dataset=ds_val)

    print("####### Test Set #######")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator_fn,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics_tst_fn,
    )
    trainer.evaluate(eval_dataset=ds_tst)


if __name__ == "__main__":
    main()
