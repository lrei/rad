"""A processor and dataset for pytorch/huggingface's transformers library."""

import logging
from copy import deepcopy
from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Union, Dict, Any, Iterator, Callable

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike

import torch
from torch.utils.data.dataset import Dataset
from transformers import DataProcessor  # type: ignore
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast  # type: ignore
from tqdm import tqdm


logger = logging.getLogger(__name__)


def _read_tsv_file(
    filepath, use_tqdm: bool = False
) -> Tuple[List[str], List[str]]:
    """Read data file.

    Data files are expected to be in the format:
    id\ttext\tlabel
    """
    logger.info(f"Reading {filepath}")

    texts = []
    labels = []

    with open(filepath, "r") as fin:
        fin.readline()  # skip header
        if use_tqdm:
            fin = tqdm(fin, desc=f"reading {filepath}", dynamic_ncols=True)
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            assert len(parts) == 3
            text = parts[1].strip()
            texts.append(text)
            label = parts[2].strip()
            labels.append(label)
    return texts, labels


class MulticlassTSVProcessor(DataProcessor):
    """Processor for multiclass classification datasets stored as TSV files."""

    def __init__(
        self,
        data_dir: str,
        use_tqdm: bool = False,
    ):
        """See class."""
        super(MulticlassTSVProcessor, self).__init__()
        self.data_dir = data_dir
        self.use_tqdm = use_tqdm

    @property
    def trn_filepath(self) -> str:
        """Get train filepath."""
        return f"{self.data_dir}/trn.tsv"

    @property
    def val_filepath(self) -> str:
        """Get val filepath."""
        return f"{self.data_dir}/val.tsv"

    @property
    def tst_filepath(self) -> str:
        """Get test filepath."""
        return f"{self.data_dir}/tst.tsv"

    def get_trn(self) -> Tuple[List[str], List[str]]:
        """Get train."""
        filepath = self.trn_filepath
        return _read_tsv_file(filepath, use_tqdm=self.use_tqdm)

    def get_val(self) -> Tuple[List[str], List[str]]:
        """Get val."""
        filepath = self.val_filepath
        return _read_tsv_file(filepath, use_tqdm=self.use_tqdm)

    def get_tst(self) -> Tuple[List[str], List[str]]:
        """Get test."""
        filepath = self.tst_filepath
        return _read_tsv_file(filepath, use_tqdm=self.use_tqdm)


class MulticlassTSVFileProcessor(DataProcessor):
    """Processor for multiclass classification datasets stored as TSV files."""

    def __init__(
        self,
        filepath: str,
        use_tqdm: bool = False,
    ):
        """See class."""
        super(MulticlassTSVFileProcessor, self).__init__()
        self.filepath = filepath
        self.use_tqdm = use_tqdm

    def get_data(self) -> Tuple[List[str], List[str]]:
        """Get data."""
        return _read_tsv_file(self.filepath, use_tqdm=self.use_tqdm)


class MulticlassDataset(Dataset):
    """Dataset for multiclass classification."""

    tkz: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    _label2id: Dict[str, int]
    _id2label: Dict[int, str]

    texts: List[str]
    labels: List[int]

    def __init__(
        self,
        processor: MulticlassTSVProcessor,
        label2id: Dict[str, int],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_seq_length: int = 0,
        mode: str = "trn",
        use_tqdm: bool = False,
    ):
        """See class."""
        self.use_tqdm = use_tqdm
        self.index = None
        self.max_seq_length = max_seq_length
        if max_seq_length <= 0:
            self.max_seq_length = tokenizer.max_len_single_sentence
        logger.info(f"Using max length={self.max_seq_length}")

        self.label2id = label2id

        if mode == "trn":
            self.texts, labels = processor.get_trn()
            self.labels = [self.label2id[label] for label in labels]
            del labels
        elif mode == "val":
            self.texts, labels = processor.get_val()
            self.labels = [self.label2id[label] for label in labels]
            del labels
        elif mode == "tst":
            self.texts, labels = processor.get_tst()
            self.labels = [self.label2id[label] for label in labels]
            del labels
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self.n = len(self.texts)
        self.tkz = tokenizer

    @property
    def n(self) -> int:
        return self._n

    @n.setter
    def n(self, n: int) -> None:
        assert isinstance(n, int)
        assert n > 0
        self._n = n

    def __len__(self):
        """Length of dataset corresponds to the number of examples."""
        return self.n

    @property
    def id2label(self) -> Dict[int, str]:
        return deepcopy(self._id2label)

    # create the setter method
    @id2label.setter
    def id2label(self, id2label: Dict[int, str]) -> None:
        # check types
        assert isinstance(id2label, dict)
        assert all([isinstance(k, int) for k in id2label.keys()])
        assert all([isinstance(v, str) for v in id2label.values()])
        # check uniqueness
        assert len(id2label) == len(set(id2label.keys()))
        # check countinuous ids starting with 0
        assert set(id2label.keys()) == set(range(len(id2label)))

        # set
        self._id2label = deepcopy(id2label)
        # also set the label2id
        self._label2id = deepcopy({v: k for k, v in id2label.items()})

    @property
    def label2id(self):
        return deepcopy(self._label2id)

    # create the setter method
    @label2id.setter
    def label2id(self, label2id: Dict[str, int]) -> None:
        # check types
        assert isinstance(label2id, dict)
        assert all([isinstance(k, str) for k in label2id.keys()])
        assert all([isinstance(v, int) for v in label2id.values()])
        # check uniqueness
        assert len(label2id) == len(set(label2id.values()))
        # check countinuous ids starting with 0
        assert set(label2id.values()) == set(range(len(label2id)))
        # set
        self._label2id = deepcopy(label2id)
        # also set the id2label
        self._id2label = deepcopy({v: k for k, v in label2id.items()})

    @property
    def labelset(self) -> List[str]:
        return list(self._id2label.values())

    @property
    def num_classes(self) -> int:
        return len(self._id2label)

    @property
    def num_labels(self) -> int:
        return self.num_classes

    def __getitem__(self, i):
        """Return the i-th example's features."""
        text = self.texts[i]
        label = self.labels[i]

        # tokenize text
        inputs = self.tkz(
            text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: inputs[k][0] for k in inputs.keys()}  # type: ignore

        # binarize labels
        item["labels"] = torch.LongTensor([label])

        return item


class MulticlassTestDataset(MulticlassDataset):
    """Dataset for multiclass classification test sets."""

    def __init__(
        self,
        processor: MulticlassTSVFileProcessor,
        label2id: Dict[str, int],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_seq_length: int = 0,
        use_tqdm: bool = False,
    ):
        """See class."""
        self.use_tqdm = use_tqdm
        self.max_seq_length = max_seq_length
        if max_seq_length <= 0:
            self.max_seq_length = tokenizer.max_len_single_sentence
        logger.info(f"Using max length={self.max_seq_length}")

        self.label2id = label2id

        # read data
        self.texts, labels = processor.get_data()
        self.labels = [self.label2id[label] for label in labels]
        del labels

        # set length and tokenizer
        self.n = len(self.texts)
        self.tkz = tokenizer
