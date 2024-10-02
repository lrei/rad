# Classification of Text Data on Rare Diseases
Code used in experiments for the paper: Automatic Classification and Visualization of Text Data on Rare Diseases.

```
@Article{jpm14050545,
AUTHOR = {Rei, Luis and Pita Costa, Joao and Zdolšek Draksler, Tanja},
TITLE = {Automatic Classification and Visualization of Text Data on Rare Diseases},
JOURNAL = {Journal of Personalized Medicine},
VOLUME = {14},
YEAR = {2024},
NUMBER = {5},
ARTICLE-NUMBER = {545},
URL = {https://www.mdpi.com/2075-4426/14/5/545},
PubMedID = {38793127},
ISSN = {2075-4426},
DOI = {10.3390/jpm14050545}
}
```

This code is provided to facilitate replication and for documentation. It includes several hardcoded paths used during the experiments.

## Model
A fine-tuned model is available at on Huggingface: [rad_small](lrei/rad-small).

```python
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="lrei/rad-small")

# Simple high-level usage
pipe(["The patient suffer from a complex genetic disorder.", "The patient suffers from a common genetic disorder."])
```

## Dataset

The dataset used to train this model is available on [zenodo](https://zenodo.org/records/13882003).
It is a subset of abstracts obtained from PubMed and sorted into the 3 classes on the basis of their MeSH terms.

Like the model, the dataset is provided for demonstration and methodology validation purposes. The original PubMed data was randomly under-sampled.

