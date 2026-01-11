# Mobility Project
Interdisciplinary Project in Data Science (with Department of Transport and Mobility) @ TU Wien

**Goal:** build a NER model for transportation references in song lyrics.

[Proposal](https://drive.google.com/file/d/1chhGARIOFdfmS3zIVvx8JhtXnhMULP6Y/view?usp=sharing)

[Report](https://drive.google.com/file/d/10IFCo0myiQTWv3s4Brwlcr1O0B_f5Dhn/view?usp=sharing)

FindVehicle dataset can be find in the [official repo](https://github.com/GuanRunwei/FindVehicle)

## Abstract
This project seeks to analyze lyrics from top-charting songs over the recent years to uncover trends in
references to various modes of transport. It is focused on the creation of transportation reference dataset
and the development of methods for the identification of transportation-related entities. We create the
transport reference dataset (335 examples) by collecting the top-charting song lyrics from open sources
and annotating them manually. We compare analytical baselines, different combinations of prompt
engineering techniques for a state-of-the-art API-accessed generative model and different training data
mixtures for local BERT-based models. Our experiments show that XLM-RoBERTa trained on data
mixture from different sources achieves the F1 score of 0.39 compared to 0.14 F1 score of few-shot
prompted OpenAI API model and 0.22 F1 score of the best analytical baseline model.

## Repository structure
- data
    - definitions.json - *entity definitions from wikipedia*
    - entity_metadata.json - *entity metadata: definitions, guidelines, few-shot examples*
    - labels_manual.jsonl - *manual annotation of song lyrics*
    - synonyms.json - *entity examples found in the data*
- models - *contains the best performing model*
- notebooks
    - baselines.ipynb - *notebook with baselines setup and results*
    - data_annotation.ipynb - *notebook with LLM annotations setups*
    - data_collection.ipynb - *notebook with data collection code*
    - data_splitting.ipynb - *notebook with annotation splits for data mixtures*
- src
    - extraction
        - train\*.py - *script for training corresponding type of models*
        - inference\*.py - *script for inferencing corresponding type of models*
        - mobility.def - *apptainer environment setup for training on slurm cluster*
        - run_slurm.sh - *bash script for running training jobs on slurm cluster*
    - scripts
        - annotate.py - *script for API-accessed generative model inference with different prompt-engineering options*
        - interpret.py - *helper for LaTeX*
        - validate.py - *validation script for extracted entities*
