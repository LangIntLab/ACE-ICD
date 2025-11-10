# ACE-ICD: Acronym Expansion as Data Augmentation for ICD coding

This repository contains the code for **ACE-ICD: Acronym Expansion as Data Augmentation for ICD coding**.
(The code for training our ICD coding models will be made available soon.)

## Data:

MIMIC-3:  
- Download from [PhysioNet](https://physionet.org/content/mimiciii/1.4/) 

Preprocess: 
- Follow [calm-mimic](https://github.com/jamesmullenbach/caml-mimic) to obtain MIMIC-3-full and MIMIC-3-50 datasets.
- Follow [KEPT](https://github.com/whaleloops/KEPT) to obtain MIMIC-3-rare50 dataset.

Data files after preprocessing:
- Full: mimic3_{train/dev/test}.json
- Top50: mimic3-50_{train/dev/test}.json
- Top50Rare: mimic3-50l_{train/dev/test}.json

Section titles: titles_synonyms.json
- We use section title synonyms from [Semi Structured ICD Coding](https://github.com/LuChang-CS/semi-structured-icd-coding)



## Zero-shot Acronym Expansion

Models: 
- meta-llama/Llama-3.2-1B-Instruct
- meta-llama/Llama-3.2-3B-Instruct
- meta-llama/Llama-3.1-8B-Instruct
- meta-llama/Llama-3.1-70B-Instruct



Generate augment data and filter low-quality outputs:

```
cd acronym_expansion/

python section_expand.py --model_name meta-llama/Llama-3.2-1B-Instruct

python preprocess.py --model_name meta-llama/Llama-3.2-1B-Instruct
```


