# Working Emotion Vocabularies

This repository contains code accompanying the paper ["Characterizing and Evaluating Working Emotion Vocabularies in Multilingual Large Language Models"](https://aclanthology.org/2026.acl-long.2188/) presented in ACL 2026.

<div style="margin-left:auto;margin-right:auto">
  <img src="/summary-fig.png?" height="400"/> 
</div>

# Repository and Setup

Install the necessary packages from `requirements.txt` in your environment.

Running `setup.sh` will download the original MASIVE texts and make experiment scripts in `code/` runnable. Python files and bash scripts for replicating the three sets of experiments are included in `code/` with corresponding experiment configurations in `code/configs/full`. Access for some models (e.g., Llama-3.1) may be gated and require requests for access through Huggingface. The main scripts for generating all model predictions and evaluation metrics can be run with the following commands:
__Identification__
```
./code/expression_preds.sh
./code/expression_eval.sh
```

__Expression__
```
./code/expression_preds.sh
./code/expression_eval.sh
```

__Verification__
```
./code/verification_exp.sh
```

# Data

Instructions for accessing data from the original MASIVE paper are included in the [accompanying repository](https://github.com/NickDeas/MASIVE/tree/main). Please reach out to us for access to the human annotations. 

# Citation and Contact

If you found this work or code helpful to your own, please consider citing our paper:
```
@inproceedings{deas-etal-2026-characterizing,
    title = "Characterizing and Evaluating Working Emotion Vocabularies in Multilingual Large Language Models",
    author = "Deas, Nicholas  and
      Mejia, Ivan Ernesto Perez  and
      Yang, Ellie  and
      McKeown, Kathleen",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.2188/",
    doi = "10.18653/v1/2026.acl-long.2188",
    pages = "47272--47294",
    ISBN = "979-8-89176-390-6",
}
```

Questions about the work can be directed to Nicholas Deas, [ndeas@cs.columbia.edu](mailto:ndeas@cs.columbia.edu).