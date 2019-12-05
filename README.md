# LearnFairMetric_Empirical
The collected responding is a dataframe is 20(human respondents) * 153 (info about the questions) .

Columns include the following informations:

*** Columns 1-3**: time metadata of the questionnaire responding

*** Columns 4-144**: Answers to questions about defendants.

> * There are 7 columns per defendant. 4 of these columns contain timing metadata, you can disregard them. 3 of these columns contain the survey answers - this is what you're interested in. The names of these columns are of the form: "**Defendant_<ID>_Likelihood**", "**Defendant_<ID>_Decision**" and "**Defendant_<ID>_Confidence**", where the substring "<ID>" is replaced with the defendant's id.
> *  The column "**Defendant_<ID>_Likelihood**" contains the answer to the question "How likely do you think it is that this person will commit another crime within 2 years?". This is what we use to learn the distance metric / decision space.
> * The column "**Defendant_<ID>_Decision"** contains the answer to the question "Do you think this person should be granted bail?". This is the respondent's decision, that we use for the sanity check.
> *  The column "**Defendant_<ID>_Confidence**" contains the answer to the question "How confident are you in your answer about granting this person bail?". This is the reported confidence, which we use for testing calibration.



*** Columns 145-153(Omitted)**: Demographic data about the respondents.



If you like it, please see our description [here](https://arxiv.org/abs/1910.10255), it is now accepted by NeurIPS 2019 HCML Workshop, you cite it with the bibtex here :)

```
@article{journals/corr/abs-1910-10255,
  author    = {Hanchen Wang and
               Nina Grgic{-}Hlaca and
               Preethi Lahoti and
               Krishna P. Gummadi and
               Adrian Weller},
  title     = {An Empirical Study on Learning Fairness Metrics for {COMPAS} Data
               with Human Supervision},
  journal   = {CoRR},
  volume    = {abs/1910.10255},
  year      = {2019},
  url       = {http://arxiv.org/abs/1910.10255},
  archivePrefix = {arXiv},
  eprint    = {1910.10255},
}
```

