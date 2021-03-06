# LearnFairMetric_Empirical
The collected responses is stored as a DataFrame, with a shape of (20, 153).

The 153 columns include the following information from the 20 respondents:

`Columns 1-3`: time metadata of the questionnaire responding

`Columns 4-144`: Answers to questions about defendants.

> * There are 7 columns per defendant. 4 of these columns contain timing metadata, you can disregard them. 3 of these columns contain the survey answers - this is what you're interested in. The names of these columns are of the form: "**Defendant_<ID>_Likelihood**", "**Defendant_<ID>_Decision**" and "**Defendant_<ID>_Confidence**", where the substring "<ID>" is replaced with the defendant's id.
> *  The column "**Defendant_<ID>_Likelihood**" contains the answer to the question "How likely do you think it is that this person will commit another crime within 2 years?". This is what we use to learn the distance metric / decision space.
> * The column "**Defendant_<ID>_Decision"** contains the answer to the question "Do you think this person should be granted bail?". This is the respondent's decision, that we use for the sanity check.
> *  The column "**Defendant_<ID>_Confidence**" contains the answer to the question "How confident are you in your answer about granting this person bail?". This is the reported confidence, which we use for testing calibration.

`Columns 145-153(Omitted)`: Demographic data about the respondents.

You can see more description and analysis from our [paper](https://arxiv.org/abs/1910.10255), 
it is accepted by NeurIPS 2019 HCML Workshop, we would love you to cite our work if you find it helpful :)

```
@article{empirical-fair-metric,
  title={An Empirical Study on Learning Fairness Metrics for COMPAS Data with Human Supervision},
  author={Wang, Hanchen and Grgic-Hlaca, Nina and Lahoti, Preethi and Gummadi, Krishna P and Weller, Adrian},
  journal={arXiv preprint arXiv:1910.10255},
  year={2019}
}
```

