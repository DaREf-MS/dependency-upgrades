# Replication Package for the Paper: *Understanding the Time to Upgrade Dependencies Suggested by Dependabot*

* **Notebooks:**

| Description       | Notebook Link                                             |
| ----------------- | --------------------------------------------------------- |
| Analysis Notebook | [main.ipynb](analysis.ipynb)                         |

This repository contains the replication package for the paper *"Understanding the Time to Upgrade Dependencies Suggested by Dependabot"*.

---

## Overview

The dataset features the following information:

| Metric | Min. | Median | Mean | Max. |
| :--- | ---: | ---: | ---: | ---: |
| **Commits** | 41 | 901 | 2,243.35 | 72,123 |
| **Age (days)** | 336 | 2,613 | 2,737.24 | 5,910 |
| **Dep. PRs** | 5 | 47 | 132.81 | 999 |
| **PRs** | 14 | 373 | 806.99 | 17,558 |
| **Stars** | 10 | 108 | 622.20 | 10,428 |
| **Contributors** | 5 | 31 | 104.12 | 7,103 |

---

## Project Structure
```text
.
├── scripts/            # Scripts used in the study
├── data/               # Data used in the study
├── prompt/             # TopicGPT prompt templates
├── topicgpt_python/    # TopicGPT code
├── figures/            # Figures used in study
├── tests/              # Unit tests
└── analysis.ipynb      # Jupyter Notebook to reproduce the results 