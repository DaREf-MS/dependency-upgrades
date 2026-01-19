# Replication Package for the Paper  
**On the Time to Complete Dependabot-Suggested Dependency Upgrades**

This repository contains the full replication package for the paper *"On the Time to Complete Dependabot-Suggested Dependency Upgrades"*. It includes all scripts, notebooks, data, and analysis artifacts needed to reproduce the results.

---

## Overview

The study investigates how long it takes for projects to upgrade dependencies suggested by Dependabot, and analyzes factors that influence upgrade delays.

---

## Dataset Statistics

The dataset used in this study contains projects with the following characteristics:

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
├── scripts/               # Scripts used in the study
├── data/                  # Collected and processed datasets
├── prompt/                # TopicGPT prompt templates
├── topicgpt_python/       # TopicGPT framework implementation
├── figures/               # Figures generated for the paper
├── tests/                 # Unit tests
├── select_projects.ipynb  # Jupyter notebook to select relevant projects
├── prep_data.ipynb        # Jupyter notebook to preprocess the data
└── analysis.ipynb         # Main Jupyter notebook to reproduce results
```

---

## Notebooks

| Description   | Notebook |
| ------------- | -------- |
| Project selection | [select_projects.ipynb](select_projects.ipynb) |
| Data preprocessing | [prep_data.ipynb](prep_data.ipynb) |
| Main Analysis | [analysis.ipynb](analysis.ipynb) |

---

## Core Data Collection and Processing

1. `scripts/collect_gh_projects.py`: Collects GitHub projects that use Dependabot.

2. `select_projects.ipynb`: Selects projects that satisfy the study’s inclusion criteria.

3. `scripts/run_pr_classifier.py`: Classifies Dependabot pull requests.

4. `scripts/process_prs.py`: Identifies upgrades that were performed outside of Dependabot pull requests.

5. `prep_data.ipynb`: Constructs and prepares the final dataset of Dependabot upgrades.

6. `analysis.ipynb`: Reproduces all figures and analysis results reported in the paper.

---

## Scripts for Research-Question–Specific Analyses

- `scripts/check_dependabot_exists.py`: Checks whether a project uses Dependabot for security upgrades.

- `scripts/mine_git_history.py`: Collects the history of Dependabot configuration changes via the GitHub API.

- `scripts/process_changed_dep_opt.py`: Extracts the Dependabot configuration options that were modified, based on the output of `scripts/mine_git_history.py`.

- `scripts/retrieve_dep_types.py`: Retrieves the types of dependencies (e.g., `dependencies`, `devDependencies`) declared in `package.json` files.

- `scripts/run_topicgpt.py`: Runs the TopicGPT framework to analyze textual discussions related to upgrades.
