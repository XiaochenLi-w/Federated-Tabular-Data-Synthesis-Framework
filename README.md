# Tabular Data Synthesis Framework

[![CI](https://github.com/username/Tabular-Data-Synthesis-Framework/actions/workflows/ci.yml/badge.svg)](https://github.com/username/Tabular-Data-Synthesis-Framework/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/username/Tabular-Data-Synthesis-Framework/branch/main/graph/badge.svg)](https://codecov.io/gh/username/Tabular-Data-Synthesis-Framework)

A framework for generating synthetic tabular data with differential privacy.

## Prerequisites
- Python 3.9+ recommended
- (Optional) Docker installed, if you want to run inside a container

## Installation (Local)

1. **Clone** the repository:

   ```bash
   git clone https://github.com/username/Tabular-Data-Synthesis-Framework.git
   ```

2. **Install** dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage
Below are two key scripts used to train and evaluate a synthesizer model.

### Training
```bash
python scripts/train_synthesizer.py -d adult -m privsyn
```
Change adult to the dataset name you want to train on, and privsyn to a different synthesizer name if desired.


### Evaluating
```bash
python scripts/eval_utility.py -d adult -m privsyn
```
This step measures how closely the synthesized data matches the original data’s statistics.

## Docker Usage

You can also use Docker to run the framework in a containerized environment.

1. Build the Docker image (in the project root):
```bash
docker build -t privsyn .
```

2. Run the Docker interactively:
```bash
docker run -it privsyn
```
Inside the container, you can call the same training and evaluation scripts.

3. Run the Docker non-interactively:
```bash
docker run privsyn train -d adult -m privsyn
```
