# Federated Tabular Data Synthesis Framework

This repository contains the implementation of the paper *“HoriFedSyn: Federated Differentially Private Tabular Data Synthesis.”*
 It is developed based on the open-source project [SynMeter](https://github.com/zealscott/SynMeter) by Yuntao Du. The repository supports **horizontal federated** tabular data synthesis and includes implementations of the centralized algorithm [PrivSyn](https://www.usenix.org/system/files/sec21-zhang-zhikun.pdf) and two federated versions, FedPrivSyn and AdvFedPrivSyn.

## Prerequisites
- Python 3.9+ recommended
- (Optional) Docker installed, if you want to run inside a container

## Installation (Local)

1. **Clone** the repository:

   ```bash
   git clone https://github.com/username/Federated-Tabular-Data-Synthesis-Framework.git
   ```

2. **Install** dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage
Below are key scripts used to train and evaluate a synthesizer model.

### Training

- Run PrivSyn synthesis algorithm.

```bash
python scripts/train_synthesizer.py -d adult -m privsyn
```
- Run FedPrivSyn synthesis algorithm.

```bash
python scripts/train_fed_synthesizer.py -d adult -m fed_privsyn
```

- Run AdvFedPrivSyn synthesis algorithm.

```bash
python scripts/train_fed_synthesizer_adaptive.py -d adult -m fed_privsyn_adv
```

Change adult to the dataset name you want to train on.

### Evaluating

- Evaluate utility of PrivSyn

```bash
python scripts/eval_utility.py -d adult -m privsyn
```
- Evaluate utility of FedPrivSyn

```bash
python scripts/eval_fed_utility.py -d adult -m fed_privsyn
```

- Evaluate utility of AdvFedPrivSyn

```bash
python scripts/eval_fed_adaptive_utility.py -d adult -m fed_privsyn_adv
```

- Evaluate Fidelity of PrivSyn

```bash
python scripts/eval_fidelity.py -d adult -m privsyn
```

- Evaluate Fidelity of FedPrivSyn

```bash
python scripts/eval_fed_fidelity.py -d adult -m fed_privsyn
```

- Evaluate Fidelity of AdvFedPrivSyn

```bash
python scripts/eval_fed_adaptive_fidelity.py -d adult -m fed_privsyn_adv
```

- Evaluate ML of PrivSyn

```bash
python scripts/eval_ml.py -d adult -m privsyn
```

- Evaluate ML of FedPrivSyn

```bash
python scripts/eval_fed_ml.py -d adult -m fed_privsyn
```

- Evaluate ML of AdvFedPrivSyn

```bash
python scripts/eval_fed_adaptive_ml.py -d adult -m fed_privsyn_adv
```

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

## Docker Compose

You can also run everything inside containers. This repository includes a docker-compose.yml file for convenience.

### Starting the Development Container
```bash
docker-compose up
```
This builds and starts the container(s) defined in docker-compose.yml. Check the logs for any errors.

### Entering the Container
```bash
docker-compose exec dev zsh
```
You can replace "dev" with the name of your service/container as specified in docker-compose.yml. 

If you want to stop or rebuild, use:

```bash
docker-compose down
docker-compose up --build
```

## Debugging with VS Code

If you use Visual Studio Code and want to debug inside the Docker container:

1. Make sure you have the Docker extension and the Python extension installed.
2. Start the container:

   ```bash
   docker-compose up
   ```

3. Attach to the container:

   ```bash
   docker-compose exec dev zsh
   ```

4. Run your Python script with debugpy:

   ```bash
   python -m debugpy --listen 0.0.0.0:5678 --wait-for-client scripts/eval_utility.py -d adult -m privsyn
   ```

5. In VS Code, set breakpoints, then press F5 or click "Run and Debug," choosing "Python: Remote Attach."

6. The debugger will attach to the script running on port 5678.

### Debug Controls
- F5: Continue
- F10: Step Over
- F11: Step Into
- Shift+F11: Step Out
- Shift+F5: Stop Debugging
- Ctrl+Shift+F5: Restart

### Troubleshooting
- If the debugger is not connecting, ensure port 5678 is free.
- If you need a fresh start, run:

  ```bash
  docker-compose down
  docker-compose up --build
  ```
