from dataclasses import dataclass
from typing import List
from pathlib import Path
import os
from dotenv import load_dotenv

@dataclass
class MLConfig:
    evaluators: List[str] = (
        "svm", "lr", "tree", "rf", "xgboost", 
        "cat_boost", "mlp", "tab_transformer"
    )
    datasets: List[str] = (
        "adult", "shoppers", "phishing", "magic",
        "faults", "bean", "obesity", "robot",
        "abalone", "news", "insurance", "wine"
    )
    algorithms: List[str] = (
        "pgm", "privsyn", "tvae", "ctgan", 
        "tabddpm", "great", "pategan_eps_1", 
        "tablediffusion_eps_1"
    )

@dataclass
class Config:
    # Load environment variables
    load_dotenv()
    
    # Paths
    root_dir: Path = Path(os.getenv('ROOT_DIR', '/app'))
    tuned_params_path: Path = Path(os.getenv('TUNED_PARAMS_PATH', '/app/exp'))
    
    # Training parameters
    nums_trials: int = int(os.getenv('NUMS_TRIALS', 50))
    n_exps: int = int(os.getenv('N_EXPS', 10))
    min_num_per_class: int = int(os.getenv('MIN_NUM_PER_CLASS', 10))
    
    # Database
    storage: str = os.getenv('STORAGE', 'sqlite:///exp.db')
    
    # ML configurations
    ml: MLConfig = MLConfig()

# Create a singleton config instance
config = Config() 