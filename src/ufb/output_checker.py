from pathlib import Path
import pandas as pd

project_root = Path(r"C:\Users\filax\OneDrive\Desktop\Code-repository\Kaggle\Competitions\05_UrbanFloodBench - flood modelling\urbanfloodbench")
submission_path = project_root / "outputs" / "submission_full.csv"

submission = pd.read_csv(submission_path, header=None)

print(submission.info())