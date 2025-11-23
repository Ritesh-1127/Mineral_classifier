# Mineral Classifier

Mineral_classifier is a lightweight PyTorch-based project for classifying mineral images. It includes training scripts, model definitions, and example checkpoints so you can reproduce training or run inference quickly.

**Contents**
- `dummy_train.py` – example training script.
- `models/mineralnet.py` – network architecture implementation.
- `checkpoints/` – saved model weights and helper scripts (ignored by `.gitignore`).
- `checkpoints_new/`, `checkpoints_new_b4/` – alternative checkpoint folders.
- `utils/confirm_rules.py` – small utility.

**Quickstart (Windows / PowerShell)**
1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

2. Install minimal dependencies (adjust versions as needed):

```powershell
pip install torch torchvision pillow tqdm
```

3. Train (example):

```powershell
# Edit training options in dummy_train.py as needed
python dummy_train.py
```

4. Predict / run inference (example):

```powershell
# There are example predict scripts in checkpoints/ and checkpoints_new/
python checkpoints/predict.py --weights checkpoints/mineralnet_best.pth --image test_images/<your-image>.jpg
```

If the repository contains `predict_new.py` or other helper scripts, adapt the command to the script's arguments.

**Checkpoints**
Model weights (`*.pth`) are present in the `checkpoints` folders for convenience but are included in `.gitignore` so you won't accidentally re-add them. Replace the path in commands with the checkpoint you want to use.

**Project Notes & Next Steps**
- Consider adding a `requirements.txt` or `pyproject.toml` to pin dependencies.
- Add a `README` section with example outputs or sample images if you want to document expected predictions.
- Add GitHub Actions for CI (e.g., linting, basic unit tests).

**License & Contribution**
This repository currently has no license file. Add a `LICENSE` if you want to allow reuse. Contributions welcome — open an issue or a PR on GitHub.

**Contact**
Repo: https://github.com/Ritesh-1127/Mineral_classifier