# ULTRA-ECG 🫀
### Transformer based model for optimized ECG signal anomaly detection in imbalanced datasets

## Requirements

This project has two sets of requirements:

### Deployment Requirements (`requirements.txt`)
Contains only the necessary packages to run the Streamlit application:
```bash
pip install -r requirements.txt
```

### Training Requirements (`requirements-training.txt`)
Contains additional packages needed for model training and development:
```bash
pip install -r requirements-training.txt
```

## Development Setup
To set up a complete development environment:
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install both sets of requirements
pip install -r requirements.txt -r requirements-training.txt
```

## Deployment
The application is deployed using only `requirements.txt`. The training requirements are included for development and reproducibility purposes.