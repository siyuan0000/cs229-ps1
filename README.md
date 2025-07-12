# CS229 Problem Set 1 - Machine Learning

This repository contains the implementation for Stanford CS229 Machine Learning Problem Set 1.

## Project Structure

```
ps1-full/
├── src/                    # Source code
│   ├── doubledescent/      # Double descent experiments
│   ├── featuremaps/        # Feature mapping experiments
│   ├── gd_convergence/     # Gradient descent convergence
│   ├── implicitreg/        # Implicit regularization
│   └── lwr/               # Locally weighted regression
├── tex/                    # LaTeX documents
├── environment.yml         # Conda environment
└── README.md              # This file
```

## Setup

1. Install dependencies:
```bash
conda env create -f environment.yml
conda activate cs229-ps1
```

## Experiments

The project includes 5 main experiments:

1. **Double Descent**: Model complexity vs generalization performance
2. **Feature Maps**: Different feature mappings and their effects
3. **GD Convergence**: Gradient descent convergence analysis
4. **Implicit Regularization**: Implicit regularization in optimization
5. **Locally Weighted Regression**: LWR implementation and optimization

## Running Experiments

Each experiment can be run independently:

```bash
# Double descent
cd src/doubledescent
python doubledescent.py

# Feature maps
cd src/featuremaps
python featuremap.py

# Gradient descent convergence
cd src/gd_convergence
python experiment.py

# Implicit regularization
cd src/implicitreg
python linear.py
python qp.py

# Locally weighted regression
cd src/lwr
python lwr.py
python tau.py
```

## Building Reports

Generate PDF reports using LaTeX:

```bash
cd tex
make
```

## License

This project is for educational purposes only, following Stanford CS229 academic integrity policies.
