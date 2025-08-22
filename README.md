# VQNHELib

A Python library for VQNHE(Variational Quantum-Neural Hybrid Eigensolver: PRL 128. 120502 (2022)) and U-VQNHE(Unitary-VQNHE: M. Kim et. al., arXiv 2507.11002 (2025))

## User's Guide

### 1. Generate Hamiltonian

```python
from vqnhelib import generate_hamiltonian

# Ising model (4 qubits)
hamiltonian, operators, coeffs = generate_hamiltonian('ising', 4)

# Heisenberg model (4 qubits)
hamiltonian, operators, coeffs = generate_hamiltonian('hei', 4)
```

### 2. Calculate Exact Ground State Energy

```python
from vqnhelib import solve_hamiltonian

exact_energy = solve_hamiltonian(hamiltonian)
print(f"Exact ground state energy: {exact_energy}")
```

### 3. Run VQE

```python
from vqnhelib import IsingAnsatzCircuit, VQETrainer

# Create ansatz circuit (4 qubits, 2 layers)
ansatz = IsingAnsatzCircuit(4, 2)

# Create VQE trainer
trainer = VQETrainer(ansatz, hamiltonian, shots=1000)

# Run training
energy, circuit, params = trainer.train(vqe_trials=10)
print(f"VQE energy: {energy}")
```

### 4. Run VQNHE

```python
from vqnhelib import VQNHETrainer

# Create VQNHE trainer
nn_options = {
    'num_epoch': 100,    # Number of training epochs
    'features': 20,       # Number of hidden features in neural network
    'lr': 0.001          # Learning rate (optional, default: 1e-3)
}

qc_options = {
    'shots': 1000,        # Number of shots for quantum measurements
    'noise_model': None   # Optional noise model
}

trainer = VQNHETrainer(
    ansatz, 
    params, 
    operators, 
    coeffs, 
    nn_options, 
    qc_options
)

# Train neural network
neural_net, loss_history = trainer.train()
```

#### VQNHE Options Explained

**Neural Network Options (`nn_options`):**
- `num_epoch` (required): Number of training epochs for the neural network
- `features` (required): Number of hidden features in the neural network layers
- `lr` (optional): Learning rate for Adam optimizer, defaults to 1e-3

**Quantum Circuit Options (`qc_options`):**
- `shots` (optional): Number of measurement shots, defaults to 2^num_qubits
- `noise_model` (optional): Qiskit noise model
- `exact` (optional): Set to `True` for exact statevector simulation (no sampling noise)

### 5. Run U-VQNHE

```python
from vqnhelib import UnitaryVQNHETrainer

# Create U-VQNHE trainer
qc_options = {
    'shots': 1000,        # Number of shots for quantum measurements
    'noise_model': None,  # Optional noise model
    'exact': False        # Set to True for exact statevector simulation
}

trainer = UnitaryVQNHETrainer(
    ansatz, 
    params, 
    operators, 
    coeffs, 
    nn_options, 
    qc_options
)

# Run training
neural_net, loss_history = trainer.train()
```

#### U-VQNHE Options Explained

**Quantum Circuit Options (`qc_options`):**
- `shots` (optional): Number of measurement shots, defaults to 2^num_qubits
- `noise_model` (optional): Qiskit noise model for realistic quantum simulation
- `exact` (optional): Set to `True` for exact statevector simulation (no sampling noise)

**Note:** When `exact=True`, the U-VQNHE trainer uses exact quantum state simulation instead of sampling.

## Class Descriptions

### Ansatz Circuits
- **`IsingAnsatzCircuit`**: Ansatz circuit for Ising model
- **`HeiAnsatzCircuit`**: Ansatz circuit for Heisenberg model

### Trainers
- **`VQETrainer`**: Basic VQE algorithm
- **`VQNHETrainer`**: Standard VQNHE algorithm
- **`UnitaryVQNHETrainer`**: VQNHE using unitary transformations

## Example

```python
import numpy as np
from vqnhelib import *

# 1. Generate Hamiltonian
hamiltonian, operators, coeffs = generate_hamiltonian('ising', 4)

# 2. Calculate exact energy
exact_energy = solve_hamiltonian(hamiltonian)

# 3. Run VQE
ansatz = IsingAnsatzCircuit(4, 2)
vqe_trainer = VQETrainer(ansatz, hamiltonian, shots=1000)
vqe_energy, _, vqe_params = vqe_trainer.train(vqe_trials=5)

# 4. Run VQNHE
nn_options = {'num_epoch': 50, 'features': 15}
vqnhe_trainer = VQNHETrainer(ansatz, vqe_params, operators, coeffs, nn_options)
neural_net, losses = vqnhe_trainer.train()

# 5. Run U-VQNHE with exact simulation
qc_options_exact = {'exact': True}  # Use exact statevector simulation
uvqnhe_trainer = UnitaryVQNHETrainer(ansatz, vqe_params, operators, coeffs, nn_options, qc_options_exact)
uvqnhe_net, uvqnhe_losses = uvqnhe_trainer.train()

print(f"Exact energy: {exact_energy:.6f}")
print(f"VQE energy: {vqe_energy:.6f}")
print(f"VQNHE final loss: {losses[-1]:.6f}")
print(f"U-VQNHE final loss (exact): {uvqnhe_losses[-1]:.6f}")
```

## Author

**Minwoo Kim**  
Department of Computer Science & Engineering, Seoul National University, South Korea

Email: [myfirstexp@snu.ac.kr](mailto:myfirstexp@snu.ac.kr)
