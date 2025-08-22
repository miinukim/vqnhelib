'''
VQNHE Simulation

Author: Minwoo Kim (Dept. of Computer Science & Engineering, Seoul National University)
'''

#Import Qiskit
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, ParameterVector
from qiskit.circuit.library import RZZGate, RXXGate, RYYGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer.primitives import Estimator, Sampler
from qiskit.visualization import circuit_drawer

#Import Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

#Import other libraries
import random
import sys
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm
import numpy as np
import time
from typing import Union, Optional

import warnings
warnings.filterwarnings("ignore")


## Define the problem Hamiltonian ##

def generate_hamiltonian(type : str, num_qubits):
    """
    Generate a SparsePauliOp instance of Hamiltonian.

    Parameters:
    type (str), num_qubits (int)

    Returns:
    hamiltonian (SparsePauliOp), operator_list (list), coeff_list (list)
    """
    
    if type not in ['ising', 'hei']:
        raise ValueError("Not a valid type of Hamiltonian.")
    
    operator_list = []
    coeff_list = []
    i_str = 'I' * num_qubits
    
    if type == 'ising':
        interaction = 1
        bias = -1
        for idx in range(num_qubits-1):
            z_str = i_str[:idx] + 'ZZ' + i_str[idx+2:]
            operator_list.append(z_str)
        for idx in range(num_qubits):
            x_str = i_str[:idx] + 'X' + i_str[idx+1:]
            operator_list.append(x_str)
        
        coeff_list = [interaction for _ in range(num_qubits - 1)] + [bias for _ in range(num_qubits)]
  
    elif type == 'hei':
        pauli_let_list = ['X', 'Y', 'Z']
        for pauli_let in pauli_let_list:
            for idx in range(num_qubits):
                tmp = list(i_str)
                tmp[idx] = pauli_let
                tmp[(idx + 1) % num_qubits] = pauli_let
                operator_list.append(''.join(tmp))
        
        coeff_list = [1 for _ in range(len(operator_list))]

    hamiltonian_list= list(zip(operator_list, coeff_list))
    hamiltonian = SparsePauliOp.from_list(hamiltonian_list)

    return hamiltonian, operator_list, coeff_list


def solve_hamiltonian(hamiltonian):
    """
    Solve for the ground state energy of the given Hamiltonian classically via diagonalization

    Parameters:
    hamiltonian (SparsePauliOp)

    Returns:
    gs_energy
    """
    t_begin = time.time()
    eigvals = np.linalg.eigvals(hamiltonian)
    gs_energy = np.real(np.min(eigvals))
    
    t_end = time.time()
    t_cl = t_end - t_begin
    print(f"Classical computation: {t_cl:.2f} seconds")
    print(f"Exact ground state energy: {gs_energy:.6f}")
    return gs_energy


# Ansatz circuit for Ising model, subclass of QuantumCircuit
class IsingAnsatzCircuit(QuantumCircuit):
    def __init__(self, num_qubits, ansatz_len):
        super(IsingAnsatzCircuit, self).__init__(num_qubits)
        self.ansatz_len = ansatz_len
        self.n_q = num_qubits
        self.num_params = ansatz_len * (2 * self.n_q - 1)
        self.params = ParameterVector("theta", length=self.num_params)
        self._build_circuit()

    def _build_circuit(self):
        # Apply initial Hadamard gates
        for i in range(self.n_q):
            self.h(i)

        for d in range(self.ansatz_len):
            params_rz = self.params[(self.n_q * 2 - 1)*d:self.n_q-1+(self.n_q * 2 - 1)*d]
            params_rx = self.params[(self.n_q * 2 - 1)*d+self.n_q-1:(self.n_q * 2 - 1)*(d+1)]
            
            # Apply RZ and CNOT gates
            cnot_list = []
            for q in range(self.n_q-3):
                cnot_list.append(q)
                cnot_list.append(q + 2)

            for i in cnot_list:
                self.cx(i, (i+1) % self.n_q)
                self.rz(params_rz[i], (i+1) % self.n_q)
                self.cx(i, (i+1) % self.n_q)

            # Apply RX gates
            for i in range(self.n_q):
                self.rx(params_rx[i], i)


    def get_parameters(self):
        return self.params

    # Return a circuit with measurement circuit attached at the end. Uses the same parameter
    def generate_measurement_circuit(self, pauli_str : str, measure = True, real = True):
        qc = self.copy()

        x_y = False
        star_qubit = -1
        for idx in range(self.n_q):
            if pauli_str[idx] == 'X' or pauli_str[idx] == 'Y':
                x_y = True
                star_qubit = idx
                break
        if x_y == True:
            for k in range(star_qubit+1,self.n_q):
                if pauli_str[k] == 'X':
                    qc.cx(star_qubit, k)
                elif pauli_str[k] == 'Y':
                    qc.cy(star_qubit, k)

            if real:
                if pauli_str[star_qubit] == 'X':
                    qc.h(star_qubit)
                elif pauli_str[star_qubit] == 'Y':
                    qc.rx(np.pi/2, star_qubit)
            else:
                if pauli_str[star_qubit] == 'Y':
                    qc.h(star_qubit)
                elif pauli_str[star_qubit] == 'X':
                    qc.rx(np.pi/2, star_qubit)
        
        if measure:
            qc.barrier()
            qc.measure_all()

        return qc, star_qubit


class HeiAnsatzCircuit(QuantumCircuit):
    def __init__(self, num_qubits, ansatz_len):
        super(HeiAnsatzCircuit, self).__init__(num_qubits)
        assert num_qubits % 2 == 0
        self.ansatz_len = ansatz_len
        self.n_q = num_qubits
        self.num_params = ansatz_len * num_qubits
        self.params = ParameterVector("theta", length=self.num_params)
        self._build_circuit()

    def _build_circuit(self):
        theta = self.params

        for i in range(0, self.n_q, 2):
            self.h(i)
            self.cx(i, i+1)
            self.x(i+1)
            self.z(i+1)

        for layer in range(self.ansatz_len):
            for i in range(self.n_q):
                param_index = layer*self.n_q + i
                targets = [i, (i+1) % self.n_q]
                self.append(RXXGate(theta[param_index]), targets)
                self.append(RYYGate(theta[param_index]), targets)
                self.append(RZZGate(theta[param_index]), targets)


    def get_parameters(self):
        return self.params


    def generate_measurement_circuit(self, pauli_str : str, measure = True, real = True):
        qc = self.copy()
        
        x_y = False
        star_qubit = -1
        for idx in range(self.n_q):
            if pauli_str[idx] == 'X' or pauli_str[idx] == 'Y':
                x_y = True
                star_qubit = idx
                break
        if x_y == True:
            for k in range(star_qubit+1,self.n_q):
                if pauli_str[k] == 'X':
                    qc.cx(star_qubit, k)
                elif pauli_str[k] == 'Y':
                    qc.cy(star_qubit, k)
            if real:
                if pauli_str[star_qubit] == 'X':
                    qc.h(star_qubit)
                elif pauli_str[star_qubit] == 'Y':
                    qc.rx(np.pi/2, star_qubit)
            else:
                if pauli_str[star_qubit] == 'Y':
                    qc.h(star_qubit)
                elif pauli_str[star_qubit] == 'X':
                    qc.rx(np.pi/2, star_qubit)
        
        if measure:
            qc.barrier()
            qc.measure_all()

        return qc, star_qubit



class VQETrainer:
    def __init__(self, ansatz, hamiltonian, shots=None, noise_model=None, print_mode = False):
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.noise_model = noise_model
        self.shots = shots
        
        self.estimator = self._create_estimator()
        
        self.callback_dict = {
            "prev_vector": None,
            "iters": 0,
            "cost_history": [],
        }
        self.print_mode = print_mode
        
    def _create_estimator(self):
        if self.noise_model == None:
            return Estimator(run_options={"shots": self.shots})
        elif self.shots == None:
            return Estimator(backend_options={"noise_model": self.noise_model})
        else:
            return Estimator(
                run_options={"shots": self.shots},
                backend_options={"noise_model": self.noise_model}
            )

    def cost_func_vqe(self, params):
        cost = self.estimator.run(self.ansatz, self.hamiltonian, parameter_values=params).result()
        return cost.values[0]

    def build_callback(self):
        def callback(current_vector):
            self.callback_dict["iters"] += 1
            self.callback_dict["prev_vector"] = current_vector
            current_cost = self.cost_func_vqe(current_vector)
            self.callback_dict["cost_history"].append(current_cost)
        return callback

    def train(self, vqe_trials=15, known_seed=None):
        """
        Train the parameters of VQE

        Parameters:
        vqe_trials (int) : number of trials to train VQE
        known_seed (int) : numpy seed if the best is known

        Returns:
        lowest_energy, params
        """
        if self.print_mode: print(f"Optimizing VQE ({vqe_trials} trials)")
        
        lowest = float('inf')
        best_seed = 0
        best_result = None

        if known_seed is None:
            for i in tqdm(range(vqe_trials)):
                result = self._run_optimization(i)
                if result.fun < lowest:
                    lowest = result.fun
                    best_seed = i
                    best_result = result
        else:
            best_seed = known_seed
            best_result = self._run_optimization(known_seed)
            lowest = best_result.fun

        if self.print_mode: 
            print(f"\nBest seed: {best_seed}")
            print(f"Lowest energy: {lowest:.6f}")
        
        return lowest, self.ansatz, best_result['x']

    def _run_optimization(self, seed):
        np.random.seed(seed)
        param_len = len(self.ansatz.get_parameters())
        x0 = 2 * np.pi * np.random.random(param_len)
        callback = self.build_callback()
        
        options = {"maxiter": 3000}
        
        result = minimize(
            self.cost_func_vqe,
            x0,
            method="cobyla",
            callback=callback,
            options=options
        )
        
        return result


class NeuralNet(nn.Module):
    def __init__(self, num_qubits, features):
        super(NeuralNet, self).__init__()
        self.FC1 = nn.Linear(num_qubits, 10)
        self.FC2 = nn.Linear(10, 20)
        self.phi = nn.Parameter(torch.empty(20))
        self.activation = nn.Tanh()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        init.normal_(self.phi)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        s1 = F.relu(self.FC1(x))
        s2 = torch.sigmoid(self.FC2(s1))
        act = self.activation(s2)
        exp = torch.matmul(act, self.phi)
        f = torch.exp(exp)
        return f.squeeze()


class ComplexNet(nn.Module):
    def __init__(self, num_qubits, features):
        super(ComplexNet, self).__init__()
        self.FC1 = nn.Linear(num_qubits, features)
        self.FC2 = nn.Linear(features, features)
        self.phi = nn.Parameter(torch.empty(features))
        self.activation = nn.Tanh()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        init.normal_(self.phi)
        self.print_mode = False

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        s1 = F.relu(self.FC1(x))
        s2 = torch.sigmoid(self.FC2(s1))
        act = self.activation(s2)              

        exp = torch.matmul(act, self.phi)     

        if self.print_mode:
            return exp.squeeze()               

        i_theta = torch.complex(torch.zeros_like(exp), exp)
        out = torch.exp(i_theta)                 
        return out.squeeze()


## VQNHE Trainer
class VQNHETrainer:
    def __init__(self, ansatz_circuit: Union[IsingAnsatzCircuit, HeiAnsatzCircuit],
            params : ParameterVector, operator_list, coeff_list, nn_options: dict, 
            qc_options: Optional[dict] = {}, print_mode = False, cutoff = -100):
        '''
        qc_options (optional, dict): ['shots', 'noise_model']
        nn_options (dict): ['num_epoch', 'features', 'lr'(optional)]
        '''
        self.ansatz_circuit = ansatz_circuit
        self.num_qubits = ansatz_circuit.num_qubits
        self.shots = qc_options.get('shots', 2**self.num_qubits)
        self.noise_model = qc_options.get('noise_model', None)
        self.sampler = Sampler(backend_options = {'shots': self.shots, 'noise_model': self.noise_model})
        self.operator_list = operator_list
        self.coeff_list = coeff_list
        self.params = params
        self.print_mode = print_mode
        self.cutoff = cutoff

        if 'num_epoch' not in nn_options.keys() or 'features' not in nn_options.keys(): 
            raise ValueError("nn_options must include 'num_epoch' and 'features'") 
        self.num_epoch = nn_options['num_epoch']
        self.features = nn_options['features']
        self.lr = nn_options.get('lr', 1e-3)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.print_mode: print(f"Torch running on " + self.device)

        self.nnet = NeuralNet(num_qubits = self.num_qubits, features = self.features).to(self.device)
        betas = (0.9, 0.999)
        self.optimizer = torch.optim.Adam(self.nnet.parameters(), lr = self.lr, betas = betas)


    def _pauli_transformation(self, bstr: torch.Tensor, pauli_str: str) -> torch.Tensor:
        # Ensure bstr is a tensor of integers
        conj_list = torch.ones_like(bstr, dtype=torch.float32)

        for i in range(len(pauli_str)):
            if pauli_str[i] in ["X", "Y"]:
                conj_list[i] = -1 if bstr[i] == 1 else 1
            else:
                conj_list[i] = bstr[i]

        return conj_list


    def _run_sampler(self, qc):
        result = self.sampler.run(qc).result().quasi_dists[0].binary_probabilities()
        return result

    def train(self, show_process = True, return_keys = False):
        self.nnet.train()
        loss_list = []
        total_loss = 0.0
        self.optimizer.zero_grad()

        # --- Build (and execute) the ansatz circuit once ---
        qc_ans, _ = self.ansatz_circuit.generate_measurement_circuit("I" * self.num_qubits)
        param_dict = {param: value for param, value in zip(qc_ans.parameters, self.params)}
        qc_ans.assign_parameters(param_dict, inplace=True)
        ansatz_res = self._run_sampler(qc_ans)  # dict: bitstring -> prob

        # --- Build (and execute) all measurement circuits once ---
        meas_res = {}
        star_idxs = {}
        for pstr in self.operator_list:
            qc_meas, star_idx = self.ansatz_circuit.generate_measurement_circuit(pstr)
            qc_meas.assign_parameters(param_dict, inplace=True)
            meas_res[pstr] = self._run_sampler(qc_meas)  # dict: bitstring -> prob
            star_idxs[pstr] = star_idx

        # Optionally return keys (unchanged from your pattern)
        if return_keys:
            ansatz_keys = set(ansatz_res.keys())
            meas_keys = set()
            for pstr in self.operator_list:
                meas_keys = meas_keys.union(set(meas_res[pstr].keys()))
            return ansatz_keys, meas_keys

        iterator = tqdm(range(self.num_epoch), desc="Processing") if show_process else range(self.num_epoch)

        for epoch in iterator:
            # ------------------------
            # 1) Batched denominator
            # ------------------------
            # Prepare batch of ansatz bitstrings
            a_tensors = []
            a_probs = []
            for a_str, a_prob in ansatz_res.items():
                a_list = torch.tensor([int(x) for x in a_str], dtype=torch.float32, device=self.device)
                a_tensor = 2 * a_list - 1  # map {0,1} -> {-1,1}
                a_tensor = a_tensor.flip(dims=[0])  # match your convention
                a_tensors.append(a_tensor)
                a_probs.append(a_prob)

            a_batch = torch.stack(a_tensors, dim=0) if len(a_tensors) > 0 else torch.empty((0, self.num_qubits), device=self.device)
            a_probs = torch.tensor(a_probs, dtype=torch.float32, device=self.device) if len(a_tensors) > 0 else torch.empty((0,), device=self.device)

            fs_denom_batch = self.nnet(a_batch)  # shape [B]
            denominator = torch.sum((fs_denom_batch.abs() ** 2) * a_probs)

            # Guard against numerical underflow
            eps = 1e-12
            denominator = denominator + eps

            # ------------------------
            # 2) Batched numerator(s)
            # ------------------------
            ham = 0.0
            for idx, pstr in enumerate(self.operator_list):
                string_prob = meas_res[pstr]  # dict: m_str -> prob
                star_idx = star_idxs[pstr]

                m_tensors = []
                m_transformed_tensors = []
                m_probs = []
                signs = []

                # Precompute Z positions once
                z_positions = [i for i, ch in enumerate(pstr) if ch == 'Z']

                for m_str, m_prob in string_prob.items():
                    # Base m_tensor
                    m_list = torch.tensor([int(x) for x in m_str], dtype=torch.float32, device=self.device)
                    m_tensor = 2 * m_list - 1
                    m_tensor = m_tensor.flip(dims=[0])

                    sign = 1.0

                    # Handle star index branch
                    if star_idx >= 0:
                        # If the star qubit reads +1, flip sign and flip the bit
                        if m_tensor[star_idx] == 1:
                            sign *= -1.0
                            # In-place change for this local tensor
                            m_tensor = m_tensor.clone()
                            m_tensor[star_idx] = -1.0

                    # Z-parity sign contribution
                    if z_positions:
                        # Count how many Z-positions have +1
                        # Equivalent to multiplying sign by -1 for each +1 at a Z position
                        # We can compute as: (-1) ** (count of +1 at Z-positions)
                        pos_plus_ones = sum(1 for j in z_positions if m_tensor[j] == 1)
                        if (pos_plus_ones % 2) == 1:
                            sign *= -1.0

                    # Transformed tensor (depends on pstr)
                    # Note: _pauli_transformation expects a single 1D tensor; we apply per-sample then batch.
                    if star_idx >= 0:
                        m_transformed = self._pauli_transformation(m_tensor, pstr)
                    else:
                        # When star_idx == -1, original code does not call transformation in the "else" branch.
                        # Keep that behavior by letting transformed == original.
                        m_transformed = m_tensor

                    m_tensors.append(m_tensor)
                    m_transformed_tensors.append(m_transformed)
                    m_probs.append(m_prob)
                    signs.append(sign)

                if len(m_tensors) == 0:
                    continue

                m_batch = torch.stack(m_tensors, dim=0)
                mt_batch = torch.stack(m_transformed_tensors, dim=0)
                m_probs_t = torch.tensor(m_probs, dtype=torch.float32, device=self.device)
                signs_t = torch.tensor(signs, dtype=torch.float32, device=self.device)

                # Two batched forward passes
                fs = self.nnet(m_batch)
                if star_idx >= 0:
                    fs_t = self.nnet(mt_batch)
                    numerator = torch.sum(fs * fs_t * m_probs_t * signs_t)
                else:
                    # Original else-branch uses |fs|^2
                    numerator = torch.sum((fs.abs() ** 2) * m_probs_t * signs_t)

                mean_pauli = numerator / denominator * self.coeff_list[idx]
                ham = ham + mean_pauli

            running_loss = ham
            running_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_list.append(float(running_loss.detach().cpu().item()))

        '''
        if epoch > 20:
            if np.abs(loss_list[-1] - loss_list[-2]) < 1e-4 * np.abs(loss_list[-1]): break
            if loss_list[-1] < self.cutoff: break
        '''
        
        if self.print_mode: print('Finished Training')
        
        if return_keys: return self.nnet, loss_list, ansatz_keys, meas_keys
        else: return self.nnet, loss_list


## VQNHE Trainer with unitary neural network
class UnitaryVQNHETrainer:
    def __init__(self, ansatz_circuit: Union[IsingAnsatzCircuit, HeiAnsatzCircuit],
            params : ParameterVector, operator_list, coeff_list, nn_options: dict, 
            qc_options: Optional[dict] = {}, print_mode = False):
        '''
        qc_options (optional, dict): ['shots', 'noise_model']
        nn_options (dict): ['num_epoch', 'features', 'lr'(optional)]
        '''
        self.ansatz_circuit = ansatz_circuit
        self.num_qubits = ansatz_circuit.num_qubits
        self.shots = qc_options.get('shots', 2**self.num_qubits)
        self.noise_model = qc_options.get('noise_model', None)
        self.sampler = Sampler(backend_options = {'shots': self.shots, 'noise_model': self.noise_model})
        self.operator_list = operator_list
        self.coeff_list = coeff_list
        self.params = params
        self.exact = qc_options.get('exact', False)
        self.print_mode = print_mode

        if 'num_epoch' not in nn_options.keys() or 'features' not in nn_options.keys(): 
            raise ValueError("nn_options must include 'num_epoch' and 'features'") 
        self.num_epoch = nn_options['num_epoch']
        self.features = nn_options['features']
        self.lr = nn_options.get('lr', 1e-3)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.print_mode: print(f"Torch running on " + self.device)

        self.nnet = ComplexNet(num_qubits = self.num_qubits, features = self.features).to(self.device)
        betas = (0.9, 0.999)
        self.optimizer = torch.optim.Adam(self.nnet.parameters(), lr = self.lr, betas = betas)


    def _pauli_transformation(self, bstr: torch.Tensor, pauli_str: str) -> torch.Tensor:
        # Ensure bstr is a tensor of integers
        conj_list = torch.ones_like(bstr, dtype=torch.float32)

        for i in range(len(pauli_str)):
            if pauli_str[i] in ["X", "Y"]:
                conj_list[i] = -1 if bstr[i] == 1 else 1
            else:
                conj_list[i] = bstr[i]

        return conj_list

    def _run_qc(self, qc):
        statevector = Statevector(qc)
        probabilities = statevector.probabilities()
        result = {format(i, f'0{self.num_qubits}b'): prob for i, prob in enumerate(probabilities)}
        return result

    def _run_sampler(self, qc):
        result = self.sampler.run(qc).result().quasi_dists[0].binary_probabilities()
        return result

    def train(self, show_process = True):
        self.nnet.train()
        loss_list = []
        total_loss = 0.0
        self.optimizer.zero_grad()

        # Bind parameters once (same as your original)
        qc_ans, _ = self.ansatz_circuit.generate_measurement_circuit("I" * self.num_qubits)
        param_dict = {param: value for param, value in zip(qc_ans.parameters, self.params)}

        # Run measurement circuits (same logic as your original)
        real_res = {}
        imag_res = {}
        star_idxs = {}

        for pstr in self.operator_list:
            qc_meas, star_idx = self.ansatz_circuit.generate_measurement_circuit(
                pstr, measure=not self.exact
            )
            qc_meas_imag, _ = self.ansatz_circuit.generate_measurement_circuit(
                pstr, measure=not self.exact, real=False
            )
            qc_meas.assign_parameters(param_dict, inplace=True)
            qc_meas_imag.assign_parameters(param_dict, inplace=True)

            if self.exact:
                real_res[pstr] = self._run_qc(qc_meas)
                imag_res[pstr] = self._run_qc(qc_meas_imag)
            else:
                real_res[pstr] = self._run_sampler(qc_meas)
                imag_res[pstr] = self._run_sampler(qc_meas_imag)

            star_idxs[pstr] = star_idx

        iterator = tqdm(range(self.num_epoch), desc="Processing") if show_process else range(self.num_epoch)

        for epoch in iterator:
            ham = 0.0

            # No normalization/denominator in this estimator (same as your code)

            for idx, pstr in enumerate(self.operator_list):
                star_idx = star_idxs[pstr]
                z_positions = [i for i, ch in enumerate(pstr) if ch == 'Z']

                # -------------------------
                # Helper to batch a channel
                # -------------------------
                def batch_channel(prob_dict, take_imag: bool):
                    """
                    Build batches for a given prob_dict (real or imag) and compute:
                    star_idx >= 0: sum( (conj(fs)*fs_t).{real/imag} * prob * sign )
                    star_idx <  0: sum( (conj(fs)*fs  ).{real/imag} * prob * sign )
                    """
                    if not prob_dict:
                        return torch.tensor(0.0, dtype=torch.float32, device=self.device)

                    m_tensors, mt_tensors, probs, signs = [], [], [], []

                    for m_str, m_prob in prob_dict.items():
                        # Base tensor in {-1, +1}
                        m_list = torch.tensor([int(x) for x in m_str], dtype=torch.float32, device=self.device)
                        m_tensor = 2 * m_list - 1
                        m_tensor = m_tensor.flip(dims=[0])

                        sign = 1.0

                        # star-index branch
                        if star_idx >= 0:
                            if m_tensor[star_idx] == 1:
                                sign *= -1.0
                                m_tensor = m_tensor.clone()
                                m_tensor[star_idx] = -1.0  # flip to -1

                        # Z-parity contribution to sign
                        if z_positions:
                            pos_plus_ones = 0
                            for j in z_positions:
                                if m_tensor[j] == 1:
                                    pos_plus_ones += 1
                            if (pos_plus_ones % 2) == 1:
                                sign *= -1.0

                        # Transformed tensor (only used when star_idx >= 0 per your original code)
                        if star_idx >= 0:
                            m_transformed = self._pauli_transformation(m_tensor, pstr)
                        else:
                            m_transformed = m_tensor  # unused, but keep shape consistent

                        m_tensors.append(m_tensor)
                        mt_tensors.append(m_transformed)
                        probs.append(m_prob)
                        signs.append(sign)

                    m_batch = torch.stack(m_tensors, dim=0)
                    probs_t = torch.tensor(probs, dtype=torch.float32, device=self.device)
                    signs_t = torch.tensor(signs, dtype=torch.float32, device=self.device)

                    # Forward passes
                    fs = self.nnet(m_batch)

                    if star_idx >= 0:
                        mt_batch = torch.stack(mt_tensors, dim=0)
                        fs_t = self.nnet(mt_batch)
                        prod = fs.conj() * fs_t
                    else:
                        prod = fs.conj() * fs  # |fs|^2; imag part should be ~0

                    channel_val = prod.imag if take_imag else prod.real
                    # Weighted sum
                    return torch.sum(channel_val * probs_t * signs_t)

                # Real channel contribution
                real_contrib = batch_channel(real_res[pstr], take_imag=False)
                # Imag channel contribution
                imag_contrib = batch_channel(imag_res[pstr], take_imag=True)

                numerator = real_contrib + imag_contrib
                exp_pauli = numerator * self.coeff_list[idx]
                ham = ham + exp_pauli

            running_loss = ham
            running_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += float(ham.detach().cpu().item())
            loss_list.append(float(ham.detach().cpu().item()))
        
        if self.print_mode: print('Finished Training')
        
        return self.nnet, loss_list


## VQNHE Trainer
class ModifiedVQNHETrainer:
    def __init__(self, ansatz_circuit: Union[IsingAnsatzCircuit, HeiAnsatzCircuit],
            params : ParameterVector, operator_list, coeff_list, nn_options: dict, 
            qc_options: Optional[dict] = {}, print_mode = False):
        '''
        qc_options (optional, dict): ['shots', 'noise_model']
        nn_options (dict): ['num_epoch', 'features', 'lr'(optional)]
        '''
        self.ansatz_circuit = ansatz_circuit
        self.num_qubits = ansatz_circuit.num_qubits
        self.shots = qc_options.get('shots', 2**self.num_qubits)
        self.noise_model = qc_options.get('noise_model', None)
        self.sampler = Sampler(backend_options = {'shots': self.shots, 'noise_model': self.noise_model})
        self.operator_list = operator_list
        self.coeff_list = coeff_list
        self.params = params
        self.print_mode = print_mode

        if 'num_epoch' not in nn_options.keys() or 'features' not in nn_options.keys(): 
            raise ValueError("nn_options must include 'num_epoch' and 'features'") 
        self.num_epoch = nn_options['num_epoch']
        self.features = nn_options['features']
        self.lr = nn_options.get('lr', 1e-3)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.print_mode: print(f"Torch running on " + self.device)

        self.nnet = NeuralNet(num_qubits = self.num_qubits, features = self.features).to(self.device)
        betas = (0.9, 0.999)
        self.optimizer = torch.optim.Adam(self.nnet.parameters(), lr = self.lr, betas = betas)


    def _pauli_transformation(self, bstr: Union[torch.Tensor, list], pauli_str: str) -> torch.Tensor:
        # Ensure bstr is a tensor of integers
        if type(bstr) == list: bstr = torch.tensor(bstr)
        conj_list = torch.ones_like(bstr, dtype=torch.float32)

        for i in range(len(pauli_str)):
            if pauli_str[i] in ["X", "Y"]:
                conj_list[i] = -1 if bstr[i] == 1 else 1
            else:
                conj_list[i] = bstr[i]

        return conj_list


    def _run_sampler(self, qc):
        result = self.sampler.run(qc, parameter_values=self.params).result().quasi_dists[0].binary_probabilities()
        return result

    def train(self, show_process = True):
        self.nnet.train()
        loss_list = []
        total_loss = 0.0
        self.optimizer.zero_grad()
        
        # Run the ansatz circuit
        qc_ans, _ = self.ansatz_circuit.generate_measurement_circuit("I" * self.num_qubits)
        param_dict = {param: value for param, value in zip(qc_ans.parameters, self.params)}
        qc_ans.assign_parameters(param_dict)
        ansatz_res = self._run_sampler(qc_ans)
        ansatz_keys = {s[::-1] for s in ansatz_res.keys()}

        # Run the measurement circuits
        meas_res = {}
        star_idxs = {}

        meas_keys = set()
        meas_trans_keys = set()

        for pstr in self.operator_list:
            qc_meas, star_idx = self.ansatz_circuit.generate_measurement_circuit(pstr)
            qc_meas.assign_parameters(self.params)
            pstr_res = self._run_sampler(qc_meas)
            meas_res[pstr] = pstr_res
            star_idxs[pstr] = star_idx
            meas_key = {s[::-1] for s in pstr_res.keys()}
            meas_keys = meas_keys.union(meas_key)

            for iter_str in pstr_res.keys():
                s_str = [(2 * int(x) - 1) for x in iter_str]
                s_str.reverse()
                if star_idx >= 0: # Pauli string has 'X' or 'Y'.
                    if s_str[star_idx] == 1:
                        s_str[star_idx] = -1
                s_conj = self._pauli_transformation(s_str, pstr)
                s_conj_str = ''.join(['1' if x == 1 else '0' for x in s_conj.cpu().detach().numpy()])
                s_string = ''.join(['1' if x == 1 else '0' for x in s_str])
                meas_trans_keys.add(s_conj_str)
                meas_keys.add(s_string)

        total_meas_keys = meas_keys.union(meas_trans_keys)

        # add 1e-7 to the bit strings that are not in the ansatz result
        eps = 1e-7

        pre_ansatz = ansatz_res

        ansatz_unique_items = list(ansatz_keys.difference(total_meas_keys))
        ansatz_keys = ansatz_keys.intersection(total_meas_keys)
        for bstr in list(ansatz_res.keys()):
            ansatz_res[bstr] = (ansatz_res[bstr] - eps if ansatz_res[bstr] > 0 else ansatz_res + eps)
            if bstr[::-1] in ansatz_unique_items:
                del ansatz_res[bstr]
        
        meas_unique_items = list(total_meas_keys.difference(ansatz_keys))     # (bstr_t U bstr_m) \ bstr_a
        meas_unique_count = len(meas_unique_items)
        ansatz_count = len(ansatz_keys)
        scaled_eps = eps * ansatz_count / np.max([meas_unique_count, 1])

        f = 0

        for bstr in meas_unique_items:
            ansatz_res[bstr[::-1]] = scaled_eps

        for pstr in self.operator_list:
            pstr_res = meas_res[pstr]
            pstr_keys = {s[::-1] for s in pstr_res.keys()}
            meas_not_in_pstr = list(total_meas_keys.difference(pstr_keys))
            for bstr in pstr_keys:
                pstr_res[bstr[::-1]] = pstr_res[bstr[::-1]] - eps
            
            for bstr in meas_not_in_pstr:
                pstr_res[bstr[::-1]] = eps * len(pstr_keys) / np.max([len(meas_not_in_pstr), 1])
            
            meas_res[pstr] = pstr_res

        for s, _ in ansatz_res.items():
            p = ansatz_res[s]
            pp = pre_ansatz.get(s, 0)
            f += np.sqrt(p) * np.sqrt(pp)
                
        if self.print_mode: print(f"State Fidelity: {(f*f):.6f}")

        # Neural network training
        iterator = tqdm(range(self.num_epoch), desc="Processing") if show_process else range(self.num_epoch)
        for epoch in iterator:
            ham = 0.0
            denominator = 0.0

            for a_str, a_prob in ansatz_res.items():
                a_list = torch.tensor([int(x) for x in a_str], dtype=torch.float32, device=self.device)
                a_tensor = 2 * a_list - 1
                a_tensor = a_tensor.flip(dims=[0])
                fs_denom = self.nnet(a_tensor)
                denominator += fs_denom * fs_denom * a_prob

            for idx in range(len(self.operator_list)):
                pstr = self.operator_list[idx]
                string_prob = meas_res[pstr]
                numerator = 0.0

                for m_str, m_prob in string_prob.items():
                    sign = 1
                    m_list = torch.tensor([int(x) for x in m_str], dtype=torch.float32, device=self.device)
                    m_tensor = 2 * m_list - 1
                    m_tensor = m_tensor.flip(dims = [0])
                    if star_idxs[pstr] >= 0:
                        if m_tensor[star_idxs[pstr]] == 1:
                            sign *= -1
                            m_tensor[star_idxs[pstr]] *= -1
                        for i in range(len(pstr)):
                            if pstr[i] == 'Z' and m_tensor[i] == 1:
                                sign *= -1
                        m_transformed = self._pauli_transformation(m_tensor, pstr)
                        fs = self.nnet(m_tensor)
                        fs_t = self.nnet(m_transformed)
                        numerator += fs * fs_t * m_prob * sign
                    else:
                        fs = self.nnet(m_tensor)
                        for i in range(len(pstr)):
                            if pstr[i] == 'Z' and m_tensor[i] == 1:
                                sign *= -1
                        numerator += (fs ** 2) * m_prob * sign

                mean_pauli = numerator / denominator * self.coeff_list[idx]
                ham += mean_pauli
            
            running_loss = ham
            running_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += ham.item()

            loss_list.append(ham.item())
        
        if self.print_mode: print('Finished Training')
        
        return self.nnet, loss_list
    

# VQNHE Trainer with exact results (using statevector simulator)
class ExactVQNHETrainer:
    def __init__(self, ansatz_circuit: Union[IsingAnsatzCircuit, HeiAnsatzCircuit],
            params : ParameterVector, operator_list, coeff_list, nn_options: dict, 
            print_mode = False):
        '''
        qc_options (optional, dict): ['shots', 'noise_model']
        nn_options (dict): ['num_epoch', 'features', 'lr'(optional)]
        '''
        self.ansatz_circuit = ansatz_circuit
        self.num_qubits = ansatz_circuit.num_qubits
        self.operator_list = operator_list
        self.coeff_list = coeff_list
        self.params = params
        self.print_mode = print_mode

        if 'num_epoch' not in nn_options.keys() or 'features' not in nn_options.keys(): 
            raise ValueError("nn_options must include 'num_epoch' and 'features'") 
        self.num_epoch = nn_options['num_epoch']
        self.features = nn_options['features']
        self.lr = nn_options.get('lr', 1e-3)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.print_mode: print(f"Torch running on " + self.device)

        self.nnet = NeuralNet(num_qubits = self.num_qubits, features = self.features).to(self.device)
        betas = (0.9, 0.999)
        self.optimizer = torch.optim.Adam(self.nnet.parameters(), lr = self.lr, betas = betas)


    def _pauli_transformation(self, bstr: torch.Tensor, pauli_str: str) -> torch.Tensor:
        # Ensure bstr is a tensor of integers
        conj_list = torch.ones_like(bstr, dtype=torch.float32)

        for i in range(len(pauli_str)):
            if pauli_str[i] in ["X", "Y"]:
                conj_list[i] = -1 if bstr[i] == 1 else 1
            else:
                conj_list[i] = bstr[i]

        return conj_list


    def _run_qc(self, qc):
        statevector = Statevector(qc)
        probabilities = statevector.probabilities()
        result = {format(i, f'0{self.num_qubits}b'): prob for i, prob in enumerate(probabilities)}
        return result

    def train(self, show_process = True):
        self.nnet.train()
        loss_list = []
        total_loss = 0.0
        self.optimizer.zero_grad()
        
        # Run the ansatz circuit
        qc_ans, _ = self.ansatz_circuit.generate_measurement_circuit("I" * self.num_qubits, measure = False)
        param_dict = {param: value for param, value in zip(qc_ans.parameters, self.params)}
        qc_ans.assign_parameters(param_dict, inplace=True)
        ansatz_res = self._run_qc(qc_ans)

        # Run the measurement circuits
        meas_res = {}
        star_idxs = {}

        for pstr in self.operator_list:
            qc_meas, star_idx = self.ansatz_circuit.generate_measurement_circuit(pstr, measure = False)
            qc_meas.assign_parameters(self.params, inplace=True)
            meas_res[pstr] = self._run_qc(qc_meas)
            star_idxs[pstr] = star_idx

        # Neural network training
        iterator = tqdm(range(self.num_epoch), desc="Processing") if show_process else range(self.num_epoch)
        for epoch in iterator:
            ham = 0.0
            denominator = 0.0

            for a_str, a_prob in ansatz_res.items():
                a_list = torch.tensor([int(x) for x in a_str], dtype=torch.float32, device=self.device)
                a_tensor = 2 * a_list - 1
                a_tensor = a_tensor.flip(dims=[0])
                fs_denom = self.nnet(a_tensor)
                denominator += fs_denom * fs_denom * a_prob

            for idx in range(len(self.operator_list)):
                pstr = self.operator_list[idx]
                string_prob = meas_res[pstr]
                numerator = 0.0

                for m_str, m_prob in string_prob.items():
                    sign = 1
                    m_list = torch.tensor([int(x) for x in m_str], dtype=torch.float32, device=self.device)
                    m_tensor = 2 * m_list - 1
                    m_tensor = m_tensor.flip(dims = [0])
                    if star_idxs[pstr] >= 0:
                        if m_tensor[star_idxs[pstr]] == 1:
                            sign *= -1
                            m_tensor[star_idxs[pstr]] *= -1
                        for i in range(len(pstr)):
                            if pstr[i] == 'Z' and m_tensor[i] == 1:
                                sign *= -1
                        m_transformed = self._pauli_transformation(m_tensor, pstr)
                        fs = self.nnet(m_tensor)
                        fs_t = self.nnet(m_transformed)
                        numerator += fs * fs_t * m_prob * sign
                    else:
                        fs = self.nnet(m_tensor)
                        for i in range(len(pstr)):
                            if pstr[i] == 'Z' and m_tensor[i] == 1:
                                sign *= -1
                        numerator += (fs ** 2) * m_prob * sign

                mean_pauli = numerator / denominator * self.coeff_list[idx]
                ham += mean_pauli
            
            running_loss = ham
            running_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += ham.item()

            loss_list.append(ham.item())
        
        if self.print_mode: print('Finished Training')
        
        return self.nnet, loss_list, ansatz_res, meas_res



# Plot the plots!
class VQNHEPlotter:
    def __init__(self, num_qubits, nnet, loss_list, exp_type, vqe = None, exact = None):
        self.num_qubits = num_qubits
        self.nnet = nnet
        self.loss_list = loss_list
        self.exp_type = exp_type
        self.vqe = vqe
        self.exact = exact
        self._loss_plot()
        if self.nnet is not None: self._bitstr_plot()

    def _loss_plot(self):
        plot_name = f"{self.num_qubits}q_{self.exp_type}_loss.jpg"
        plt.plot(range(len(self.loss_list)), self.loss_list, label = 'VQNHE')
        if self.vqe is not None: plt.plot(range(len(self.loss_list)), [self.vqe for _ in range(len(self.loss_list))], label = 'VQE')
        if self.exact is not None: plt.plot(range(len(self.loss_list)), [self.exact  for _ in range(len(self.loss_list))], label = 'Exact')

        plt.xlabel('epoch (5)')
        plt.ylabel('Expectation value')
        plt.title(f'VQNHE on {self.num_qubits} qubits')
        plt.legend()
        plt.savefig('./result/' + plot_name, dpi = 300)

    
    def _bitstr_plot(self):
        plot_name = f"{self.num_qubits}q_{self.exp_type}_bitstr.jpg"
        
        nn_res = []
        for s in range(2**self.num_qubits):
            ss = [int(digit) for digit in bin(s)[2:].zfill(self.num_qubits)]
            ss.reverse()
            ss = torch.tensor(ss, device = next(self.nnet.parameters()).device, dtype=torch.float32)
            nn_res.append(self.nnet(ss).cpu().detach().numpy())

        plt.clf()
        plt.plot(nn_res, label = 'Neural Network')
        plt.xlabel('s')
        plt.ylabel('f(s)')
        plt.title(f"{self.num_qubits}-qubit VQNHE: Neural Network Landscape")
        plt.legend()
        plt.savefig('./result/' + plot_name, dpi = 300)