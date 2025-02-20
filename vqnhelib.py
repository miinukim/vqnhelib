'''
VQNHE Simulation

Author: Minwoo Kim (Dept. of Computer Science & Engineering, Seoul National University)
'''

#Import Qiskit
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, ParameterVector
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

#from projUNN import projunn

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
            for idx in range(num_qubits - 1):
                pauli_str = i_str[:idx] + pauli_let + pauli_let + i_str[idx+2:]
                operator_list.append(pauli_str)
        
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
        self.ansatz_len = ansatz_len
        self.n_q = num_qubits
        self.num_params = ansatz_len * num_qubits
        self.params = ParameterVector("theta", length=self.num_params)
        self._build_circuit()

    def _build_circuit(self):
         # Prepare |phi-> state
        for i in range(self.n_q):
            self.x(i)
        
        for i in range(self.n_q):
            if i % 2 == 0:
                self.h(i)
                self.cx(i, (i+1)%self.n_q)

        for d in range(self.ansatz_len):
            for i in range(self.n_q):
                self.rzz(self.params[d * self.n_q + i], i, (i+1)%self.n_q)
                self.swap(i, (i+1)%self.n_q)


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
        self.FC1 = nn.Linear(num_qubits, features)
        self.FC2 = nn.Linear(features, features)
        self.phi = nn.Parameter(torch.empty(features))
        self.activation = nn.Tanh()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        init.normal_(self.phi)

    def forward(self, x):
        s1 = F.sigmoid(self.FC1(x))
        s2 = F.sigmoid(self.FC2(s1))
        exp = torch.dot(self.phi, self.activation(s2))
        f = torch.exp(exp)
        return f


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
        s1 = F.sigmoid(self.FC1(x))
        s2 = F.sigmoid(self.FC2(s1))
        exp = torch.dot(self.phi, self.activation(s2))

        if self.print_mode: return exp  # Return \theta if in print_mode
        else: return torch.exp(torch.complex(torch.tensor(0, dtype=torch.float32, device=self.device), exp))
        


## VQNHE Trainer
class VQNHETrainer:
    def __init__(self, ansatz_circuit: Union[IsingAnsatzCircuit, HeiAnsatzCircuit],
            params : ParameterVector, operator_list, coeff_list, nn_options: dict, 
            qc_options: Optional[dict] = {}, print_mode = False, cutoff = -20):
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

    def train(self, show_process = True):
        self.nnet.train()
        loss_list = []
        total_loss = 0.0
        self.optimizer.zero_grad()
        
        # Run the ansatz circuit
        qc_ans, _ = self.ansatz_circuit.generate_measurement_circuit("I" * self.num_qubits)
        param_dict = {param: value for param, value in zip(qc_ans.parameters, self.params)}
        qc_ans.assign_parameters(param_dict, inplace= True)
        ansatz_res = self._run_sampler(qc_ans)

        ansatz_keys = set(ansatz_res.keys())

        # Run the measurement circuits
        meas_res = {}
        star_idxs = {}

        meas_keys = set()

        for pstr in self.operator_list:
            qc_meas, star_idx = self.ansatz_circuit.generate_measurement_circuit(pstr)
            qc_meas.assign_parameters(param_dict, inplace=True)
            meas_res[pstr] = self._run_sampler(qc_meas)
            star_idxs[pstr] = star_idx
            meas_keys = meas_keys.union(set(meas_res[pstr].keys()))

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
                denominator += torch.abs(fs_denom) ** 2 * a_prob

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
                        numerator += (torch.abs(fs) ** 2) * m_prob * sign

                mean_pauli = numerator / denominator * self.coeff_list[idx]
                ham += mean_pauli
            
            running_loss = ham
            running_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_list.append(ham.item())

            '''
            if epoch > 20:
                if np.abs(loss_list[-1] - loss_list[-2]) < 1e-4 * loss_list[-1]: break
                if loss_list[-1] < self.cutoff: break
            '''
        
        if self.print_mode: print('Finished Training')
        
        return self.nnet, loss_list, ansatz_keys, meas_keys


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
        
        # Bind the parameters
        qc_ans, _ = self.ansatz_circuit.generate_measurement_circuit("I" * self.num_qubits)
        param_dict = {param: value for param, value in zip(qc_ans.parameters, self.params)}

        # Run the measurement circuits
        real_res = {}
        imag_res = {}
        star_idxs = {}

        for pstr in self.operator_list:
            qc_meas, star_idx = self.ansatz_circuit.generate_measurement_circuit(pstr, measure=not(self.exact))
            qc_meas_imag, _ = self.ansatz_circuit.generate_measurement_circuit(pstr, measure=not(self.exact), real=False)
            qc_meas.assign_parameters(param_dict, inplace=True)
            qc_meas_imag.assign_parameters(param_dict, inplace=True)
            
            if self.exact:
                real_res[pstr] = self._run_qc(qc_meas)
                imag_res[pstr] = self._run_qc(qc_meas_imag)
            else:
                real_res[pstr] = self._run_sampler(qc_meas)
                imag_res[pstr] = self._run_sampler(qc_meas_imag)
            star_idxs[pstr] = star_idx
            

        # Neural network training
        iterator = tqdm(range(self.num_epoch), desc="Processing") if show_process else range(self.num_epoch)
        for epoch in iterator:
            ham = 0.0

            ## No need to run ansatz for normalization

            for idx in range(len(self.operator_list)):
                pstr = self.operator_list[idx]
                real_prob = real_res[pstr]
                imag_prob = imag_res[pstr]
                numerator = 0.0

                for m_str, m_prob in real_prob.items():
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
                        numerator += (torch.conj(fs) * fs_t).real * m_prob * sign
                    else:
                        fs = self.nnet(m_tensor)
                        for i in range(len(pstr)):
                            if pstr[i] == 'Z' and m_tensor[i] == 1:
                                sign *= -1
                        numerator += (torch.conj(fs) * fs).real * m_prob * sign

                for m_str, m_prob in imag_prob.items():
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
                        numerator += (torch.conj(fs) * fs_t).imag * m_prob * sign
                        
                exp_pauli = numerator * self.coeff_list[idx]
                ham += exp_pauli
            
            running_loss = ham
            running_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += ham.item()

            loss_list.append(ham.item())
        
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
        
        return self.nnet, loss_list



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