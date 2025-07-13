"""
Pytest suite for the entanglement-distance jewel MVP project.
This file should be placed in the 'Tests' directory.
To run: navigate to the 'Jewel_Test' root directory and run 'pytest'.
"""
import pytest
import numpy as np
import quimb.tensor as qtn
import quimb

# Adjust path to import modules from the parent directory (Jewel_Test)
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from our project modules
from ham_xxz import build_xxz_mpo
from dmrg_runner import run_dmrg
from mi_tools import (
    _calculate_subsystem_entropy,
    calculate_mutual_information_I_r,
    calculate_central_charge,
    calculate_correlation_length,
    fit_beta_from_I_r,
    fit_k_from_d_E_r,
    get_max_bond_entropy
)
from validate_jewel import main as validate_jewel_main, run_and_evaluate_delta

# --- Fixtures ---

@pytest.fixture
def mpo_l2_delta1():
    """MPO for L=2, Delta=1.0 (XXX model)."""
    return build_xxz_mpo(L=2, Delta=1.0)

@pytest.fixture
def mpo_l2_delta0():
    """MPO for L=2, Delta=0.0 (XX model)."""
    return build_xxz_mpo(L=2, Delta=0.0)

@pytest.fixture
def mps_l2_product_00():
    """MPS for L=2, product state |00>."""
    psi = qtn.MPS_computational_state("00")
    psi.left_canonicalize(inplace=True) 
    return psi

@pytest.fixture
def mps_l2_bell_state():
    """MPS for L=2, Bell state (|00> + |11>) / sqrt(2)."""
    psi_dense = np.zeros(4, dtype=complex) 
    psi_dense[0] = 1.0 / np.sqrt(2)
    psi_dense[3] = 1.0 / np.sqrt(2)
    psi = qtn.MatrixProductState.from_dense(psi_dense, dims=[2, 2])
    psi.left_canonicalize(inplace=True) 
    return psi

# --- Tests for ham_xxz.py ---

def test_build_xxz_mpo_l2_properties(mpo_l2_delta1):
    assert mpo_l2_delta1.L == 2
    assert mpo_l2_delta1.phys_dim(0) == 2
    assert mpo_l2_delta1.max_bond() == 5

def test_build_xxz_mpo_hermiticity_small(mpo_l2_delta1):
    H_dense = mpo_l2_delta1.to_dense()
    assert np.allclose(H_dense, H_dense.conj().T)

# --- Tests for dmrg_runner.py ---

def test_dmrg_l2_xx_model_energy(mpo_l2_delta0):
    L = 2
    bond_dims = [4, 8] 
    cutoffs = [1e-10, 1e-10]
    n_sweeps = 4
    energy_gs, psi_gs, ham_variance, _ = run_dmrg(
        mpo_l2_delta0, L, bond_dims, cutoffs, n_sweeps, verbosity=0
    )
    assert psi_gs is not None
    assert np.isclose(energy_gs, -0.5, atol=1e-6) 
    assert ham_variance is not None 
    if not np.isnan(ham_variance):
        assert ham_variance < 1e-8

def test_dmrg_variance_is_low_for_gs(mpo_l2_delta1):
    L = 2
    bond_dims = [8]
    cutoffs = [1e-10]
    n_sweeps = 4
    _, _, ham_variance, _ = run_dmrg(
        mpo_l2_delta1, L, bond_dims, cutoffs, n_sweeps, verbosity=0
    )
    assert ham_variance is not None
    if not np.isnan(ham_variance):
        assert ham_variance < 1e-8

# --- Tests for mi_tools.py ---

@pytest.mark.parametrize("sites_to_keep, expected_entropy_nats", [
    ([0], 0.0), ([1], 0.0), ([0, 1], 0.0) 
])
def test_calculate_subsystem_entropy_product_state(mps_l2_product_00, sites_to_keep, expected_entropy_nats):
    entropy = _calculate_subsystem_entropy(mps_l2_product_00, sites_to_keep, num_lanczos_vecs=4)
    # Adjusted tolerance for zero-valued entropies
    assert np.isclose(entropy, expected_entropy_nats, atol=1e-5)


@pytest.mark.parametrize("sites_to_keep, expected_entropy_nats", [
    ([0], np.log(2)), ([1], np.log(2)), ([0, 1], 0.0)          
])
def test_calculate_subsystem_entropy_bell_state(mps_l2_bell_state, sites_to_keep, expected_entropy_nats):
    entropy = _calculate_subsystem_entropy(mps_l2_bell_state, sites_to_keep, num_lanczos_vecs=4)
    if expected_entropy_nats == 0.0:
        assert np.isclose(entropy, expected_entropy_nats, atol=1e-5) # Adjusted for zero
    else:
        assert np.isclose(entropy, expected_entropy_nats, atol=1e-7)


def test_mutual_information_product_state(mps_l2_product_00):
    L = 2
    r_max = 1
    r_vals, I_r_vals = calculate_mutual_information_I_r(mps_l2_product_00, L, r_max, num_lanczos_vecs=4)
    assert len(I_r_vals) == 1
    # Adjusted tolerance for zero-valued MI
    assert np.isclose(I_r_vals[0], 0.0, atol=1e-5)

def test_mutual_information_bell_state(mps_l2_bell_state):
    L = 2
    r_max = 1
    r_vals, I_r_vals = calculate_mutual_information_I_r(mps_l2_bell_state, L, r_max, num_lanczos_vecs=4)
    assert len(I_r_vals) == 1
    expected_mi_nats = 2 * np.log(2)
    assert np.isclose(I_r_vals[0], expected_mi_nats, atol=1e-7)

def test_get_max_bond_entropy(mps_l2_bell_state, mps_l2_product_00):
    assert np.isclose(get_max_bond_entropy(mps_l2_product_00), 0.0, atol=1e-7)
    assert np.isclose(get_max_bond_entropy(mps_l2_bell_state), np.log(2), atol=1e-7)

def test_central_charge_runs(mps_l2_bell_state):
    L = mps_l2_bell_state.L
    if L < 4:
        c_val, c_err_val = calculate_central_charge(mps_l2_bell_state, L)
        assert np.isnan(c_val)
        assert np.isnan(c_err_val)
    else: 
        psi_large_test = qtn.MPS_rand_state(L=8, bond_dim=4, phys_dim=2)
        psi_large_test.left_canonicalize(inplace=True) 
        c, c_err = calculate_central_charge(psi_large_test, L=8)
        assert isinstance(c, float)

def test_correlation_length_runs(mps_l2_product_00):
    L = mps_l2_product_00.L
    r_max = 1 
    sz_op = quimb.spin_operator('Z', S=0.5)
    xi = calculate_correlation_length(mps_l2_product_00, L, r_max, sz_op)
    assert isinstance(xi, float) 

def test_fit_beta_synthetic_data():
    r_values = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    true_beta = 2.0
    C = 10.0
    I_r_values = C * (r_values ** (-true_beta)) + np.random.normal(0, 0.001, size=len(r_values))
    beta_fit, beta_err, R2_I = fit_beta_from_I_r(list(r_values), list(I_r_values))
    assert np.isclose(beta_fit, true_beta, atol=0.05)
    assert R2_I > 0.98

def test_fit_k_synthetic_data():
    r_values = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    true_k = 1.5
    d_E_values = true_k * r_values + np.random.normal(0, 0.01, size=len(r_values))
    k_fit, k_err, R2_dE = fit_k_from_d_E_r(list(r_values), list(d_E_values))
    assert np.isclose(k_fit, true_k, atol=0.05)
    assert R2_dE > 0.98

# --- Tests for validate_jewel.py (Integration) ---

@pytest.mark.slow
def test_validate_jewel_main_runs_small_system(tmp_path, monkeypatch):
    output_dir = tmp_path / "test_results"
    L_test_validate = 4
    bond_dims_test_validate = [4, 8] 
    cutoffs_test_validate = [1e-7, 1e-7]
    n_sweeps_test_validate = 2 
    r_max_test_validate = 1    
    if L_test_validate >= 4: 
        r_max_test_validate = 2

    monkeypatch.setattr('validate_jewel.L_DEFAULT', L_test_validate)
    monkeypatch.setattr('validate_jewel.BOND_DIMS_DEFAULT', bond_dims_test_validate)
    monkeypatch.setattr('validate_jewel.CUTOFFS_DEFAULT', cutoffs_test_validate)
    monkeypatch.setattr('validate_jewel.N_SWEEPS_DEFAULT', n_sweeps_test_validate)
    monkeypatch.setattr('validate_jewel.R_MAX_DEFAULT', r_max_test_validate)
    monkeypatch.setattr('validate_jewel.FIT_R_MIN_DEFAULT', 1) 
    monkeypatch.setattr('validate_jewel.FIT_R_MAX_FACTOR_DEFAULT', 1.0) 
    
    results_delta1, summary_delta1 = run_and_evaluate_delta(
        delta_val=1.0, L=L_test_validate,
        bond_dims_schedule=bond_dims_test_validate,
        cutoffs_schedule=cutoffs_test_validate,
        n_sweeps=n_sweeps_test_validate,
        r_max_mi=r_max_test_validate, r_max_corr=r_max_test_validate,
        fit_r_min=1, fit_r_max_factor=1.0,
        output_dir=str(output_dir)
    )
    
    results_delta2, summary_delta2 = run_and_evaluate_delta(
        delta_val=2.0, L=L_test_validate,
        bond_dims_schedule=bond_dims_test_validate,
        cutoffs_schedule=cutoffs_test_validate,
        n_sweeps=n_sweeps_test_validate,
        r_max_mi=r_max_test_validate, r_max_corr=r_max_test_validate,
        fit_r_min=1, fit_r_max_factor=1.0,
        output_dir=str(output_dir)
    )

    assert "error" not in results_delta1, f"Delta 1.0 run failed: {results_delta1.get('error')}"
    assert "error" not in results_delta2, f"Delta 2.0 run failed: {results_delta2.get('error')}"
    
    assert os.path.exists(output_dir / "results_Delta1.0.json")
    assert os.path.exists(output_dir / "results_Delta2.0.json")

    assert 'status' in summary_delta1
    assert 'status' in summary_delta2
