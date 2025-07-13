"""
Tools for calculating Mutual Information (MI), Entanglement Distance (d_E),
Central Charge (c), Correlation Length (xi), and related fitting parameters
from a Matrix Product State (MPS).
"""
import numpy as np
import quimb # For quimb.spin_operator, quimb.approx_spectral_function
import quimb.tensor as qtn
from scipy.stats import linregress
from scipy.optimize import curve_fit

# --- Helper function for Von Neumann Entropy of a subsystem ---
def _calculate_subsystem_entropy(psi_gs, sites_A_to_keep, num_lanczos_vecs=20):
    """
    Calculates the von Neumann entropy of a subsystem A from a pure state MPS.
    S(A) = -Tr(rho_A log rho_A) (using natural logarithm for nats)

    Args:
        psi_gs (quimb.tensor.MatrixProductState): The pure state MPS.
        sites_A_to_keep (list[int]): List of site indices defining subsystem A.
        num_lanczos_vecs (int): Number of Lanczos vectors for approx_spectral_function.

    Returns:
        float: The von Neumann entropy of subsystem A in nats.
               Returns 0.0 if sites_A_to_keep is empty.
               Returns np.nan if calculation encounters an error.
    """
    L = psi_gs.L
    if not sites_A_to_keep:
        return 0.0
    if not all(isinstance(s, int) and 0 <= s < L for s in sites_A_to_keep):
        raise ValueError(f"All site indices in {sites_A_to_keep} must be integers within [0, {L-1}]")

    # --- 1. Construct Reduced Density Matrix (RDM) rho_A for sites_A_to_keep ---
    psi_ket_local = psi_gs.copy()
    psi_bra_local = psi_gs.H.copy()

    ket_phys_id_pattern = psi_ket_local.site_ind_id if psi_ket_local.site_ind_id else 'k{}'
    
    if ket_phys_id_pattern and len(ket_phys_id_pattern) > 1 and ket_phys_id_pattern[0].isalpha():
        bra_phys_id_pattern = f"b{ket_phys_id_pattern[1:]}" 
    elif ket_phys_id_pattern: 
         bra_phys_id_pattern = f"b_{ket_phys_id_pattern}"
    else: 
        bra_phys_id_pattern = 'b{}'

    if ket_phys_id_pattern == bra_phys_id_pattern: 
        bra_phys_id_pattern = f"bra_phys_{ket_phys_id_pattern}"


    idx_map_bra_phys_globally = {
        ket_phys_id_pattern.format(s): bra_phys_id_pattern.format(s)
        for s in range(L)
    }
    psi_bra_local.reindex(idx_map_bra_phys_globally, inplace=True)

    sites_to_trace_out = [s for s in range(L) if s not in sites_A_to_keep]
    
    trace_idx_map = {
        bra_phys_id_pattern.format(s): ket_phys_id_pattern.format(s)
        for s in sites_to_trace_out
    }
    psi_bra_local.reindex(trace_idx_map, inplace=True)
    
    rho_A_tn = (psi_bra_local & psi_ket_local)

    # --- 2. Convert RDM Tensor Network to a Scipy LinearOperator ---
    open_ket_indices = [ket_phys_id_pattern.format(s) for s in sites_A_to_keep]
    open_bra_indices = [bra_phys_id_pattern.format(s) for s in sites_A_to_keep]
    
    # FIX: Use .outer_inds() for TensorNetwork objects to get uncontracted (open) indices
    current_open_rho_A_inds = rho_A_tn.outer_inds() 
    
    if not (all(idx in current_open_rho_A_inds for idx in open_ket_indices) and \
            all(idx in current_open_rho_A_inds for idx in open_bra_indices)):
        if sites_A_to_keep and len(sites_A_to_keep) == L: 
             pass 
        elif sites_A_to_keep: 
            # print(f"Warning: Mismatch in expected open indices for sites {sites_A_to_keep} when creating LinearOperator.")
            # print(f"  Expected Kets: {open_ket_indices}, Expected Bras: {open_bra_indices}")
            # print(f"  Actual Open Inds: {current_open_rho_A_inds}")
            return np.nan


    rho_A_lo = rho_A_tn.aslinearoperator(open_ket_indices, open_bra_indices)

    # --- 3. Calculate von Neumann Entropy using approx_spectral_function ---
    nlogn = lambda x_val: -x_val.real * np.log(x_val.real) if x_val.real > 1e-15 else 0.0
    
    dim_subsystem = 2**len(sites_A_to_keep)
    current_R = min(num_lanczos_vecs, dim_subsystem) 
    if current_R <= 0 : 
        return 0.0

    try:
        raw_S_A = quimb.approx_spectral_function(rho_A_lo, f=nlogn, R=current_R)
        S_A = raw_S_A.real 
    except Exception as e:
        # print(f"Error in approx_spectral_function for sites {sites_A_to_keep}: {e}")
        # import traceback
        # traceback.print_exc()
        return np.nan 

    return S_A if S_A is not None else np.nan

# --- Main Calculation Functions (These should be mostly correct from previous version) ---

def calculate_mutual_information_I_r(psi_gs, L, r_max, num_lanczos_vecs=20):
    avg_I_r_values = []
    r_values = list(range(1, r_max + 1))
    for r_dist in r_values:
        I_at_this_r = []
        num_pairs = 0
        for i in range(L - r_dist):
            j = i + r_dist
            S_i = _calculate_subsystem_entropy(psi_gs, [i], num_lanczos_vecs)
            S_j = _calculate_subsystem_entropy(psi_gs, [j], num_lanczos_vecs)
            S_ij = _calculate_subsystem_entropy(psi_gs, [i, j], num_lanczos_vecs)
            if np.isnan(S_i) or np.isnan(S_j) or np.isnan(S_ij):
                continue
            mutual_info_ij = S_i + S_j - S_ij
            I_at_this_r.append(max(0.0, mutual_info_ij)) 
            num_pairs += 1
        if num_pairs > 0: avg_I_r_values.append(np.mean(I_at_this_r))
        else: avg_I_r_values.append(np.nan)
    return r_values, avg_I_r_values

def calculate_entanglement_distance_d_E_r(r_values, avg_I_r_values):
    d_E_r_values = []
    for I_r in avg_I_r_values:
        if np.isnan(I_r) or I_r <= 1e-12: d_E_r_values.append(np.nan)
        else: d_E_r_values.append(I_r**(-0.5))
    return d_E_r_values

def fit_beta_from_I_r(r_values_fit, I_r_values_fit):
    valid_indices = [i for i, (r, I_val) in enumerate(zip(r_values_fit, I_r_values_fit)) 
                     if not np.isnan(r) and not np.isnan(I_val) and r > 0 and I_val > 1e-12]
    if len(valid_indices) < 2: return np.nan, np.nan, np.nan 
    log_r = np.log([r_values_fit[i] for i in valid_indices])
    log_I = np.log([I_r_values_fit[i] for i in valid_indices])
    slope, intercept, r_value, p_value, std_err = linregress(log_r, log_I)
    beta = -slope
    beta_err = std_err 
    R2 = r_value**2 if not np.isnan(r_value) else np.nan
    return beta, beta_err, R2

def fit_k_from_d_E_r(r_values_fit, d_E_r_values_fit):
    valid_indices = [i for i, (r, dE_val) in enumerate(zip(r_values_fit, d_E_r_values_fit))
                     if not np.isnan(r) and not np.isnan(dE_val)]
    if len(valid_indices) < 2: return np.nan, np.nan, np.nan
    r_fit = np.array([r_values_fit[i] for i in valid_indices])
    d_E_fit = np.array([d_E_r_values_fit[i] for i in valid_indices])
    r_fit_reshaped = r_fit[:, np.newaxis]
    try:
        k_val_arr, _, _, _ = np.linalg.lstsq(r_fit_reshaped, d_E_fit, rcond=None)
        k_val = k_val_arr[0]
        sum_sq_residuals = np.sum((d_E_fit - k_val * r_fit)**2)
        sum_sq_total = np.sum((d_E_fit - np.mean(d_E_fit))**2) 
        if sum_sq_total < 1e-12 : R2 = 1.0 if sum_sq_residuals < 1e-12 else 0.0
        else: R2 = 1 - (sum_sq_residuals / sum_sq_total) if sum_sq_total > 0 else np.nan
        _, _, _, _, stderr_lin = linregress(r_fit, d_E_fit)
        k_err = stderr_lin 
    except np.linalg.LinAlgError: return np.nan, np.nan, np.nan
    return k_val, k_err, R2

def calculate_central_charge(psi_gs, L):
    if L < 4: return np.nan, np.nan 
    min_block_for_fit = 2
    max_block_for_fit = L // 2
    ells = np.arange(min_block_for_fit, max_block_for_fit + 1, 2) 
    if len(ells) < 2 : 
        ells = np.arange(min_block_for_fit, max_block_for_fit +1) 
        if len(ells) < 2: return np.nan, np.nan
    entropies_block_bits = np.array([psi_gs.entropy(ell) for ell in ells])
    entropies_block_nats = entropies_block_bits * np.log(2)
    valid_indices = ~np.isnan(entropies_block_nats)
    if np.sum(valid_indices) < 2: return np.nan, np.nan
    ells_fit = ells[valid_indices]
    entropies_fit_nats = entropies_block_nats[valid_indices]
    x_cc = np.log((L / np.pi) * np.sin(np.pi * ells_fit / L))
    slope, _, _, _, std_err = linregress(x_cc, entropies_fit_nats)
    c_calabrese = 3 * slope
    c_err = 3 * std_err 
    return c_calabrese, c_err

def calculate_correlation_length(psi_gs, L, r_max, op_for_corr):
    avg_C_r_values = []
    r_values_corr = list(range(1, r_max + 1))
    for r_dist in r_values_corr:
        r_corrs_vals = []
        num_pairs = 0
        for i_ref in range(L - r_dist):
            j_curr = i_ref + r_dist
            try:
                corr_val_raw = psi_gs.correlation(A=op_for_corr, i=i_ref, j=j_curr, B=None)
                corr_val = corr_val_raw.item() if hasattr(corr_val_raw, 'item') else corr_val_raw
                r_corrs_vals.append(corr_val.real if isinstance(corr_val, complex) else corr_val)
                num_pairs +=1
            except Exception: pass 
        if num_pairs > 0: avg_C_r_values.append(np.mean(r_corrs_vals))
        else: avg_C_r_values.append(np.nan) 
    valid_indices = [i for i, C_r in enumerate(avg_C_r_values) 
                     if not np.isnan(C_r) and np.abs(C_r) > 1e-12]
    if len(valid_indices) < 2: return np.nan 
    r_fit = np.array([r_values_corr[i] for i in valid_indices])
    log_abs_C_fit = np.log(np.abs([avg_C_r_values[i] for i in valid_indices]))
    slope, _, _, _, _ = linregress(r_fit, log_abs_C_fit)
    if slope >= -1e-9 or np.isnan(slope):
        return np.inf if abs(slope) < 1e-9 else np.nan
    xi = -1.0 / slope
    return xi

def get_max_bond_entropy(psi_gs):
    L = psi_gs.L
    if L < 2: return 0.0
    bond_entropies_bits = [psi_gs.entropy(i) for i in range(1, L)]
    bond_entropies_nats = [s * np.log(2) for s in bond_entropies_bits if not np.isnan(s)]
    return np.max(bond_entropies_nats) if bond_entropies_nats else np.nan

def extract_all_properties(psi_gs, L, r_max_mi, r_max_corr, fit_r_min=1, fit_r_max_factor=0.8):
    results = {}
    num_lanczos_mi = 20 
    r_vals_mi, I_r_vals = calculate_mutual_information_I_r(psi_gs, L, r_max_mi, num_lanczos_mi)
    results['r_values_mi'] = r_vals_mi; results['I_r_values'] = I_r_vals
    d_E_r_vals = calculate_entanglement_distance_d_E_r(r_vals_mi, I_r_vals)
    results['d_E_r_values'] = d_E_r_vals
    actual_r_max_fit_mi_val = 0
    if r_vals_mi: 
        idx_limit = min(int(len(r_vals_mi) * fit_r_max_factor), len(r_vals_mi) - 1)
        if idx_limit >=0: actual_r_max_fit_mi_val = r_vals_mi[idx_limit]
        elif r_vals_mi: actual_r_max_fit_mi_val = r_vals_mi[0]
    actual_r_max_fit_mi_val = max(actual_r_max_fit_mi_val, fit_r_min)
    fit_indices = [idx for idx, r_val in enumerate(r_vals_mi) if fit_r_min <= r_val <= actual_r_max_fit_mi_val]
    if not fit_indices: 
        beta, beta_err, R2_I, k, k_err, R2_dE = (np.nan,)*6
    else:
        r_for_fit_beta = [r_vals_mi[i] for i in fit_indices]
        I_for_fit_beta = [I_r_vals[i] for i in fit_indices]
        d_E_for_fit_k = [d_E_r_vals[i] for i in fit_indices]
        beta, beta_err, R2_I = fit_beta_from_I_r(r_for_fit_beta, I_for_fit_beta)
        k, k_err, R2_dE = fit_k_from_d_E_r(r_for_fit_beta, d_E_for_fit_k)
    results.update({'beta': beta, 'beta_err': beta_err, 'R2_I': R2_I, 'k': k, 'k_err': k_err, 'R2_dE': R2_dE})
    c, c_err = calculate_central_charge(psi_gs, L)
    results.update({'central_charge': c, 'central_charge_err': c_err})
    sz_op = quimb.spin_operator('Z', S=0.5) 
    xi = calculate_correlation_length(psi_gs, L, r_max_corr, sz_op)
    results['correlation_length_xi'] = xi
    results['max_bond_entropy'] = get_max_bond_entropy(psi_gs)
    return results

if __name__ == '__main__':
    print("--- Testing mi_tools.py (Corrected v2) ---")
    L_test = 8 # Small L for quick test, but large enough for some structure
    bond_dim_test = 4 
    print(f"Creating a random MPS for L={L_test}, bond_dim={bond_dim_test}...")
    try:
        psi_test = qtn.MPS_rand_state(L_test, bond_dim_test, phys_dim=2, normalize=True)
        psi_test.left_canonise() # Corrected method name
        print("Random MPS created and canonized.")
        r_max_mi_test = L_test // 2 
        r_max_corr_test = L_test // 2
        print(f"Extracting properties with r_max_mi={r_max_mi_test}, r_max_corr={r_max_corr_test}...")
        print("\nTesting _calculate_subsystem_entropy...")
        S_0 = _calculate_subsystem_entropy(psi_test, [0])
        print(f"  S(site 0): {S_0}")
        if L_test > 1: S_01 = _calculate_subsystem_entropy(psi_test, [0, 1]); print(f"  S(sites 0,1): {S_01}")
        if L_test > 1: S_0_L_minus_1 = _calculate_subsystem_entropy(psi_test, [0, L_test-1]); print(f"  S(sites 0, {L_test-1}): {S_0_L_minus_1}")
        print("\nTesting extract_all_properties...")
        all_props = extract_all_properties(psi_test, L_test, r_max_mi_test, r_max_corr_test, fit_r_min=1, fit_r_max_factor=1.0)
        print("\n--- Extracted Properties (Test MPS) ---")
        for key, val in all_props.items():
            if isinstance(val, (list, np.ndarray)) and val is not None and hasattr(val, '__len__'): 
                if len(val) > 5: print(f"  {key}: Array of length {len(val)}, first 5: {val[:5]}")
                else: print(f"  {key}: {val}")
            elif isinstance(val, float): print(f"  {key}: {val:.4f}")
            else: print(f"  {key}: {val}")
        print("\nBasic tests for mi_tools.py finished.")
    except ImportError: print("Quimb library not found.")
    except Exception as e: print(f"An error: {e}"); import traceback; traceback.print_exc()
