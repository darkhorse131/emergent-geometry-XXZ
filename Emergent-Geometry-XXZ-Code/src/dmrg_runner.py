"""
Runs the DMRG calculation for a given XXZ MPO to find the ground state
and calculate its energy and Hamiltonian variance.
"""
import numpy as np
import quimb.tensor as qtn

# Assuming ham_xxz.py is in the same directory or PYTHONPATH
try:
    from ham_xxz import build_xxz_mpo
except ImportError:
    pass


def run_dmrg(H_mpo, L, bond_dims_schedule, cutoffs_schedule, n_sweeps_brief, verbosity=1):
    """
    Performs a single‐site DMRG (DMRG1) calculation to find the ground state.

    Returns:
        energy_gs (float), psi_gs (MPS), hamiltonian_variance (float), dmrg_converged_status (bool)
    """
    print(f"\n--- Starting DMRG for L={L} ---")
    print(f"Bond dimension schedule: {bond_dims_schedule}")
    print(f"Cutoff schedule:        {cutoffs_schedule}")
    print(f"Max sweeps:             {n_sweeps_brief}")

    if not isinstance(H_mpo, qtn.MatrixProductOperator):
        raise TypeError("H_mpo must be a quimb.tensor.MatrixProductOperator.")
    if L != H_mpo.L:
        raise ValueError(f"L ({L}) must match H_mpo.L ({H_mpo.L}).")

    dmrg_engine = qtn.DMRG1(H_mpo,
                            bond_dims=bond_dims_schedule,
                            cutoffs=cutoffs_schedule)

    print(f"Running DMRG solve with tol=1e-7, max_sweeps={n_sweeps_brief}...")
    try:
        dmrg_converged_status = dmrg_engine.solve(
            tol=1e-7,
            max_sweeps=n_sweeps_brief,
            verbosity=verbosity
        )
    except Exception as e:
        print(f"DMRG solve error: {e}")
        import traceback; traceback.print_exc()
        return None, None, np.nan, False

    energy_gs_raw = dmrg_engine.energy
    psi_gs        = dmrg_engine.state
    energy_gs     = float(np.real(energy_gs_raw))

    print(f"DMRG finished. Converged: {dmrg_converged_status}, "
          f"Energy = {energy_gs_raw} (real: {energy_gs:.8f})")

    # --- Variance calculation ---
    print("Calculating Hamiltonian variance...")
    hamiltonian_variance = np.nan
    try:
        print("  Building phi = H |psi> ...")
        # pick a max bond for phi
        max_bond_phi = psi_gs.max_bond() * H_mpo.max_bond()
        max_bond_phi = min(max_bond_phi, psi_gs.max_bond() * 4, 1024)

        phi_mps = H_mpo.apply(psi_gs,
                              compress=True,
                              cutoff=1e-10,
                              max_bond=max_bond_phi)
        print(f"  phi MPS created. max_bond(phi) = {phi_mps.max_bond()}")

        # compute the inner product <phi|phi> = <psi|H^2|psi>
        inner = phi_mps.H @ phi_mps

        # inner may be a numpy scalar or a TensorNetwork
        if isinstance(inner, (int, float, complex, np.generic)):
            E2 = inner
        else:
            # contract down to a scalar tensor then .item()
            E2 = inner.contract().item()

        E2 = float(np.real(E2))
        hamiltonian_variance = E2 - energy_gs**2
        print(f"  <H^2> = {E2:.8f}")
        print(f"  Variance = {hamiltonian_variance:.2e}")

    except Exception as e:
        print(f"Variance calc error: {e}")
        import traceback; traceback.print_exc()
        # leave variance = nan

    return energy_gs, psi_gs, hamiltonian_variance, dmrg_converged_status


if __name__ == '__main__':
    # Quick local test
    print("--- Testing dmrg_runner.py (variance fix) ---")
    L_test = 4
    Delta_test = 1.0
    from ham_xxz import build_xxz_mpo  # ensure this import works

    H_test = build_xxz_mpo(L_test, Delta_test)
    print(f"Built MPO: max_bond={H_test.max_bond()}")

    e, psi, var, conv = run_dmrg(
        H_test, L_test,
        bond_dims_schedule=[8, 16],
        cutoffs_schedule=[1e-8, 1e-8],
        n_sweeps_brief=4,
        verbosity=0
    )
    print(f"\nRESULTS: converged={conv}, E={e:.8f}, var={var:.3e}")
    if psi is not None:
        print(f" MPS max bond={psi.max_bond()}, sites={psi.L}")