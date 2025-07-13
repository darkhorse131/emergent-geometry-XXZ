"""
Builds the MPO for the 1D spin-1/2 XXZ model.

H = sum_i (J_xy * (S^x_i S^x_{i+1} + S^y_i S^y_{i+1}) + J_z * S^z_i S^z_{i+1})
For this project, J_xy = 1.0 and J_z = Delta.
"""
import quimb.tensor as qtn

def build_xxz_mpo(L, Delta, S=0.5, jxy=1.0):
    """
    Constructs the MPO for the 1D XXZ Hamiltonian with open boundary conditions.

    Args:
        L (int): The number of sites in the spin chain.
        Delta (float): The anisotropy parameter (J_z coupling).
                       Delta = 1.0 is the isotropic Heisenberg point (critical).
                       Delta > 1.0 is the Ising-like Neel phase (gapped).
        S (float, optional): The spin value. Defaults to 0.5 for spin-1/2.
        jxy (float, optional): The coupling strength for SxSx and SySy terms.
                               Defaults to 1.0 as per the project brief.

    Returns:
        quimb.tensor.MatrixProductOperator: The MPO representation of the XXZ Hamiltonian.
    """
    if not isinstance(L, int) or L < 2:
        raise ValueError("System size L must be an integer greater than or equal to 2.")
    if not isinstance(Delta, (int, float)):
        raise ValueError("Delta must be a numerical value.")
    if S != 0.5:
        # This project specifically uses spin-1/2
        raise ValueError("This implementation is for S=1/2 (qubits).")

    # Initialize the Hamiltonian builder for 1D spin systems
    # S=0.5 corresponds to spin-1/2 (qubits)
    ham_builder = qtn.SpinHam1D(S=S) # OBC is default

    # Add the types of interactions.
    # SpinHam1D applies these to all nearest-neighbor bonds by default for OBC.
    # The loop "for i in range(L-1)" is not needed here.

    # Add S^x_i S^x_{i+1} term type
    ham_builder.add_term(jxy, 'X', 'X')

    # Add S^y_i S^y_{i+1} term type
    ham_builder.add_term(jxy, 'Y', 'Y')

    # Add Delta * S^z_i S^z_{i+1} term type
    ham_builder.add_term(Delta, 'Z', 'Z')

    # Build the MPO for the specified number of sites L
    # This will apply the defined terms to all L-1 bonds.
    H_mpo = ham_builder.build_mpo(L)

    return H_mpo

if __name__ == '__main__':
    # Example usage and basic test
    L_test = 4
    Delta_critical = 1.0
    Delta_gapped = 2.0

    print(f"--- Testing ham_xxz.py (Corrected) ---")
    try:
        print(f"\nAttempting to build MPO for L={L_test}, Delta={Delta_critical} (critical point)...")
        H_critical_mpo = build_xxz_mpo(L_test, Delta_critical)
        print(f"Successfully built MPO for critical point.")
        print(f"MPO type: {type(H_critical_mpo)}")
        print(f"MPO sites: {H_critical_mpo.L}, Max bond: {H_critical_mpo.max_bond()}")
        # print(H_critical_mpo) # Can be verbose for larger L

        print(f"\nAttempting to build MPO for L={L_test}, Delta={Delta_gapped} (gapped phase)...")
        H_gapped_mpo = build_xxz_mpo(L_test, Delta_gapped)
        print(f"Successfully built MPO for gapped phase.")
        print(f"MPO type: {type(H_gapped_mpo)}")
        # print(H_gapped_mpo)

        # Test with a slightly larger L
        L_larger = 10
        H_larger_mpo = build_xxz_mpo(L_larger, Delta_critical)
        print(f"\nSuccessfully built MPO for L={L_larger}, Delta={Delta_critical}.")
        print(f"MPO sites: {H_larger_mpo.L}, Max bond: {H_larger_mpo.max_bond()}")

        print("\nBasic tests passed for ham_xxz.py (Corrected).")

    except ImportError:
        print("Quimb library not found. Please install it (pip install quimb).")
    except Exception as e:
        print(f"An error occurred during ham_xxz.py test: {e}")
        import traceback
        traceback.print_exc()
