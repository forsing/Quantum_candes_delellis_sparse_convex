# Quantum_candes_delellis_sparse_convex 



import random
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator, Statevector

random.seed(39)
np.random.seed(39)



FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90_loto_2981.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90_loto_plus_1737.csv"),
)



NUM_QUBITS = 6
STATE_DIM = 2 ** NUM_QUBITS
MAX_NUM = 39
NUM_POSITIONS = 7
WINDOW_LEN = 5
LAMBDA_L1_GRID = (0.015, 0.035, 0.065)
OSC_AMP_GRID = (0.15, 0.35)


def load_draws(file_path: Path) -> np.ndarray:
    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean:
                continue
            parts = [p.strip() for p in clean.split(",") if p.strip()]
            nums = [int(x) for x in parts[:NUM_POSITIONS]]
            if len(nums) != NUM_POSITIONS:
                raise ValueError(f"Neispravan broj elemenata u redu: {clean}")
            if len(set(nums)) != NUM_POSITIONS or any(n < 1 or n > MAX_NUM for n in nums):
                raise ValueError(f"Neispravni brojevi u redu: {clean}")
            rows.append(sorted(nums))
    data = np.array(rows, dtype=np.int64)
    if len(data) < WINDOW_LEN + 35:
        raise ValueError(f"Nedovoljno istorijskih kola u fajlu: {file_path}")
    return data


def build_candes_rip_sensing_matrix() -> np.ndarray:
    qc = QuantumCircuit(NUM_QUBITS)
    for q in range(NUM_QUBITS):
        qc.h(q)
        qc.rz(np.pi * float(q + 1) / float(NUM_QUBITS + 1), q)
    for q in range(NUM_QUBITS - 1):
        qc.cx(q, q + 1)
    qc.append(QFTGate(NUM_QUBITS), list(range(NUM_QUBITS)))

    u_full = Operator(qc).data
    phi = u_full[:, :MAX_NUM]
    col_norms = np.linalg.norm(phi, axis=0, keepdims=True)
    return phi / np.maximum(col_norms, 1e-12)


def draws_to_sparse_indicators(draws: np.ndarray) -> np.ndarray:
    n_total = len(draws)
    sparse_mat = np.zeros((n_total, MAX_NUM), dtype=np.float64)
    row_idx = np.arange(n_total)[:, None]
    sparse_mat[row_idx, draws - 1] = 1.0
    return sparse_mat


def fit_convex_integration_wave_operator(
    sparse_train: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_train = len(sparse_train)
    n_pairs = n_train - WINDOW_LEN
    meas_train = sparse_train @ phi.T

    feat_dim = WINDOW_LEN * STATE_DIM
    x_mat = np.empty((n_pairs, feat_dim), dtype=np.complex128)
    y_mat = np.empty((n_pairs, STATE_DIM), dtype=np.complex128)

    for idx in range(n_pairs):
        t = idx + WINDOW_LEN
        x_mat[idx] = meas_train[t - WINDOW_LEN : t].reshape(-1)
        y_mat[idx] = meas_train.__getitem__(t)

    gram = np.conj(x_mat.T) @ x_mat + 2.0 * np.eye(feat_dim, dtype=np.complex128)
    rhs = np.conj(x_mat.T) @ y_mat
    w_op = np.linalg.solve(gram, rhs)

    diffs = sparse_train[1:] - sparse_train[:-1]
    osc_vel = np.mean(np.abs(diffs), axis=0)
    osc_vel = osc_vel / max(float(np.max(osc_vel)), 1e-12)
    return w_op, osc_vel


def convex_integration_batch_measurements(
    sparse_all: np.ndarray,
    start_t: int,
    end_t: int,
    phi: np.ndarray,
    w_op: np.ndarray,
    osc_vel: np.ndarray,
    osc_amp: float,
) -> np.ndarray:
    n_batch = end_t - start_t
    meas_all = sparse_all @ phi.T

    x_win = np.empty((n_batch, WINDOW_LEN * STATE_DIM), dtype=np.complex128)
    last_vecs = np.empty((n_batch, MAX_NUM), dtype=np.float64)
    prev_vecs = np.empty((n_batch, MAX_NUM), dtype=np.float64)

    for offset in range(n_batch):
        t = start_t + offset
        x_win[offset] = meas_all[t - WINDOW_LEN : t].reshape(-1)
        last_vecs[offset] = sparse_all.__getitem__(t - 1)
        prev_vecs[offset] = sparse_all.__getitem__(t - 2)

    y_pred_meas = x_win @ w_op

    grad_dir = last_vecs - prev_vecs
    freq_grid = np.arange(1, MAX_NUM + 1, dtype=np.float64)[None, :]
    phase_shifts = np.sum(last_vecs * freq_grid, axis=1, keepdims=True) * np.pi / float(MAX_NUM)
    high_freq_osc = np.cos(3.0 * np.pi * freq_grid / float(MAX_NUM) + phase_shifts)
    de_lellis_pert = osc_amp * osc_vel[None, :] * high_freq_osc * (1.0 - 0.5 * np.abs(grad_dir))

    return y_pred_meas + (de_lellis_pert.astype(np.complex128) @ phi.T)






def ista_l1_batch_recovery(
    y_meas_batch: np.ndarray,
    phi: np.ndarray,
    lambda_l1: float,
) -> np.ndarray:
    n_batch = int(y_meas_batch.shape.__getitem__(0))
    y_proj = np.real(y_meas_batch @ np.conj(phi))
    x_hat = np.maximum(y_proj - lambda_l1, 0.0)

    masses = np.sum(x_hat, axis=1, keepdims=True)
    uniform_x = np.full((n_batch, MAX_NUM), float(NUM_POSITIONS) / float(MAX_NUM), dtype=np.float64)
    return np.where(masses > 1e-12, x_hat * (float(NUM_POSITIONS) / np.maximum(masses, 1e-12)), uniform_x)






def evaluate_sparse_nll(
    sparse_all: np.ndarray,
    start_t: int,
    end_t: int,
    phi: np.ndarray,
    w_op: np.ndarray,
    osc_vel: np.ndarray,
    lambda_l1: float,
    osc_amp: float,
) -> float:
    y_meas_batch = convex_integration_batch_measurements(
        sparse_all, start_t, end_t, phi, w_op, osc_vel, osc_amp
    )
    x_rec_batch = ista_l1_batch_recovery(y_meas_batch, phi, lambda_l1=lambda_l1)

    prob_mat = 0.85 * (x_rec_batch / float(NUM_POSITIONS)) + 0.15 * (1.0 / float(MAX_NUM))
    prob_mat = prob_mat / np.sum(prob_mat, axis=1, keepdims=True)

    targets = sparse_all[start_t:end_t] > 0.5
    log_probs = np.log(np.maximum(prob_mat, 1e-12))
    nll_per_draw = -np.sum(np.where(targets, log_probs, 0.0), axis=1)
    return float(np.mean(nll_per_draw))


def decode_top_k_sparse_support(x_rec: np.ndarray) -> list[int]:
    top_indices = np.argsort(x_rec)[::-1][:NUM_POSITIONS]
    combo = sorted([int(idx) + 1 for idx in top_indices])
    if len(set(combo)) != NUM_POSITIONS:
        raise RuntimeError("Neuspesna l1 sparse rekonstrukcija 7 razlicitih brojeva.")
    return combo


def main() -> None:
    print("Metoda: Candes Compressed Sensing (l1 ISTA) & De Lellis Convex Integration (1x6 qubita)")
    print("Posebnost: Kvantna RIP merna matrica 64x39 + visokofrekventne oscilacije i l1 Basis Pursuit")

    phi = build_candes_rip_sensing_matrix()

    qc_verify = QuantumCircuit(NUM_QUBITS)
    qc_verify.append(QFTGate(NUM_QUBITS), list(range(NUM_QUBITS)))
    init_sv = Statevector.from_int(0, dims=STATE_DIM)
    _ = init_sv.evolve(qc_verify)

    for file_path in FILES:
        if not file_path.exists():
            print(f"Fajl nije pronadjen: {file_path}")
            continue

        draws = load_draws(file_path)
        sparse_all = draws_to_sparse_indicators(draws)
        n_total = len(draws)

        train_end = WINDOW_LEN + int(0.75 * (n_total - WINDOW_LEN))
        val_end = WINDOW_LEN + int(0.88 * (n_total - WINDOW_LEN))

        w_tr, osc_tr = fit_convex_integration_wave_operator(sparse_all[:train_end], phi)

        best_lam = float(LAMBDA_L1_GRID.__getitem__(0))
        best_osc = float(OSC_AMP_GRID.__getitem__(0))
        best_val_nll = float("inf")

        for lam in LAMBDA_L1_GRID:
            for osc in OSC_AMP_GRID:
                val_nll = evaluate_sparse_nll(
                    sparse_all,
                    start_t=train_end,
                    end_t=val_end,
                    phi=phi,
                    w_op=w_tr,
                    osc_vel=osc_tr,
                    lambda_l1=lam,
                    osc_amp=osc,
                )
                if val_nll < best_val_nll:
                    best_val_nll = val_nll
                    best_lam = lam
                    best_osc = osc

        test_nll = evaluate_sparse_nll(
            sparse_all,
            start_t=val_end,
            end_t=n_total,
            phi=phi,
            w_op=w_tr,
            osc_vel=osc_tr,
            lambda_l1=best_lam,
            osc_amp=best_osc,
        )

        w_full, osc_full = fit_convex_integration_wave_operator(sparse_all, phi)
        extended_sparse = np.vstack([sparse_all, np.zeros((1, MAX_NUM), dtype=np.float64)])
        latest_meas = convex_integration_batch_measurements(
            extended_sparse,
            start_t=n_total,
            end_t=n_total + 1,
            phi=phi,
            w_op=w_full,
            osc_vel=osc_full,
            osc_amp=best_osc,
        )
        next_sparse_x = ista_l1_batch_recovery(latest_meas, phi, lambda_l1=best_lam).__getitem__(0)
        next_combo = decode_top_k_sparse_support(next_sparse_x)

        print(f"CSV: {file_path.name}")
        print(f"Validacija NLL (lambda_l1={best_lam}, osc_amp={best_osc}): {best_val_nll:.4f}")
        print(f"Test NLL: {test_nll:.4f}")
        print(f"NEXT: {next_combo}")


if __name__ == "__main__":
    main()



"""
Metoda: Candes Compressed Sensing (l1 ISTA) & De Lellis Convex Integration (1x6 qubita)
Posebnost: Kvantna RIP merna matrica 64x39 + visokofrekventne oscilacije i l1 Basis Pursuit

CSV: loto7_4718_k90.csv
Validacija NLL (lambda_l1=0.015, osc_amp=0.15): 27.2823
Test NLL: 27.4350
NEXT: [11, 12, 16, 19, 37, 38, 39]

CSV: loto7_4718_k90_loto_2981.csv
Validacija NLL (lambda_l1=0.015, osc_amp=0.15): 27.6345
Test NLL: 27.8334
NEXT: [1, 4, 5, 6, 7, 28, 29]

CSV: loto7_4718_k90_loto_plus_1737.csv
Validacija NLL (lambda_l1=0.015, osc_amp=0.15): 27.8808
Test NLL: 27.9134
NEXT: [8, 11, 15, 16, 17, 20, 24]
"""



"""
2026 winners are: 
Ken'ichi Nomoto and Stanford Woosley in astronomy; 
Anne Dejean, Hugues de Thé and Zhu Chen in life science and medicine; 
and Emmanuel Candès and Camillo De Lellis in mathematical sciences.


Dobitnici Shaw Prize 2026 — svaku radim kao zaseban, potpuno drugačiji matematičko-kvantni mehanizam na 6 qubita.

1. Oblast 1 (Mathematical Sciences — Emmanuel Candès & Camillo De Lellis):
	• Metoda: Compressed Sensing (\(\ell _{1}\) Sparse Basis Pursuit) & Convex Integration Oscillatory Perturbations na 1x6 qubita.
	
		• Svako Loto kolo od 7 brojeva u prostoru 1-39 je 7-sparse signal (ima tačno 7 nenultih komponenti od 39).
		• Kvantno kolo na 6 qubita (QFTGate + nekomutativne fazne rotacije) formira mernu matricu \(\Phi \in \mathbb{C}^{64 \times 39}\) (Restricted Isometry Property — Candès).
		• Rekonstrukcija sledećeg stanja vrši se ortogonalnom iterativnom \(\ell _{1}\) projekcijom (ISTA / Basis Pursuit Denoising) u kombinaciji sa De Lellis-ovim visoko-frekventnim konveksnim oscilacijama na aktivnom talasnom frontu poslednjih kola.
		• NEXT kombinacija dobija se direktnim \(\ell _{1}\)-sparse izborom 7 najjačih amplituda rekonstruisanog retkog vektora \(\hat{x} \in \mathbb{R}^{39}\) (bez ikakve pozicione matrice 7×39, bez pos_bias i bez DP lanca!).

2. Oblast 2 (Astronomy — Ken'ichi Nomoto & Stanford Woosley):
	• Metoda: Core-Collapse Supernova Nucleosynthesis & Stellar Shell Reaction Network na 1x6 qubita.
	
        • Kvantni evolucioni operator nuklearnih reakcija između ljuski (He → C → Ne → O → Si → Fe) u 39 izotopskih kanala i kolaps jezgra sa udarnim talasom (shock-revival).

3. Oblast 3 (Life Science & Medicine — Anne Dejean, Hugues de Thé & Zhu Chen):
	• Metoda: PML-RARA Oncoprotein Degradation & Epigenetic Chromatin Remodeling Dynamical System.
	
        • Bistabilni nelinearni dinamički sistem diferencijacije i ciljane degradacije kompleksa na 6-qubitnom simulatoru.
"""
