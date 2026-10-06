#!/usr/bin/env python3
"""
BB84 Quantum Key Distribution -- end-to-end simulation and post-processing with Qiskit
======================================================================================

  1. State preparation   : Arjun picks random bits and random bases (Z or X) and prepares
                           each qubit with conditional X and H gates.
  2. Channel + Esha       : depolarising noise, photon loss and detector (readout) errors;
                           Esha can run an intercept-resend attack or an entangling
                           ancilla-probe (CNOT) attack on any fraction of the qubits.
  3. Measurement/sifting : Bhavesh measures in random bases; public basis reconciliation.
  4. Parameter estimation: QBER on a sacrificed sample; abort above 11 % (Shor-Preskill).
  5. Error correction    : Cascade (block parities + binary search), leak is counted.
  6. Privacy amplification: Toeplitz-matrix hashing to the final secret key.
  7. Metrics             : QBER, R = 1 - 2 H2(QBER), I(A;B), I(A;E), Delta I.
  8. PQC comparison      : BB84 against ML-KEM (Kyber) and classical key exchange.

All quantum steps are Qiskit circuits executed with the StatevectorSampler primitive.

Usage
-----
    python bb84_qkd.py                                  # 2 % noisy channel, no Esha
    python bb84_qkd.py --attack intercept --esha 1.0     # full intercept-resend (QBER ~ 25 %)
    python bb84_qkd.py --attack probe --esha 0.3         # CNOT ancilla probe on 30 % of qubits
    python bb84_qkd.py --noise 0.05 --loss 0.3 --readout 0.01
    python bb84_qkd.py --no-sweeps

Outputs (in --outdir, default ./results_bb84): bb84_sweeps.csv, bb84_analysis.png,
pqc_comparison.csv
References
----------
  [1] C. H. Bennett and G. Brassard, 'Quantum cryptography: public key distribution and coin tossing', Proc. IEEE Int. Conf. on Computers, Systems and Signal Processing, Bangalore, pp. 175-179 (1984).
  [2] P. W. Shor and J. Preskill, 'Simple proof of security of the BB84 quantum key distribution protocol', Phys. Rev. Lett. 85, 441 (2000).
  [3] G. Brassard and L. Salvail, 'Secret-key reconciliation by public discussion', EUROCRYPT '93, LNCS 765, pp. 410-423 (1994). [Cascade]
  [4] C. H. Bennett, G. Brassard, C. Crepeau and U. M. Maurer, 'Generalized privacy amplification', IEEE Trans. Inf. Theory 41, 1915 (1995).
  [5] H. Krawczyk, 'LFSR-based hashing and authentication', CRYPTO '94, LNCS 839, pp. 129-139 (1994). [Toeplitz hashing]
  [6] V. Scarani et al., 'The security of practical quantum key distribution', Rev. Mod. Phys. 81, 1301 (2009).
  [7] P. W. Shor, 'Polynomial-time algorithms for prime factorization and discrete logarithms on a quantum computer', SIAM J. Comput. 26, 1484 (1997).
  [8] NIST, 'Module-Lattice-Based Key-Encapsulation Mechanism Standard', FIPS 203 (2024), doi:10.6028/NIST.FIPS.203.
Software
  - Qiskit: A. Javadi-Abhari et al., 'Quantum computing with Qiskit', arXiv:2405.08810 (2024). Apache 2.0.
  - NumPy: C. R. Harris et al., Nature 585, 357 (2020).
  - Matplotlib: J. D. Hunter, Comput. Sci. Eng. 9, 90 (2007).
 
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import time

import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.primitives import StatevectorSampler

QBER_ABORT = 0.11  # Shor-Preskill threshold for BB84
Z_BASIS, X_BASIS = 0, 1


# --------------------------------------------------------------------------------------
# 1. Quantum layer
# --------------------------------------------------------------------------------------
def bb84_round_circuit(bit, basis, probe, err_x, err_z, meas_basis, probe_angle=np.pi):
    """
    One BB84 round. q0 is the travelling signal qubit, q1 is Esha's ancilla probe.

      Arjun  : X if bit == 1, then H if basis == X          -> |0>, |1>, |+>, |->
      Esha    : optional probe, CNOT(signal -> ancilla)      (CRY(angle) for a weaker probe)
      Channel: Pauli error X, Z or Y (depolarising noise)
      Bhavesh    : H if measuring in the X basis, then measure
      Esha    : measures her ancilla in Z (after the bases are announced)
    """
    q = QuantumRegister(2, "q")
    bhavesh, esha = ClassicalRegister(1, "bhavesh"), ClassicalRegister(1, "esha")
    qc = QuantumCircuit(q, bhavesh, esha)
    if bit:
        qc.x(0)
    if basis == X_BASIS:
        qc.h(0)
    qc.barrier()
    if probe:
        if np.isclose(probe_angle, np.pi):
            qc.cx(0, 1)
        else:
            qc.cry(probe_angle, 0, 1)
        qc.barrier()
    if err_x and err_z:
        qc.y(0)
    elif err_x:
        qc.x(0)
    elif err_z:
        qc.z(0)
    if meas_basis == X_BASIS:
        qc.h(0)
    qc.measure(q[0], bhavesh[0])
    qc.measure(q[1], esha[0])
    return qc


class QuantumLink:
    """Runs BB84 rounds on StatevectorSampler. Rounds with identical settings share a circuit."""

    def __init__(self, seed=None, probe_angle=np.pi):
        self.rng = np.random.default_rng(seed)
        # a Generator (not an int) so that every sampler call draws fresh randomness
        self.sampler = StatevectorSampler(seed=np.random.default_rng(self.rng.integers(2**32)))
        self.probe_angle = probe_angle
        self.circuits_run = 0

    def send(self, bits, bases, probe, err_x, err_z, meas_bases):
        """Returns (receiver outcome, ancilla outcome) for every round."""
        settings = np.column_stack([bits, bases, probe, err_x, err_z, meas_bases]).astype(int)
        n = len(settings)
        out_rx, out_anc = np.zeros(n, dtype=np.uint8), np.zeros(n, dtype=np.uint8)
        if n == 0:
            return out_rx, out_anc
        unique, inverse = np.unique(settings, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        counts = np.bincount(inverse, minlength=len(unique))
        pubs = [(bb84_round_circuit(*row, probe_angle=self.probe_angle), None, int(c))
                for row, c in zip(unique, counts)]
        results = self.sampler.run(pubs).result()
        self.circuits_run += len(pubs)
        for g, res in enumerate(results):
            idx = np.flatnonzero(inverse == g)
            rx = np.asarray(res.data.bhavesh.array).reshape(-1) & 1
            anc = np.asarray(res.data.esha.array).reshape(-1) & 1
            order = self.rng.permutation(len(idx))  # shots are i.i.d.; assign them at random
            out_rx[idx], out_anc[idx] = rx[order], anc[order]
        return out_rx, out_anc


def depolarising_errors(n, p, rng):
    """With probability p apply X, Y or Z (p/3 each). Returns X-flag and Z-flag arrays."""
    kind = rng.choice(4, size=n, p=[1 - p, p / 3, p / 3, p / 3])  # 0=I 1=X 2=Y 3=Z
    return ((kind == 1) | (kind == 2)).astype(int), ((kind == 3) | (kind == 2)).astype(int)


def transmit(n, noise, loss, readout, attack, esha_fraction, link, rng):
    """Arjun -> (Esha) -> noisy channel -> Bhavesh. Returns a dict of per-round records."""
    a_bits, a_bases = rng.integers(0, 2, n), rng.integers(0, 2, n)
    b_bases = rng.integers(0, 2, n)
    attacked = (rng.random(n) < esha_fraction) if attack != "none" else np.zeros(n, dtype=bool)
    e_bases = rng.integers(0, 2, n)
    e_bits = np.zeros(n, dtype=np.uint8)
    zeros = np.zeros(n, dtype=int)

    sent_bits, sent_bases = a_bits.copy(), a_bases.copy()
    probe = zeros.copy()
    if attack == "intercept" and attacked.any():
        # Esha measures Arjun's qubit in a random basis, then re-prepares what she saw
        idx = np.flatnonzero(attacked)
        e_bits[idx], _ = link.send(a_bits[idx], a_bases[idx], zeros[idx], zeros[idx], zeros[idx], e_bases[idx])
        sent_bits[idx], sent_bases[idx] = e_bits[idx], e_bases[idx]
    elif attack == "probe":
        probe = attacked.astype(int)
        e_bases[:] = Z_BASIS  # the ancilla is read out in Z

    err_x, err_z = depolarising_errors(n, noise, rng)
    b_bits, anc = link.send(sent_bits, sent_bases, probe, err_x, err_z, b_bases)
    if attack == "probe":
        e_bits = anc

    b_bits = b_bits ^ (rng.random(n) < readout).astype(np.uint8)  # detector bit-flip errors
    detected = rng.random(n) >= loss                                # photon loss
    return dict(a_bits=a_bits, a_bases=a_bases, b_bits=b_bits, b_bases=b_bases,
                e_bits=e_bits, e_bases=e_bases, attacked=attacked, detected=detected)


# --------------------------------------------------------------------------------------
# 2. Information theory
# --------------------------------------------------------------------------------------
def h2(p):
    """Binary Shannon entropy H2(p)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def secret_key_rate(qber):
    """Asymptotic Shor-Preskill rate per sifted bit: R = 1 - 2 H2(QBER)."""
    return np.maximum(0.0, 1.0 - 2.0 * h2(qber))


def empirical_mutual_information(x, y):
    """I(X;Y) in bits from the joint histogram of two binary arrays."""
    if len(x) == 0:
        return 0.0
    joint = np.zeros((2, 2))
    np.add.at(joint, (x.astype(int), y.astype(int)), 1)
    joint /= joint.sum()
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log2(joint[mask] / (px @ py)[mask])))


def esha_information(rec, sifted):
    """
    I(A;E) per sifted bit. After sifting Esha knows Arjun's basis, so her information is
    averaged over the cases she can tell apart (attacked or not, Arjun's basis, her basis).
    """
    a, e = rec["a_bits"][sifted], rec["e_bits"][sifted]
    att, ab, eb = rec["attacked"][sifted], rec["a_bases"][sifted], rec["e_bases"][sifted]
    total, info = len(a), 0.0
    for basis_a in (0, 1):
        for basis_e in (0, 1):
            cls = att & (ab == basis_a) & (eb == basis_e)
            if cls.sum() >= 20:
                info += cls.sum() / total * empirical_mutual_information(a[cls], e[cls])
    return info


# --------------------------------------------------------------------------------------
# 3. Classical post-processing
# --------------------------------------------------------------------------------------
def estimate_qber(a_key, b_key, sample_fraction, rng):
    """QBER = N_error / N_sample on a publicly compared random sample (then discarded)."""
    n = len(a_key)
    n_sample = min(n, max(1, int(round(sample_fraction * n))))
    sample = np.zeros(n, dtype=bool)
    sample[rng.choice(n, size=n_sample, replace=False)] = True
    return float(np.mean(a_key[sample] != b_key[sample])), a_key[~sample], b_key[~sample], n_sample


def cascade(a_key, b_key, qber, rng, passes=6):
    """
    Cascade error correction. Each pass shuffles the key, compares block parities over the
    public channel and binary-searches every block whose parity differs. Correcting a bit
    flips the parity of the blocks that contained it in earlier passes, so those blocks are
    searched again (the "cascade" step). Returns Bhavesh's corrected key and the number of
    parity bits disclosed.
    """
    n = len(a_key)
    bhavesh, leaked = b_key.copy(), 0
    if n < 2:
        return bhavesh, leaked

    def mismatch(idx):
        return (int(a_key[idx].sum()) + int(bhavesh[idx].sum())) % 2 == 1

    def bisect(idx):
        nonlocal leaked
        while len(idx) > 1:
            half = idx[: len(idx) // 2]
            leaked += 1
            idx = half if mismatch(half) else idx[len(idx) // 2:]
        bhavesh[idx[0]] ^= 1
        return idx[0]

    block = int(min(max(4, round(0.73 / max(qber, 0.005))), max(2, n // 2)))
    layers = []  # (blocks, block index of every bit) for each pass so far
    for p in range(passes):
        order = np.arange(n) if p == 0 else rng.permutation(n)
        blocks = [order[s:s + block] for s in range(0, n, block)]
        block_of = np.empty(n, dtype=int)
        for bi, blk in enumerate(blocks):
            block_of[blk] = bi
        layers.append((blocks, block_of))
        leaked += len(blocks)
        queue = [bisect(blk) for blk in blocks if mismatch(blk)]
        while queue:  # cascade back through earlier passes
            i = queue.pop()
            for layer_blocks, layer_of in layers:
                blk = layer_blocks[layer_of[i]]
                if mismatch(blk):
                    queue.append(bisect(blk))
        block = min(block * 2, max(2, n // 2))
    return bhavesh, leaked


def toeplitz_hash(key, out_len, seed_bits):
    """Privacy amplification: multiply by an out_len x n Toeplitz matrix over GF(2)."""
    n = len(key)
    if out_len <= 0 or n == 0:
        return np.zeros(0, dtype=np.uint8)
    i, j = np.arange(out_len)[:, None], np.arange(n)[None, :]
    matrix = seed_bits[i - j + n - 1]  # constant along every diagonal
    return (matrix.astype(np.int64) @ key.astype(np.int64) % 2).astype(np.uint8)


def fingerprint(bits):
    return hashlib.sha256(np.packbits(bits).tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------------------
# 4. Full protocol
# --------------------------------------------------------------------------------------
def run_bb84(n_qubits=4096, noise=0.0, loss=0.0, readout=0.0, attack="none", esha_fraction=0.0,
             sample_fraction=0.25, security_bits=64, probe_angle=np.pi, seed=7, link=None):
    rng = np.random.default_rng(seed)
    link = link or QuantumLink(seed=seed, probe_angle=probe_angle)
    rec = transmit(n_qubits, noise, loss, readout, attack, esha_fraction, link, rng)

    sifted = rec["detected"] & (rec["a_bases"] == rec["b_bases"])  # public basis reconciliation
    a_sift, b_sift = rec["a_bits"][sifted], rec["b_bits"][sifted]
    n_sift = int(sifted.sum())
    qber_true = float(np.mean(a_sift != b_sift)) if n_sift else 0.0
    qber, a_key, b_key, n_sample = estimate_qber(a_sift, b_sift, sample_fraction, rng)

    i_ab = float(1.0 - h2(qber_true))
    i_ae = esha_information(rec, sifted)
    out = {
        "attack": attack, "esha_fraction": esha_fraction, "noise": noise, "loss": loss, "readout": readout,
        "qubits_sent": n_qubits, "detected": int(rec["detected"].sum()), "sifted_bits": n_sift,
        "sample_bits": n_sample, "qber_sample": qber, "qber_sifted": qber_true,
        "I_AB": i_ab, "I_AE": i_ae, "delta_I": i_ab - i_ae,
        "R_asymptotic": float(secret_key_rate(qber)), "abort": qber > QBER_ABORT,
        "ec_leak_bits": 0, "residual_errors": 0, "final_key_bits": 0, "keys_match": False,
        "secret_bits_per_qubit": 0.0,
    }
    if out["abort"]:
        return out, None

    b_corr, leaked = cascade(a_key, b_key, qber, rng)
    out["ec_leak_bits"], out["residual_errors"] = leaked, int(np.sum(a_key != b_corr))

    # final length: remove Esha's share h2(Q) per bit, the Cascade parity leak and a safety margin
    q_eff = max(qber, 1.0 / max(n_sample, 1))
    final_len = int(np.floor(len(a_key) * (1.0 - float(h2(q_eff))) - leaked - security_bits))
    final_len = max(final_len, 0)
    toeplitz_seed = rng.integers(0, 2, size=len(a_key) + final_len, dtype=np.uint8)  # public
    a_final = toeplitz_hash(a_key, final_len, toeplitz_seed)
    b_final = toeplitz_hash(b_corr, final_len, toeplitz_seed)

    out["final_key_bits"] = len(a_final)
    out["keys_match"] = bool(len(a_final) > 0 and np.array_equal(a_final, b_final))
    out["secret_bits_per_qubit"] = len(a_final) / n_qubits
    return out, (a_final, b_final)


# --------------------------------------------------------------------------------------
# 5. PQC comparison
# --------------------------------------------------------------------------------------
PQC_COLUMNS = ["scheme", "type", "security_basis", "public_key_bytes", "ciphertext_bytes",
               "quantum_attack", "hardware", "key_rate", "network_scaling"]
PQC_TABLE = [
    ("RSA-2048", "classical", "integer factoring", 256, 256, "broken by Shor",
     "any CPU", "software, ms per exchange", "any IP network"),
    ("X25519 (ECDH)", "classical", "elliptic-curve discrete log", 32, 32, "broken by Shor",
     "any CPU", "software, well under 1 ms", "any IP network"),
    ("ML-KEM-512", "post-quantum (NIST FIPS 203)", "module-LWE lattices", 800, 768, "none known",
     "any CPU", "software, well under 1 ms", "any IP network"),
    ("ML-KEM-768", "post-quantum (NIST FIPS 203)", "module-LWE lattices", 1184, 1088, "none known",
     "any CPU", "software, well under 1 ms", "any IP network"),
    ("ML-KEM-1024", "post-quantum (NIST FIPS 203)", "module-LWE lattices", 1568, 1568, "none known",
     "any CPU", "software, well under 1 ms", "any IP network"),
    ("BB84 QKD", "quantum", "quantum mechanics (information-theoretic)", 0, 0,
     "secure in principle; needs authenticated classical channel",
     "single-photon source and detectors, dedicated fibre or free-space link",
     "falls exponentially with fibre loss", "point-to-point, trusted nodes or repeaters beyond a few hundred km"),
]


def print_pqc(result):
    print(f"{'scheme':<15}{'security rests on':<44}{'pk B':>6}{'ct B':>6}  quantum attack")
    for r in PQC_TABLE:
        print(f"{r[0]:<15}{r[2]:<44}{r[3]:>6}{r[4]:>6}  {r[5]}")
    print("\nBB84 in this run: "
          f"{result['secret_bits_per_qubit']:.3f} secret bits per qubit sent; one 256-bit key needs about "
          + (f"{int(np.ceil(256 / result['secret_bits_per_qubit']))} qubits." if result["secret_bits_per_qubit"] > 0
             else "no key (protocol aborted)."))
    print("ML-KEM-768 delivers a 256-bit shared secret from one 1184-byte public key and one 1088-byte ciphertext.")


# --------------------------------------------------------------------------------------
# 6. Sweeps and plots
# --------------------------------------------------------------------------------------
def sweeps(n_qubits, seed):
    link, rows = QuantumLink(seed=seed), []
    for i, f in enumerate(np.linspace(0, 1, 6)):
        for attack in ("intercept", "probe"):
            r, _ = run_bb84(n_qubits, attack=attack, esha_fraction=float(f), seed=seed + 10 + i, link=link)
            r.update({"sweep": attack, "x": float(f), "qber_theory": f / 4, "I_AE_theory": f / 2})
            rows.append(r)
    for i, p in enumerate(np.linspace(0, 0.24, 9)):
        r, _ = run_bb84(n_qubits, noise=float(p), seed=seed + 40 + i, link=link)
        r.update({"sweep": "noise", "x": float(p), "qber_theory": 2 * p / 3, "I_AE_theory": 0.0})
        rows.append(r)
    return rows


def make_plot(path, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = lambda name: [r for r in rows if r["sweep"] == name]
    ir, pr, nz = sel("intercept"), sel("probe"), sel("noise")
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.5))

    a = ax[0, 0]
    a.plot([r["x"] for r in ir], [r["qber_theory"] for r in ir], "-", color="black", label="Theory f/4")
    a.plot([r["x"] for r in ir], [r["qber_sifted"] for r in ir], "o", color="tab:blue", label="Intercept-resend")
    a.plot([r["x"] for r in pr], [r["qber_sifted"] for r in pr], "s", mfc="none", color="tab:orange", label="CNOT ancilla probe")
    a.axhline(QBER_ABORT, color="tab:red", ls="--", label="11 % abort threshold")
    a.set(xlabel="Fraction of qubits attacked by Esha", ylabel="QBER", title="Eavesdropping raises the QBER")
    a.legend(frameon=False); a.grid(alpha=0.3)

    a = ax[0, 1]
    a.plot([r["x"] for r in ir], [r["I_AB"] for r in ir], "o-", color="tab:blue", label="I(A;B)")
    a.plot([r["x"] for r in ir], [r["I_AE"] for r in ir], "s-", color="tab:red", label="I(A;E)")
    a.plot([r["x"] for r in ir], [r["delta_I"] for r in ir], "^--", color="tab:green", label="Delta I = I(A;B) - I(A;E)")
    a.axhline(0, color="black", lw=0.8)
    a.set(xlabel="Fraction intercepted (intercept-resend)", ylabel="bits per sifted bit", title="Mutual information advantage")
    a.legend(frameon=False); a.grid(alpha=0.3)

    a = ax[1, 0]
    a.plot([r["x"] for r in nz], [r["qber_theory"] for r in nz], "-", color="black", label="Theory 2p/3")
    a.plot([r["x"] for r in nz], [r["qber_sifted"] for r in nz], "o", color="tab:purple", label="Simulated")
    a.axhline(QBER_ABORT, color="tab:red", ls="--", label="11 % abort threshold")
    a.set(xlabel="Depolarising probability p", ylabel="QBER", title="Channel noise without an eavesdropper")
    a.legend(frameon=False); a.grid(alpha=0.3)

    a = ax[1, 1]
    q = np.linspace(0, 0.15, 200)
    a.plot(q, secret_key_rate(q), "-", color="black", label="Asymptotic R = 1 - 2 H2(QBER)")
    pts = [r for r in rows if r["sifted_bits"] > r["sample_bits"]]
    a.plot([r["qber_sample"] for r in pts],
           [r["final_key_bits"] / (r["sifted_bits"] - r["sample_bits"]) for r in pts],
           "o", color="tab:green", label="Simulated final key (Cascade + Toeplitz)")
    a.axvline(QBER_ABORT, color="tab:red", ls="--")
    a.set(xlabel="QBER", ylabel="Secret bits per sifted bit", title="Secret key rate")
    a.legend(frameon=False); a.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def report(res, keys):
    print(f"qubits sent / detected        : {res['qubits_sent']} / {res['detected']}")
    print(f"sifted key (same basis)       : {res['sifted_bits']} bits")
    print(f"QBER on {res['sample_bits']:>5}-bit sample     : {res['qber_sample']:.4f}   (whole sifted key {res['qber_sifted']:.4f})")
    print(f"I(A;B) = {res['I_AB']:.3f}   I(A;E) = {res['I_AE']:.3f}   Delta I = {res['delta_I']:+.3f} bits per sifted bit")
    print(f"asymptotic rate 1 - 2 H2(Q)   : {res['R_asymptotic']:.3f}")
    if res["abort"]:
        print(f"RESULT                        : ABORT, QBER above {QBER_ABORT:.0%}, no key is produced")
        return
    print(f"Cascade leak                  : {res['ec_leak_bits']} parity bits, residual errors {res['residual_errors']}")
    print(f"final secret key (Toeplitz)   : {res['final_key_bits']} bits = {res['secret_bits_per_qubit']:.3f} per qubit sent")
    print(f"Arjun / Bhavesh key fingerprints  : {fingerprint(keys[0])} / {fingerprint(keys[1])}   match = {res['keys_match']}")


def main():
    ap = argparse.ArgumentParser(description="BB84 QKD end-to-end simulation (Qiskit)")
    ap.add_argument("--qubits", type=int, default=4096)
    ap.add_argument("--noise", type=float, default=0.02, help="depolarising probability")
    ap.add_argument("--loss", type=float, default=0.0, help="photon loss probability")
    ap.add_argument("--readout", type=float, default=0.0, help="detector bit-flip probability")
    ap.add_argument("--attack", choices=["none", "intercept", "probe"], default="none")
    ap.add_argument("--esha", type=float, default=1.0, help="fraction of qubits Esha attacks")
    ap.add_argument("--probe-angle", type=float, default=np.pi, help="pi = CNOT probe, smaller = weaker CRY probe")
    ap.add_argument("--sample", type=float, default=0.25, help="fraction of sifted key sacrificed for QBER")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sweep-qubits", type=int, default=4000)
    ap.add_argument("--no-sweeps", action="store_true")
    ap.add_argument("--outdir", default="results_bb84")
    args, _ = ap.parse_known_args()  # also works inside Jupyter / Colab
    os.makedirs(args.outdir, exist_ok=True)

    print("\n=== Example round: Arjun sends |-> (bit 1, X basis), Esha probes, Bhavesh measures in X ===")
    print(bb84_round_circuit(1, X_BASIS, 1, 0, 0, X_BASIS).draw(output="text"))

    print(f"\n=== Run: {args.qubits} qubits | noise {args.noise} | loss {args.loss} | readout {args.readout} "
          f"| attack {args.attack}" + (f" on {args.esha:.0%}" if args.attack != "none" else "") + " ===")
    t0 = time.time()
    res, keys = run_bb84(args.qubits, args.noise, args.loss, args.readout, args.attack, args.esha,
                         args.sample, probe_angle=args.probe_angle, seed=args.seed)
    report(res, keys)
    print(f"({time.time() - t0:.1f} s)")

    print("\n=== BB84 versus classical and post-quantum key exchange ===")
    print_pqc(res)
    with open(os.path.join(args.outdir, "pqc_comparison.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(PQC_COLUMNS)
        w.writerows(PQC_TABLE)

    if not args.no_sweeps:
        print(f"\n=== Sweeps ({args.sweep_qubits} qubits per point) ===")
        t0 = time.time()
        rows = sweeps(args.sweep_qubits, args.seed)
        print(f"{'sweep':<10}{'x':>6}{'QBER':>8}{'theory':>8}{'I(A;B)':>8}{'I(A;E)':>8}{'dI':>8}{'R':>7}{'abort':>7}{'key':>6}{'match':>7}")
        for r in rows:
            print(f"{r['sweep']:<10}{r['x']:>6.2f}{r['qber_sifted']:>8.3f}{r['qber_theory']:>8.3f}{r['I_AB']:>8.3f}"
                  f"{r['I_AE']:>8.3f}{r['delta_I']:>8.3f}{r['R_asymptotic']:>7.3f}{str(r['abort']):>7}"
                  f"{r['final_key_bits']:>6}{str(r['keys_match']):>7}")
        with open(os.path.join(args.outdir, "bb84_sweeps.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        make_plot(os.path.join(args.outdir, "bb84_analysis.png"), rows)
        print(f"({time.time() - t0:.1f} s)  saved bb84_sweeps.csv and bb84_analysis.png in {args.outdir}/")


if __name__ == "__main__":
    main()
