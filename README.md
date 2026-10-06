# BB84 Quantum Key Distribution: end-to-end simulation with Qiskit

IBM Qiskit Fall Fest 2026, Track 7: Information-Theoretic Cybersecurity via BB84 QKD.

Author: Sandeep S (Team Lead)

## Problem statement

Simulate the complete BB84 protocol between Arjun and Bhavna in the presence of an eavesdropper
(Esha) and a noisy channel, extract a verified secret key with classical post-processing, measure
how much information leaks, and compare QKD with NIST post-quantum key exchange (ML-KEM / Kyber).

## Circuit design and methodology

Every round is a two-qubit Qiskit circuit run on the `StatevectorSampler` primitive
(`q0` = signal qubit, `q1` = Esha's ancilla).

| Stage | Implementation |
|---|---|
| State preparation | Arjun applies `X` if bit = 1 and `H` if basis = X, giving \|0>, \|1>, \|+>, \|-> |
| Channel noise | Depolarising noise (random X, Y or Z gate), photon loss, detector bit-flips |
| Intercept-resend attack | Esha measures in a random basis and re-prepares the state she saw |
| Entangling probe attack | Esha couples an ancilla with `CNOT` (or a weaker `CRY`) and reads it in Z |
| Measurement and sifting | Bhavna measures in a random basis; rounds with matching bases are kept |
| Parameter estimation | QBER on a sacrificed random sample; abort above 11 % (Shor-Preskill) |
| Error correction | Cascade: block parities, binary search and back-tracking; leak is counted |
| Privacy amplification | Toeplitz-matrix hashing to the final key length |
| Metrics | QBER, R = 1 - 2 H2(QBER), I(A;B), I(A;E), Delta I = I(A;B) - I(A;E) |

Rounds with identical settings share one circuit, so 4096 rounds need at most 64 distinct circuits.

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

```
python3 bb84_qkd.py                                # 2 % noisy channel, no Esha
python3 bb84_qkd.py --attack intercept --esha 1.0   # full intercept-resend
python3 bb84_qkd.py --attack probe --esha 0.3       # CNOT ancilla probe on 30 % of qubits
python3 bb84_qkd.py --noise 0.05 --loss 0.3 --readout 0.01
```

Outputs are written to `results_bb84/`: `bb84_sweeps.csv`, `bb84_analysis.png`, `pqc_comparison.csv`.

## Key results

| Scenario (4096 qubits) | QBER | Delta I | Outcome |
|---|---|---|---|
| 2 % depolarising noise, no Esha | 1.2 % (sample) | +0.89 | 1127-bit secret key, Arjun and Bhavna match |
| Full intercept-resend | 24.8 % (sifted key) | -0.33 | Abort, no key |
| Esha attacks 20 % of qubits | about 4.5 % | +0.64 | Key of about 520 to 550 bits |

- Both attacks follow the theoretical QBER of f/4 for an attacked fraction f.
- The 11 % abort triggers at about f = 0.44; Delta I turns negative between f = 0.6 and 0.8.
- Without Esha the QBER follows 2p/3 for depolarising probability p.

![BB84 analysis](results_bb84/bb84_analysis.png)

## Post-quantum comparison

| | BB84 QKD | ML-KEM-768 (NIST FIPS 203) |
|---|---|---|
| Security rests on | Quantum mechanics (information-theoretic) | Hardness of module-LWE lattice problems |
| Hardware | Single-photon source and detectors, dedicated link | Any CPU |
| Key material | 0.275 secret bits per qubit sent in the 2 % noise run | 256-bit secret from a 1184-byte key and 1088-byte ciphertext |
| Network scaling | Point-to-point; trusted nodes or repeaters at long range | Any IP network |

ML-KEM is not executed or benchmarked in this repository; its figures are the published parameter sizes.

## Limitations

- The classical channel is assumed to be authenticated; no authentication step is implemented.
- Error correction uses Cascade only (no Winnow).
- Ideal single-photon source: no multi-photon pulses or decoy states.

## References

### Scientific papers and standards

1. C. H. Bennett and G. Brassard, "Quantum cryptography: public key distribution and coin tossing", Proc. IEEE Int. Conf. on Computers, Systems and Signal Processing, Bangalore, pp. 175-179 (1984).
2. P. W. Shor and J. Preskill, "Simple proof of security of the BB84 quantum key distribution protocol", Phys. Rev. Lett. 85, 441 (2000).
3. G. Brassard and L. Salvail, "Secret-key reconciliation by public discussion", EUROCRYPT '93, LNCS 765, pp. 410-423 (1994). [Cascade]
4. C. H. Bennett, G. Brassard, C. Crepeau and U. M. Maurer, "Generalized privacy amplification", IEEE Trans. Inf. Theory 41, 1915 (1995).
5. H. Krawczyk, "LFSR-based hashing and authentication", CRYPTO '94, LNCS 839, pp. 129-139 (1994). [Toeplitz hashing]
6. V. Scarani et al., "The security of practical quantum key distribution", Rev. Mod. Phys. 81, 1301 (2009).
7. P. W. Shor, "Polynomial-time algorithms for prime factorization and discrete logarithms on a quantum computer", SIAM J. Comput. 26, 1484 (1997).
8. NIST, "Module-Lattice-Based Key-Encapsulation Mechanism Standard", FIPS 203 (2024), doi:10.6028/NIST.FIPS.203.

### Software libraries

- Qiskit: A. Javadi-Abhari et al., "Quantum computing with Qiskit", arXiv:2405.08810 (2024). Apache 2.0.
- NumPy: C. R. Harris et al., Nature 585, 357 (2020).
- Matplotlib: J. D. Hunter, Comput. Sci. Eng. 9, 90 (2007).

### Templates and acknowledgements

No external code template or notebook was copied. The code and slides were written for this submission with AI assistance (Claude, Anthropic).
