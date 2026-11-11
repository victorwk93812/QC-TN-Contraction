"""Which circuit each plan name in the authors' release was made on.

`candidates(name)` returns the builds to try for a circuit name, most likely first. A plan is matched to the
first candidate on which it replays (`qctn.tn.replay`). Most names have one candidate; where the release's
plans for one name were made on more than one network (RQC with and without the final Hadamard layer, QNN
with the current and the earlier QASM reader), all are listed. Name suffixes `_gpu`, `_fil_<k>`,
`_fils_<k>` (threads) and `_gn` don't change the circuit.

Anomalies found by replay (the plan's name and the circuit it replays on differ):

- `RQC_3_7_50`: all five plans replay on depth 49 (same step count as `RQC_3_7_49`), none on depth 50.
- `RQC_4_6_30`: two plans replay on depth 30, five on depth 32.
- `GROVER_6`: both plans replay on the 5-qubit Grover circuit (`GROVER_5`); the release has no 6-qubit
  v-chain file.
- `dj_indep_qiskit_25`: both plans replay on `pricingcall_nativegates_ibm_qiskit_opt3_25`.
- `RQC_4_20`, `RQC_4_60`: the plans contract only tensors t1-t193, which are gate tensors for any depth >= 17;
  they agree step by step with depth 20 / 60 but don't pin the depth down. `RQC_4_11`'s only plan has one step.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from qctn import builders as B
from qctn.circuits import Circuit
from qctn.paths import authors_circuits


@dataclass
class Candidate:
    family: str
    source: str  # generator (QXTools / QXZoo / inferred) or QASM file name
    params: dict = field(default_factory=dict)
    variant: str = ""  # short label of what distinguishes it from the other candidates
    build: Callable[[], Circuit] = None
    note: str = ""

    def label(self) -> str:
        p = ",".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.source}({p})" + (f"[{self.variant}]" if self.variant else "")


_PARTIAL = {
    "RQC_4_20": "plans contract only t1-t193 (gate tensors, no inputs/outputs); they replay on any depth >= 17",
    "RQC_4_60": "plans contract only t1-t193 (gate tensors, no inputs/outputs); they replay on any depth >= 17",
    "RQC_4_11": "the only plan has one step; it doesn't identify the network",
}
_SUFFIX = re.compile(r"(_fils?_\d+)?(_gpu)?$")


def base_name(name: str) -> str:
    """Strip the run suffixes `_fil_<k>`, `_fils_<k>`, `_gpu`."""
    return _SUFFIX.sub("", name)


def _qasm(family, file, rewrite=None, early=False, variant="", note="", params=None):
    path = authors_circuits() / f"{file}.qasm"
    src = f"{file}.qasm"
    if rewrite is B.replace_registers_grover:
        src += " + replace_qasm_registers_grover"
    return Candidate(family, src, params or {}, variant or ("earlier reader" if early else ""),
                     lambda: B.from_qasm(path, rewrite, early=early), note)


def _rqc(r, c, d, h, variant="", note=""):
    return Candidate("RQC", "QXZoo create_RQC", dict(rows=r, cols=c, depth=d, final_h=h),
                     variant or ("final H" if h else "no final H"),
                     lambda: B.rqc(r, c, d, final_h=h), note)


def candidates(name: str) -> list[Candidate]:
    n = base_name(name)
    if m := re.fullmatch(r"GHZ_(\d+)", n):
        k = int(m[1])
        return [Candidate("GHZ", "QXTools create_ghz_circuit", dict(n=k), build=lambda: B.ghz(k))]
    if m := re.fullmatch(r"(?:QFT|qft)_(\d+)", n):
        k = int(m[1])
        return [Candidate("QFT", "QXTools create_qft_circuit", dict(n=k), build=lambda: B.qft(k),
                          note="qubits 1..n (QXTools numbers them 0..n-1)")]
    if m := re.fullmatch(r"QFT_IQFT_(\d+)", n):
        k = int(m[1])
        return [Candidate("QFT_IQFT", "QXZoo apply_qft!(1:ceil(n/2)) + apply_iqft!(1:n)", dict(n=k),
                          build=lambda: B.qft_iqft(k), note="inferred from the plans")]
    if m := re.fullmatch(r"QFT_qasm_(\d+)", n):
        return [_qasm("QFT_qasm", f"qft_nativegates_ibm_qiskit_opt3_{m[1]}")]
    if m := re.fullmatch(r"QNN_(\d+)", n):
        f = f"qnn_indep_qiskit_{m[1]}"
        return [_qasm("QNN", f), _qasm("QNN", f, early=True)]
    if m := re.fullmatch(r"VQE_(\d+)", n):
        return [_qasm("VQE", f"vqe_nativegates_ibm_qiskit_opt3_{m[1]}")]
    if m := re.fullmatch(r"(GROVER|Grover)_(\d+)", n):
        k = m[2] if m[2] != "6" else "5"
        note = "plans replay on the 5-qubit circuit; no 6-qubit v-chain file" if m[2] == "6" else ""
        return [_qasm("Grover", f"grover-v-chain_nativegates_ibm_qiskit_opt3_{k}", B.replace_registers_grover,
                      note=note)]
    if m := re.fullmatch(r"QWALK_(\d+)", n):
        return [_qasm("QWalk", f"qwalk-noancilla_nativegates_ibm_qiskit_opt3_{m[1]}")]
    if m := re.fullmatch(r"Qwalk_(\d+)(?:_fil_\d+)?", n):
        return [_qasm("QWalk", f"qwalk-noancilla_nativegates_ibm_qiskit_opt3_{m[1]}", early=True)]
    if m := re.fullmatch(r"(?:RANDOM|Random_IBM)_(\d+)(?:_gn)?", n):
        return [_qasm("Random", f"random_nativegates_ibm_qiskit_opt3_{m[1]}")]
    if m := re.fullmatch(r"RQC_(\d+)_(\d+)_(\d+)", n):
        r, c, d = map(int, m.groups())
        if (r, c, d) == (3, 7, 50):
            return [_rqc(3, 7, 49, False, note="plans replay on depth 49"), _rqc(3, 7, 50, False)]
        if (r, c, d) == (4, 6, 30):
            return [_rqc(4, 6, 30, False), _rqc(4, 6, 32, False, note="five plans replay on depth 32")]
        if r == c:  # RQC_<n>_<n>_14 and RQC_11_11_11: with the final Hadamard layer
            return [_rqc(r, c, d, True), _rqc(r, c, d, False)]
        return [_rqc(r, c, d, False), _rqc(r, c, d, True)]
    if m := re.fullmatch(r"RQC_(\d+)_(\d+)(?:_mockpla_\w+)?", n):
        k, d = int(m[1]), int(m[2])
        note = _PARTIAL.get(n, "")
        return [_rqc(k, k, d, True, note=note), _rqc(k, k, d, False, note=note)]
    if n == "groundstate_large_indep_qiskit_14_14":
        return [_qasm("Ground state", "groundstate_large_indep_qiskit_14")]
    if n == "dj_indep_qiskit_25":
        return [_qasm("Pricing call", "pricingcall_nativegates_ibm_qiskit_opt3_25",
                      note="named dj_indep_qiskit_25; plans replay on the pricing-call circuit")]
    if (authors_circuits() / f"{n}.qasm").is_file():
        fam = {"dj": "Deutsch-Jozsa", "portfolioqaoa": "Portfolio QAOA", "pricingcall": "Pricing call"}
        return [_qasm(fam.get(n.split("_")[0], n.split("_")[0]), n)]
    return []
