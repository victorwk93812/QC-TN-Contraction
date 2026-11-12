"""Execute and time pairwise contraction plans (phase 4): `tensors` (leaf values), `executor` (NumPy / CuPy,
dense or QXTns hyperindex form), `timing` (synchronised CUDA-event timing, raw repeats)."""
from qctn.execute.executor import Program, compile_plan, contract_plan, prepare, run, steps_of
from qctn.execute.tensors import diagonal_groups, leaf_arrays, statevector
from qctn.execute.timing import Timing, time_program

__all__ = ["Program", "Timing", "compile_plan", "contract_plan", "diagonal_groups", "leaf_arrays", "prepare",
           "run", "statevector", "steps_of", "time_program"]
