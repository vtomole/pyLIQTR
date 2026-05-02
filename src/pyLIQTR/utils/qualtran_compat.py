"""
Copyright (c) 2024 Massachusetts Institute of Technology
SPDX-License-Identifier: BSD-2-Clause
"""

import cirq

from qualtran import Bloq
from qualtran.cirq_interop import CirqGateAsBloq
from qualtran.bloqs.basic_gates import Hadamard, SGate, TGate, XGate, YGate, ZGate
from qualtran.bloqs.mcmt.multi_control_pauli import MultiControlPauli as _MultiControlPauli


def _as_tuple_cvs(cvs):
    if isinstance(cvs, tuple):
        return cvs
    if isinstance(cvs, list):
        return tuple(cvs)
    return (cvs,)


def _as_bloq(target_gate) -> Bloq:
    if isinstance(target_gate, Bloq):
        return target_gate
    if isinstance(target_gate, cirq.Gate):
        cirq_to_bloq = {
            cirq.X: XGate(),
            cirq.Y: YGate(),
            cirq.Z: ZGate(),
            cirq.H: Hadamard(),
            cirq.S: SGate(),
            cirq.T: TGate(),
        }
        if target_gate in cirq_to_bloq:
            return cirq_to_bloq[target_gate]
        return CirqGateAsBloq(target_gate)
    raise TypeError(f"Unsupported target gate for MultiControlPauli: {target_gate!r}")


def MultiControlPauli(*cv_args, cvs=None, target_gate=None, target_bloq=None):
    """Compatibility wrapper for Qualtran 0.7.0's target_bloq-based constructor."""
    if cvs is None:
        if len(cv_args) == 1:
            cvs = cv_args[0]
        else:
            cvs = cv_args
    elif cv_args:
        raise TypeError("Specify control values either positionally or with cvs=, not both.")

    if target_bloq is None:
        if target_gate is None:
            raise TypeError("target_gate or target_bloq must be specified.")
        target_bloq = _as_bloq(target_gate)

    return _MultiControlPauli(cvs=_as_tuple_cvs(cvs), target_bloq=target_bloq)
