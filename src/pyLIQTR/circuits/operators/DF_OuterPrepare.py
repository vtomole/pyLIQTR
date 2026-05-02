"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
from numpy.typing import NDArray
from typing import Sequence, Tuple

import cirq
import numpy as np
from qualtran import DecomposeNotImplementedError
from qualtran.bloqs.state_preparation import StatePreparationAliasSampling
from qualtran.bloqs.arithmetic import LessThanEqual
from qualtran.bloqs.basic_gates.swap import CSwap
from qualtran.bloqs.data_loading.qrom import QROM
from functools import cached_property
from qualtran._infra.data_types import BQUInt
from qualtran._infra.registers import Register, Signature
from qualtran.linalg.lcu_util import preprocess_probabilities_for_reversible_sampling

from pyLIQTR.circuits.operators.FlaggedPrepareUniformSuperposition import FlaggedPrepareUniformSuperposition

class OuterPrepare(StatePreparationAliasSampling):
    '''
    Implements first block [prep] in Appendix C Fig 16 from https://arxiv.org/abs/2011.03494
    (Step 1 pg 51-52)
    '''
    def build_composite_bloq(self, bb, **soqs):
        raise DecomposeNotImplementedError(f"{self} uses its Cirq decomposition.")

    @classmethod
    def from_lcu_probs(
        cls, lcu_probabilities: Sequence[float], probability_epsilon: float = 1.0e-5
    ) -> "OuterPrepare":
        if not all(x >= 0 for x in lcu_probabilities):
            raise ValueError(f"{cls} expects only non-negative probabilities")

        alt, keep, mu = preprocess_probabilities_for_reversible_sampling(
            unnormalized_probabilities=lcu_probabilities,
            epsilon=probability_epsilon,
        )
        N = len(lcu_probabilities)
        return cls(
            selection_registers=Register("selection", BQUInt((N - 1).bit_length(), N)),
            alt=np.array(alt),
            keep=np.array(keep),
            mu=mu,
            sum_of_unnormalized_probabilities=sum(lcu_probabilities),
        )

    @classmethod
    def from_probabilities(
        cls, unnormalized_probabilities: Sequence[float], *, precision: float = 1.0e-5
    ) -> "OuterPrepare":
        if not all(x >= 0 for x in unnormalized_probabilities):
            raise ValueError(f"{cls} expects only non-negative probabilities")

        qlambda = sum(unnormalized_probabilities)
        alt, keep, mu = preprocess_probabilities_for_reversible_sampling(
            unnormalized_probabilities=unnormalized_probabilities,
            epsilon=precision / qlambda,
        )
        N = len(unnormalized_probabilities)
        return cls(
            selection_registers=Register("selection", BQUInt((N - 1).bit_length(), N)),
            alt=np.array(alt),
            keep=np.array(keep),
            mu=mu,
            sum_of_unnormalized_probabilities=qlambda,
        )

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        return tuple(
            Signature.build(
                sigma_mu=self.sigma_mu_bitsize,
                alt=self.alternates_bitsize,
                keep=self.keep_bitsize,
                rot_ancilla=1,
                less_than_equal=1,
                success = 1
            )
        )
     
    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:

        selection, less_than_equal = quregs['selection'], quregs['less_than_equal']
        sigma_mu, alt, keep = quregs.get('sigma_mu', ()), quregs['alt'], quregs.get('keep', ())
        success, rot_ancilla = quregs['success'], quregs['rot_ancilla']
        N = self.selection_registers[0].dtype.iteration_length

        yield FlaggedPrepareUniformSuperposition(N).on_registers(target=selection,success=success,less_than_ancilla=less_than_equal,rot_ancilla=rot_ancilla)

        yield cirq.H.on_each(*sigma_mu)

        qrom_gate = QROM(
            [self.alt, self.keep],
            (self.selection_bitsize,),
            (self.alternates_bitsize, self.keep_bitsize),
        )
        yield qrom_gate.on_registers(selection=selection, target0_=alt, target1_=keep)

        yield LessThanEqual(self.mu, self.mu).on(
            *keep, *sigma_mu, *less_than_equal
        )
        yield CSwap.make_on(
            ctrl=less_than_equal, x=alt, y=selection
        )

        # uncompute less than equal
        yield LessThanEqual(self.mu, self.mu).on(
            *keep, *sigma_mu, *less_than_equal
        )
