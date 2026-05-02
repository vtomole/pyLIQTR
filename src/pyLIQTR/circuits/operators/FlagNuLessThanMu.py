"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
from functools import cached_property
from numpy.typing import NDArray
from typing import Set, Optional, Tuple
from qualtran import GateWithRegisters, Signature, QAny, QBit, Register, Side
from qualtran.bloqs.basic_gates import CNOT, Toffoli, XGate
from qualtran.bloqs.mcmt import MultiAnd, And
from qualtran.drawing import WireSymbol
from qualtran.drawing.musical_score import Text

class FlagNuLessThanMu(GateWithRegisters):
    """Gate to flag if all components of :math:`\\nu` are smaller in absolute value than :math:`2^{\mu-2}`. The register :math:`|\\mu\\rangle` should be prepared in a unary-encoded superposition prior to input. This can be done using :class:`PrepareMuUnaryEncoded`. The register :math:`|\\nu\\rangle` should also be prepared in a superposition state using :class:`PrepareNuSuperposition`.

    Registers:

    .. line-block::
        mu: the register containing the state of :math:`\\mu`.
        nu: the register containing the state of :math:`\\nu`
        flag_nu_lt_mu: a flag bit for when all :math:`|\\nu_i| < 2^{\\mu-2}`
        and_ancilla: ancilla qubits used for the AND ladder.

    References:
        `Fault-Tolerant Quantum Simulations of Chemistry in First Quantization <https://arxiv.org/abs/2105.12767>`_
        page 21, Eq 81.

    :param int bitsize: the number of bits in the mu register, :math:`n_p` in the reference.
    """

    def __init__(self,bitsize:int,is_adjoint:bool=False):
        self.bitsize = bitsize
        self.is_adjoint = is_adjoint

    @cached_property
    def signature(self) -> Signature:
        side = Side.LEFT if self.is_adjoint else Side.RIGHT
        return Signature(
            [
                Register("mu", QAny(self.bitsize)),
                Register("nu", QAny(self.bitsize + 1), shape=(3,)),
                Register("flag_nu_lt_mu", QBit(), side=side),
                Register("and_ancilla",QBit(),shape=(self.bitsize,2), side=side)
            ]
        )

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text(r'|ν|<2^(μ−2)')
        return super().wire_symbol(reg, idx)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:

        mu_reg = quregs['mu']
        nu_reg = quregs['nu']
        flag = quregs['flag_nu_lt_mu']
        ancillas = quregs['and_ancilla']
        
        ops = list(self._decompose_forward_from_registers(mu_reg, nu_reg, flag, ancillas))
        if self.is_adjoint:
            yield from (cirq.inverse(op) for op in reversed(ops))
        else:
            yield from ops

    def _decompose_forward_from_registers(self, mu_reg, nu_reg, flag, ancillas):
        # convert mu to one-hot unary
        for bit in range(self.bitsize-1):
            yield CNOT().on(mu_reg[bit+1],mu_reg[bit])

        #for each value of mu, flag (on zero) when all abs(nu_i) < 2^(mu-1) excluding when all abs(nu_i) < 2^(mu-2)
        # ie, if flag=|1>, all abs(nu_i) < 2^(mu-2)
        for i in range(self.bitsize):
            # yield MultiAnd(cvs=(1,0,0,0)).on_registers(ctrl=[[mu_reg[i]],[nu_reg[0][i]],[nu_reg[1][i]],[nu_reg[2][i]]],junk=ancillas[i],target=flag)
            yield And(cv1=1,cv2=0).on_registers(ctrl=[[mu_reg[i]],[nu_reg[0][i]]],target=ancillas[i][0])
            yield And(cv1=1,cv2=0).on_registers(ctrl=[ancillas[i][0],[nu_reg[1][i]]],target=ancillas[i][1])
            
            yield XGate().on(nu_reg[2][i])
            yield Toffoli().on_registers(ctrl=[ancillas[i][1],[nu_reg[2][i]]],target=flag)
            yield XGate().on(nu_reg[2][i])

            # MultiAnd error since And assumes target is in 0 state so it can't be used multiple times, BUT for our case we know only one bit of the mu register should be 1 so should technically be able to use. Using Toffoli until error is resolved.

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs):
        axes = tuple(args.axes)
        nu_start = self.bitsize
        nu_axis = lambda row, col: axes[nu_start + row * (self.bitsize + 1) + col]
        flag_axis = axes[self.bitsize + 3 * (self.bitsize + 1)]
        ancilla_start = self.bitsize + 3 * (self.bitsize + 1) + 1
        ancilla_axis = lambda row, col: axes[ancilla_start + row * 2 + col]

        operations = []
        for bit in range(self.bitsize - 1):
            operations.append(((axes[bit + 1],), (1,), axes[bit]))

        for i in range(self.bitsize):
            operations.append(((axes[i], nu_axis(0, i)), (1, 0), ancilla_axis(i, 0)))
            operations.append(((ancilla_axis(i, 0), nu_axis(1, i)), (1, 0), ancilla_axis(i, 1)))
            operations.append(((), (), nu_axis(2, i)))
            operations.append(((ancilla_axis(i, 1), nu_axis(2, i)), (1, 1), flag_axis))
            operations.append(((), (), nu_axis(2, i)))

        if self.is_adjoint:
            operations = reversed(operations)

        for control_axes, control_values, target_axis in operations:
            self._apply_controlled_x(args.target_tensor, control_axes, control_values, target_axis)
        return args.target_tensor

    @staticmethod
    def _apply_controlled_x(target_tensor, control_axes, control_values, target_axis):
        idx_zero = [slice(None)] * target_tensor.ndim
        idx_one = [slice(None)] * target_tensor.ndim
        for axis, value in zip(control_axes, control_values):
            idx_zero[axis] = value
            idx_one[axis] = value
        idx_zero[target_axis] = 0
        idx_one[target_axis] = 1

        idx_zero = tuple(idx_zero)
        idx_one = tuple(idx_one)
        tmp = target_tensor[idx_zero].copy()
        target_tensor[idx_zero] = target_tensor[idx_one]
        target_tensor[idx_one] = tmp

    def adjoint(self):
        return FlagNuLessThanMu(self.bitsize, is_adjoint=not self.is_adjoint)

    def __pow__(self, power):
        if power == 1:
            return self
        if power == -1:
            return self.adjoint()
        return NotImplemented

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # return {(CNOT(),self.bitsize-1),(MultiAnd(cvs=(1,0,0,0)), self.bitsize)}   
        return {(CNOT(),self.bitsize-1),(And(cv1=1,cv2=0), 2*self.bitsize),(Toffoli(),self.bitsize),(XGate(),2*self.bitsize)}   
