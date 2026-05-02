"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
from functools import cached_property
from numpy.typing import NDArray
from typing import Set
from qualtran import GateWithRegisters, Signature, QAny, QBit, Side, Register
from qualtran.bloqs.mcmt import MultiAnd, And
from qualtran.bloqs.basic_gates import XGate

class FlagZeroAsFailure(GateWithRegisters):
    """Flags the state representing minus zero in the :math:`\\nu` register.

    Parameters:
        bitsize: the number of bits in the :math:`\\nu` register.

    Registers:

    .. line-block::
        nu: The register being checked for the minus zero state.
        flag_dim: Qubits flagging occurence of minus zero for each dimension of :math:`\\nu`.
        flag_minus_zero: Qubit flagging occurence of minus zero overall.
        and_ancilla: Ancilla used for the And ladder.
        flag_ancilla: Ancilla used for OR to mark flag_minus_zero.

    References:
        `Fault-Tolerant Quantum Simulations of Chemistry in First Quantization <https://arxiv.org/abs/2105.12767>`_
        Eq 80, pg 22
    """

    def __init__(self,bitsize:int,is_adjoint:bool=False):
        self.bitsize = bitsize 
        self.is_adjoint = is_adjoint

    @cached_property
    def signature(self) -> Signature:
        side = Side.LEFT if self.is_adjoint else Side.RIGHT
        return Signature(
            [
                Register("nu", QAny(self.bitsize + 1), shape=(3,)),
                Register("flag_dim", QBit(), shape=(3,), side=side),
                Register("flag_minus_zero", QBit(), side=side),
                Register("and_ancilla", QAny(self.bitsize - 1), shape=(3,), side=side),
                Register("flag_ancilla", QBit(), side=side)
            ]
        )

    def pretty_name(self) -> str:
        return r'ν≠-0'

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:

        nu_reg = quregs['nu']
        flags = quregs['flag_dim']
        ancillas = quregs['and_ancilla']
        flag_minus_zero = quregs['flag_minus_zero'][0]
        flag_ancilla = quregs['flag_ancilla'][0]

        ops = list(self._decompose_forward_from_registers(nu_reg, flags, ancillas, flag_minus_zero, flag_ancilla))
        if self.is_adjoint:
            yield from (cirq.inverse(op) for op in reversed(ops))
        else:
            yield from ops

    def _decompose_forward_from_registers(self, nu_reg, flags, ancillas, flag_minus_zero, flag_ancilla):
        and_gate = MultiAnd(cvs=(1,)+(0,)*self.bitsize)

        # check each dimension for the state |-0> = |10...0> and flag it
        for dim,qbs in enumerate(nu_reg):

            yield and_gate.on_registers(ctrl=[[bit] for bit in qbs],target=flags[dim],junk=[[bit] for bit in ancillas[dim]])

        # check if any of the 3 flag qubits are 1 using 2 OR gates
        yield And(cv1=0,cv2=0).on(flags[0][0],flags[1][0],flag_ancilla)
        yield XGate().on(flag_ancilla)
        yield And(cv1=0,cv2=0).on(flag_ancilla,flags[2][0],flag_minus_zero)
        yield XGate().on(flag_minus_zero)

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs):
        axes = tuple(args.axes)
        nu_len = 3 * (self.bitsize + 1)
        flag_dim_start = nu_len
        flag_minus_zero_axis = axes[flag_dim_start + 3]
        and_ancilla_start = flag_dim_start + 4
        flag_ancilla_axis = axes[and_ancilla_start + 3 * (self.bitsize - 1)]

        nu_axis = lambda dim, bit: axes[dim * (self.bitsize + 1) + bit]
        flag_dim_axis = lambda dim: axes[flag_dim_start + dim]
        and_ancilla_axis = lambda dim, bit: axes[and_ancilla_start + dim * (self.bitsize - 1) + bit]

        operations = []
        cvs = (1,) + (0,) * self.bitsize
        for dim in range(3):
            controls = [nu_axis(dim, bit) for bit in range(self.bitsize + 1)]
            if self.bitsize == 1:
                operations.append((tuple(controls), cvs, flag_dim_axis(dim)))
            else:
                operations.append(((controls[0], controls[1]), cvs[:2], and_ancilla_axis(dim, 0)))
                for bit in range(1, self.bitsize - 1):
                    operations.append(((and_ancilla_axis(dim, bit - 1), controls[bit + 1]), (1, cvs[bit + 1]), and_ancilla_axis(dim, bit)))
                operations.append(((and_ancilla_axis(dim, self.bitsize - 2), controls[-1]), (1, cvs[-1]), flag_dim_axis(dim)))

        operations.append(((flag_dim_axis(0), flag_dim_axis(1)), (0, 0), flag_ancilla_axis))
        operations.append(((), (), flag_ancilla_axis))
        operations.append(((flag_ancilla_axis, flag_dim_axis(2)), (0, 0), flag_minus_zero_axis))
        operations.append(((), (), flag_minus_zero_axis))

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
        return FlagZeroAsFailure(self.bitsize, is_adjoint=not self.is_adjoint)

    def __pow__(self, power):
        if power == 1:
            return self
        if power == -1:
            return self.adjoint()
        return NotImplemented


    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(MultiAnd(cvs=(1,)+(0,)*self.bitsize), 3), (And(cv1=0,cv2=0), 2),(XGate(),2)}
