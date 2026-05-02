"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import  numpy as np

import  cirq
import  qualtran                     as  qt
import  qualtran.cirq_interop.testing  as  qt_test

import  qualtran.bloqs.chemistry.hubbard_model.qubitization as  qt_hm

from    functools                                import   cached_property
from    attrs                                    import   frozen

from    pyLIQTR.BlockEncodings                 import   VALID_ENCODINGS
from    pyLIQTR.BlockEncodings.BlockEncoding   import   BlockEncoding_select_prepare
from    pyLIQTR.circuits.operators.prepare_FermiHubbard import PrepareHubbardPYL,PrepareHubbardPYL_invert_workaround



@frozen
class SelectHubbardControlZero(qt.GateWithRegisters):
    x_dim: int
    y_dim: int

    @cached_property
    def _select_gate(self):
        return qt_hm.SelectHubbard(x_dim=self.x_dim, y_dim=self.y_dim, control_val=1)

    @cached_property
    def control_registers(self):
        return self._select_gate.control_registers

    @cached_property
    def selection_registers(self):
        return self._select_gate.selection_registers

    @cached_property
    def target_registers(self):
        return self._select_gate.target_registers

    @cached_property
    def signature(self):
        return self._select_gate.signature

    def decompose_from_registers(self, context, **quregs):
        control = np.ravel(quregs["control"])
        yield cirq.X.on(*control)
        yield self._select_gate.on_registers(**quregs)
        yield cirq.X.on(*control)

    def __str__(self):
        return f"C0SelectHubbard({self.x_dim}, {self.y_dim})"





class fermi_hubbard_square_encoding(BlockEncoding_select_prepare):

    def __init__(self,ProblemInstance, **kwargs):

        super().__init__(ProblemInstance,**kwargs)

        if (self.PI._model != 'Fermi-Hubbard Model - SquareLattice(regular)'):
            raise NotImplementedError

        self.dims = ProblemInstance.shape

        self._encoding_type  =  VALID_ENCODINGS.FermiHubbardSquare

        if self._control_val == 0:
            self._select_gate = SelectHubbardControlZero(x_dim=self.dims[0], y_dim=self.dims[1])
        else:
            self._select_gate = qt_hm.SelectHubbard(x_dim=self.dims[0],
                                                    y_dim=self.dims[1],
                                                    control_val=self._control_val)
   
        self._prepare_gate   =  PrepareHubbardPYL( x_dim=self.dims[0], 
                                                        y_dim=self.dims[1], 
                                                        t=-ProblemInstance.J,
                                                        u=ProblemInstance.U )
        self._inverse_prepare_workaround = PrepareHubbardPYL_invert_workaround( x_dim=self.dims[0], 
                                                        y_dim=self.dims[1], 
                                                        t=-ProblemInstance.J,
                                                        u=ProblemInstance.U )
        

    @property
    def alpha(self):
        N  =  2*np.prod(self.PI.shape)
        J  =  np.abs(self.PI.J)
        U  =  np.abs(self.PI.U)
        alpha = 2*N*J + U*N/2
        return (alpha)

