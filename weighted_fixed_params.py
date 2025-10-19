import math
import numpy as np
import sympy
from sympy import Symbol, Function

import physicsnemo.sym

# PhysicsNeMo imports
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.key import Key
from physicsnemo.sym.geometry.primitives_1d import Line1D
from physicsnemo.sym.domain.domain import Domain
from physicsnemo.sym.domain.constraint import PointwiseBoundaryConstraint, PointwiseInteriorConstraint
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.geometry.parameterization import Parameterization
from physicsnemo.sym.eq.pde import PDE


# -------------------------------
# PDE Definition
# -------------------------------
class NeutronDiffusionNonMult1D(PDE):
    def __init__(self, D, Sa):
        x = Symbol("x")
        input_variables = {"x": x}
        u = Function("u")(*input_varia_*
