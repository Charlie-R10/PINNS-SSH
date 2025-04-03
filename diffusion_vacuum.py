import math
import numpy as np

import sympy
from sympy import Symbol, Number, Function

from modulus.sym.key import Key
from modulus.sym.geometry.primitives_1d import Line1D
from modulus.sym.domain import Domain
from modulus.sym.solver import Solver
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.utils.io import InferencerPlotter
from modulus.sym.eq.pde import PDE
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.domain.constraint import PointwiseBoundaryConstraint, PointwiseInteriorConstraint
from modulus.sym.eq.pdes.diffusion import Diffusion
from modulus.sym.domain.parameterization import Parameterization


# Setup of class with 1d NDE
class NeutronDiffusionNonMult1D(PDE):
    def __init__(self, D, Sa):
        x = Symbol("x")

        input_variables = {"x": x}

        u = Function("u")(*input_variables)

        # set equations
        L_square = D / Sa
        coef = -1/L_square
        self.equations = {}
        # self.equations["fick"] = (0.25 * u - 0.5 * D * u.diff(x))
        self.equations["custom_pde"] = (u.diff(x, 2) + coef * u)

@modulus.sym.main(config_path="ode_conf", config_name="config")
def run(cfg: ModulusConfig) -> None:

    D = 1 / (3 * 1.5)
    Sa = 0.005
    s0_sym = Symbol("s0") # declares s0 as sympy symbol not set value
    param_ranges = {s0_sym: (10.0, 20.0)} # introduces param ranges and parameterization - set 10 to 20 for now
    pr = Parameterization(param_ranges)
    L_square = D / Sa
    L = math.sqrt(L_square)

    ode = NeutronDiffusionNonMult1D(D, Sa)

    x = Symbol("x")

    # Creating net
    custom_net = instantiate_arch(
        input_keys=[Key("x"), Key("s0")], # S0 becomes input key as parameterized
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = ode.make_nodes() + [custom_net.make_node(name="ode_network")]

    # Defining geometry
    a = 1.
    a_ex = a + 0.7104 * 3 * D
    min_x = 0
    max_x = a_ex # extrapolated length

    line = Line1D(min_x, max_x)
    ode_domain = Domain()

    # Boundary condition of analytical solution at LHS
    phi_0 = s0_sym * L * (1 - math.exp(-2 * a_ex / L)) / (2 * D * (1 + math.exp(-2 * a_ex / L)))
    bc_min_x = PointwiseBoundaryConstraint(nodes=nodes,
                                           geometry=line,
                                           outvar={"u": phi_0},
                                           criteria=sympy.Eq(x, min_x),
                                           batch_size=cfg.batch_size.bc_min,
                                           parameterization=pr) #only BC that contains S0, therefore only BC needs parameterizaion?
    ode_domain.add_constraint(bc_min_x, "bc_min")

    # Boundary condition that u = 0 at RHS (max x)
    bc_max_x = PointwiseBoundaryConstraint(nodes=nodes,
                                           geometry=line,
                                           outvar={"u": 0},
                                           criteria=sympy.Eq(x, max_x),
                                           batch_size=cfg.batch_size.bc_max)
    ode_domain.add_constraint(bc_max_x, "bc_max")

    # Interior
    interior = PointwiseInteriorConstraint(nodes=nodes,
                                           geometry=line,
                                           outvar={"custom_pde": 0},
                                           batch_size=cfg.batch_size.interior)
    ode_domain.add_constraint(interior, "interior")

    # Add inferencer
    points = np.linspace(0, 1, 101).reshape(101, 1)
    inferencer = PointwiseInferencer(nodes=nodes, invar={"x": points}, output_names=["u"], batch_size=1024, plotter=InferencerPlotter())
    # could add ', "s0": np.full_like(points, 15.0)' to inferencer?
    ode_domain.add_inferencer(inferencer, "inf_data")

    # make solver
    slv = Solver(cfg, ode_domain)

    # start solver
    slv.solve()

if __name__ == '__main__':
