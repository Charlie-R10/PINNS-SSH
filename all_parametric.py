import math
import numpy as np

import sympy
from sympy import Symbol, Number, Function
import modulus.sym
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
from modulus.sym.geometry.parameterization import Parameterization


# Setup of class with 1d NDE
class NeutronDiffusionNonMult1D(PDE):
    def __init__(self):
        # x, D and Sa all now symbols - not passed through as values
        x = Symbol("x")
        D = Symbol("D")
        Sa = Symbol("Sa")

        input_variables = {"x": x, "D": D, "Sa": Sa}
        u = Function("u")(*input_variables)

        L_square = D / Sa
        coef = -1 / L_square
        self.equations = {}
        self.equations["custom_pde"] = u.diff(x, 2) + coef * u

@modulus.sym.main(config_path="ode_conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    D_sym = Symbol("D")
    Sa_sym = Symbol("Sa")
    s0_sym = Symbol("s0")
    param_ranges = {
        s0_sym: (10.0, 20.0),
        D_sym: (0.1, 1.0),  # example range, adjust as needed
        Sa_sym: (0.001, 0.01)  # example range, adjust as needed
    }
    pr = Parameterization(param_ranges)
    L_sym = sympy.sqrt(D_sym / Sa_sym)

    ode = NeutronDiffusionNonMult1D()

    x = Symbol("x")

    # Creating net
    custom_net = instantiate_arch(
        input_keys=[Key("x"), Key("s0"), Key("D"), Key("Sa")], # S0,D and Sa becomes input key as parameterized
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = ode.make_nodes() + [custom_net.make_node(name="ode_network")]

    # Defining geometry
    a = 1.
    a_ex = a + 0.7104 * 3 * D_sym
    min_x = 0
    max_x = 3 # extrapolated length - set as 3 for max possible lenght for now

    line = Line1D(min_x, max_x)
    ode_domain = Domain()

    # Boundary condition of analytical solution at LHS - includes Lsym, Dsym etc as parametric
    phi_0 = s0_sym * L_sym * (1 - sympy.exp(-2 * a_ex / L_sym)) / (2 * D_sym * (1 + sympy.exp(-2 * a_ex / L_sym))) #change to sympy.exp
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
                                           batch_size=cfg.batch_size.bc_max,
                                           parameterization=pr)
    ode_domain.add_constraint(bc_max_x, "bc_max")

    # Interior
    interior = PointwiseInteriorConstraint(nodes=nodes,
                                           geometry=line,
                                           outvar={"custom_pde": 0},
                                           batch_size=cfg.batch_size.interior,
                                           parameterization=pr)
    ode_domain.add_constraint(interior, "interior")

    # Add inferencer
    points = np.linspace(0, 1, 101).reshape(101, 1)
    inferencer = PointwiseInferencer(nodes=nodes, invar={
        "x": points,
        "s0": np.full_like(points, 15.0), #key baseline values for each paremeter to work off
        "D": np.full_like(points, 1 / (3 * 1.5)),
        "Sa": np.full_like(points, 0.005),
        }, output_names=["u"], batch_size=1024, plotter=InferencerPlotter())
    ode_domain.add_inferencer(inferencer, "inf_data")

    # make solver
    slv = Solver(cfg, ode_domain)

    # start solver
    slv.solve()

if __name__ == '__main__':
    run()
