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
from modulus.sym.domain.monitor import PointwiseMonitor
import itertools
import torch


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

    # Add inferencers for all parameters
    x_vals = np.linspace(0, 1, 101).reshape(-1, 1)

    # Parameter values to sweep through
    s0_vals = [10.0, 15.0, 20.0]
    D_vals = [0.1, 0.5, 1.0]
    Sa_vals = [0.001, 0.005, 0.01]

    # Create all combinations of parameters
    param_combos = list(itertools.product(s0_vals, D_vals, Sa_vals))

    for i, (s0_val, D_val, Sa_val) in enumerate(param_combos):
        inferencer = PointwiseInferencer(
            nodes=nodes,
            invar={
                "x": x_vals,
                "s0": np.full_like(x_vals, s0_val),
                "D": np.full_like(x_vals, D_val),
                "Sa": np.full_like(x_vals, Sa_val),
            },
            output_names=["u"],
            batch_size=1024,
            plotter=InferencerPlotter(),
        )
        ode_domain.add_inferencer(inferencer, name=f"inf_s0{s0_val}_D{D_val}_Sa{Sa_val}")

        # Compute L and a_ex analytically for this combo
        L_val = np.sqrt(D_val / Sa_val)
        a = 1.0
        a_ex = a + 0.7104 * 3 * D_val

        # Create analytical solution at each x
        u_true_vals = (
            s0_val * L_val * (1 - np.exp(-2 * a_ex / L_val))
            / (2 * D_val * (1 + np.exp(-2 * a_ex / L_val)))
            * (np.cosh((x_vals - a) / L_val) / np.cosh(a_ex / L_val))
        )

        # Add monitor
        monitor = PointwiseMonitor(
            invar={
                "x": x_vals,
                "s0": np.full_like(x_vals, s0_val),
                "D": np.full_like(x_vals, D_val),
                "Sa": np.full_like(x_vals, Sa_val),
            },
            output_names=["u"],
            metrics={
                "l2_error_u": lambda var: torch.norm(var["u"] - torch.tensor(u_true_vals, dtype=var["u"].dtype), 2),
            },
            nodes=nodes
        )
        ode_domain.add_monitor(monitor)

    # make solver
    slv = Solver(cfg, ode_domain)

    # start solver
    slv.solve()

if __name__ == '__main__':
    run()
