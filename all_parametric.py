import math
import numpy as np

import sympy
from sympy import Symbol, Function
import modulus.sym
from modulus.sym.key import Key
from modulus.sym.geometry.primitives_1d import Line1D
from modulus.sym.domain import Domain
from modulus.sym.solver import Solver
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.utils.io import InferencerPlotter
from modulus.sym.eq.pde import PDE
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.domain.constraint import PointwiseBoundaryConstraint, PointwiseInteriorConstraint
from modulus.sym.geometry.parameterization import Parameterization

# Setup of class with 1d NDE
class NeutronDiffusionNonMult1D(PDE):
    def __init__(self):
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
        D_sym: (0.1, 1.0),
        Sa_sym: (0.001, 0.01)
    }
    pr = Parameterization(param_ranges)

    ode = NeutronDiffusionNonMult1D()

    x = Symbol("x")
    custom_net = instantiate_arch(
        input_keys=[Key("x"), Key("s0"), Key("D"), Key("Sa")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = ode.make_nodes() + [custom_net.make_node(name="ode_network")]

    a = 1.
    min_x = 0
    max_x = 3
    line = Line1D(min_x, max_x)
    ode_domain = Domain()

    # Boundary condition at LHS
    L_sym = sympy.sqrt(D_sym / Sa_sym)
    a_ex = a + 0.7104 * 3 * D_sym
    phi_0 = s0_sym * L_sym * (1 - sympy.exp(-2 * a_ex / L_sym)) / (2 * D_sym * (1 + sympy.exp(-2 * a_ex / L_sym)))

    bc_min_x = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=line,
        outvar={"u": phi_0},
        criteria=sympy.Eq(x, min_x),
        batch_size=cfg.batch_size.bc_min,
        parameterization=pr)

    ode_domain.add_constraint(bc_min_x, "bc_min")

    # Boundary condition at RHS
    bc_max_x = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=line,
        outvar={"u": 0},
        criteria=sympy.Eq(x, max_x),
        batch_size=cfg.batch_size.bc_max,
        parameterization=pr)

    ode_domain.add_constraint(bc_max_x, "bc_max")

    # Interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=line,
        outvar={"custom_pde": 0},
        batch_size=cfg.batch_size.interior,
        parameterization=pr)

    ode_domain.add_constraint(interior, "interior")

    # Define points
    points = np.linspace(0, 1, 101).reshape(101, 1)
    s0_values = np.arange(10, 20.01, 0.5)

    # Validator loop
    for s0_val in s0_values:
        for D_val in [0.1, 0.5, 1.0]:
            for Sa_val in [0.001, 0.005, 0.01]:
                L_val = math.sqrt(D_val / Sa_val)
                a_ex = a + 0.7104 * 3 * D_val

                # Analytical Solution
                u_true = (
                    s0_val * L_val * (1 - np.exp(-2 * a_ex / L_val)) /
                    (2 * D_val * (1 + np.exp(-2 * a_ex / L_val))) *
                    (np.cosh((points - a) / L_val) / np.cosh(a_ex / L_val))
                )

                # Validator
                validator = PointwiseValidator(
                    nodes=nodes,
                    invar={"x": points, "s0": np.full_like(points, s0_val), "D": np.full_like(points, D_val), "Sa": np.full_like(points, Sa_val)},
                    true_outvar={"u": u_true.reshape(-1, 1)},
                    batch_size=1024)

                ode_domain.add_validator(validator, f"validator_s0_{s0_val}_D_{D_val}_Sa_{Sa_val}")

    # Solver
    slv = Solver(cfg, ode_domain)
    slv.solve()

if __name__ == '__main__':
    run()
