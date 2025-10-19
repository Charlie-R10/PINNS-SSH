import math
import numpy as np
import sympy
from sympy import Symbol, Function
import torch
import torch.nn as nn


import physicsnemo.sym

# PhysicsNeMo v25.03 imports
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


class NeutronDiffusionNonMult1D(PDE):
    def __init__(self, D, Sa):
        x = Symbol("x")
        input_variables = {"x": x}
        u = Function("u")(*input_variables)

        # set equations
        L_square = D / Sa
        coef = -1 / L_square

        # IMPORTANT: use the same equation key that constraints will reference
        self.equations = {}
        self.equations["custom_pde"] = (u.diff(x, 2) + coef * u)


# Config from PhysicsNeMo
@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:

    # physical params
    D = 1 / (3 * 1.5)
    Sa = 18.0
    S0 = 1.0

    # derived lengths
    L_square = D / Sa
    L = math.sqrt(L_square)

    # PDE
    ode = NeutronDiffusionNonMult1D(D, Sa)

    x = Symbol("x")

    # Creating net
    custom_net = instantiate_arch(
        input_keys=[Key("x")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
        activation=nn.Softplus(),
    )

    nodes = ode.make_nodes() + [custom_net.make_node(name="ode_network")]

    # Defining geometry
    a = 1.0
    a_ex = a + 0.7104 * 3 * D
    min_x = 0.0
    max_x = a_ex/2

    line = Line1D(min_x, max_x)
    ode_domain = Domain()

    # Analytical value at x=0 (phi_0)
    numerator_phi0 = np.sinh((a_ex) / (2 * L))
    denominator_phi0 = np.cosh(a_ex / (2 * L))
    phi_0 = ((S0 * L) / (2 * D)) * (numerator_phi0 / denominator_phi0)

    # Boundary condition at x = 0
    bc_min_x = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=line,
        outvar={"u": phi_0},
        criteria=sympy.Eq(x, sympy.Float(min_x)),
        batch_size=cfg.batch_size.bc_min
    )
    ode_domain.add_constraint(bc_min_x, "bc_min")

    # Boundary condition at x = max_x (use sympy.Float)
    bc_max_x = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=line,
        outvar={"u": 0.0},
        criteria=sympy.Eq(x, sympy.Float(max_x)),
        batch_size=cfg.batch_size.bc_max
    )
    ode_domain.add_constraint(bc_max_x, "bc_max")

    # Interior PDE residual
    # NOTE: the key "custom_pde" must match the key in NeutronDiffusionNonMult1D.self.equations
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=line,
        outvar={"custom_pde": 0},   # <-- MATCHING KEY
        batch_size=cfg.batch_size.interior
    )
    ode_domain.add_constraint(interior, "interior")

    # Add validator: sample points across the actual domain [min_x, max_x]
    points = np.linspace(min_x, max_x, 101).reshape(101, 1)

    def analytical_solution_fixed(x, D_val, a_ex_val):
        S0_loc = 1.0
        Sa_loc = 18.0
        L_loc = math.sqrt(D_val / Sa_loc)
        numerator = np.sinh((a_ex_val - 2 * x) / (2 * L_loc))
        denominator = np.cosh(a_ex_val / (2 * L_loc))
        return (S0_loc * L_loc / (2 * D_val)) * (numerator / denominator)

    u_true = analytical_solution_fixed(points.flatten(), D, a_ex)

    validator = PointwiseValidator(
        nodes=nodes,
        invar={
            "x": points,
            "s0": np.full_like(points, S0),   # fixed S0
            "Sa": np.full_like(points, Sa),   # fixed Sa
        },
        true_outvar={"u": u_true.reshape(-1, 1)},
        batch_size=1024
    )
    ode_domain.add_validator(validator, "validator_fixed_S0_1_Sa_18")

    # make solver
    slv = Solver(cfg, ode_domain)

    # start solver
    slv.solve()


if __name__ == '__main__':
    run()

