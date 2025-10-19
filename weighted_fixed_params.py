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
        u = Function("u")(*input_variables)

        L_square = D / Sa
        coef = -1 / L_square

        self.equations = {}
        self.equations["custom_pde"] = (u.diff(x, 2) + coef * u)


# -------------------------------
# Solver Function
# -------------------------------
@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:

    # Physical parameters
    D = 1 / (3 * 1.5)
    Sa = 18.0
    S0 = 1.0

    L_square = D / Sa
    L = math.sqrt(L_square)

    ode = NeutronDiffusionNonMult1D(D, Sa)
    x = Symbol("x")

    # Network
    custom_net = instantiate_arch(
        input_keys=[Key("x")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = ode.make_nodes() + [custom_net.make_node(name="ode_network")]

    # -------------------------------
    # Geometry setup (CHANGED SECTION)
    # -------------------------------
    a = 1.0
    a_ex = a + 0.7104 * 3 * D
    min_x = 0.0
    max_x = a_ex / 2

    line = Line1D(min_x, max_x, parameterization=Parameterization({"x": [0, 1]}))
    ode_domain = Domain()

    # Custom sampling + weighting near RHS
    def rhs_bias_sampler(n):
        """Generate biased samples toward RHS (x^2 scaling)."""
        uniform = np.linspace(0, 1, n)
        biased = uniform**2
        return min_x + (max_x - min_x) * biased

    def rhs_weight(invar):
        """Increase PDE loss weight toward RHS."""
        x_norm = invar["x"] / max_x
        return 1.0 + 4.0 * x_norm  # weight up to ×5 near RHS

    # -------------------------------
    # Boundary Conditions
    # -------------------------------
    numerator_phi0 = np.sinh((a_ex) / (2 * L))
    denominator_phi0 = np.cosh(a_ex / (2 * L))
    phi_0 = ((S0 * L) / (2 * D)) * (numerator_phi0 / denominator_phi0)

    bc_min_x = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=line,
        outvar={"u": phi_0},
        criteria=sympy.Eq(x, sympy.Float(min_x)),
        batch_size=cfg.batch_size.bc_min
    )
    ode_domain.add_constraint(bc_min_x, "bc_min")

    bc_max_x = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=line,
        outvar={"u": 0.0},
        criteria=sympy.Eq(x, sympy.Float(max_x)),
        batch_size=cfg.batch_size.bc_max
    )
    ode_domain.add_constraint(bc_max_x, "bc_max")

    # -------------------------------
    # Interior PDE constraint (CHANGED SECTION)
    # -------------------------------
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=line,
        outvar={"custom_pde": 0},
        batch_size=cfg.batch_size.interior,
        loss_weight_fn=rhs_weight,  # emphasize RHS
        invar_fn=lambda n: {"x": rhs_bias_sampler(n).reshape(-1, 1)}  # biased sampling
    )
    ode_domain.add_constraint(interior, "interior")

    # -------------------------------
    # Validator
    # -------------------------------
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
            "s0": np.full_like(points, S0),
            "Sa": np.full_like(points, Sa),
        },
        true_outvar={"u": u_true.reshape(-1, 1)},
        batch_size=1024
    )
    ode_domain.add_validator(validator, "validator_fixed_S0_1_Sa_18")

    # -------------------------------
    # Solver
    # -------------------------------
    slv = Solver(cfg, ode_domain)
    slv.solve()


if __name__ == '__main__':
    run()
