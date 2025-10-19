import math
import numpy as np
import sympy
from sympy import Symbol, Function
import matplotlib.pyplot as plt

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


# ---------------------------------------
# PDE Definition
# ---------------------------------------
class NeutronDiffusionNonMult1D(PDE):
    def __init__(self, D, Sa):
        x = Symbol("x")
        input_variables = {"x": x}
        u = Function("u")(*input_variables)

        L_square = D / Sa
        coef = -1 / L_square

        self.equations = {}
        self.equations["custom_pde"] = (u.diff(x, 2) + coef * u)


# ---------------------------------------
# Solver Function
# ---------------------------------------
@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:

    # Physical parameters
    D = 1 / (3 * 1.5)
    Sa = 18.0
    S0 = 1.0

    L_square = D / Sa
    L = math.sqrt(L_square)

    # Define PDE
    ode = NeutronDiffusionNonMult1D(D, Sa)
    x = Symbol("x")

    # Define Network
    custom_net = instantiate_arch(
        input_keys=[Key("x")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = ode.make_nodes() + [custom_net.make_node(name="ode_network")]

    # ---------------------------------------
    # Geometry setup
    # ---------------------------------------
    a = 1.0
    a_ex = a + 0.7104 * 3 * D
    min_x = 0.0
    max_x = a_ex / 2

    line = Line1D(min_x, max_x, parameterization=Parameterization({"x": (0, 1)}))
    ode_domain = Domain()

    # --- Custom sampling and weighting ---
    def rhs_bias_sampler(n):
        """Generate biased samples toward RHS (x^2 scaling)."""
        uniform = np.linspace(0, 1, n)
        biased = uniform**2
        return min_x + (max_x - min_x) * biased

    def rhs_weight(invar):
        """Increase PDE loss weight toward the right-hand side."""
        x_norm = invar["x"] / max_x
        return 1.0 + 4.0 * x_norm  # up to ×5 weighting near RHS

    # ---------------------------------------
    # Boundary Conditions (explicit points)
    # ---------------------------------------
    numerator_phi0 = np.sinh((a_ex) / (2 * L))
    denominator_phi0 = np.cosh(a_ex / (2 * L))
    phi_0 = ((S0 * L) / (2 * D)) * (numerator_phi0 / denominator_phi0)

    # Explicit invars for 1D boundaries
    bc_min = {"x": np.array([[min_x]])}
    bc_max = {"x": np.array([[max_x]])}

    # BC at x = 0
    bc_min_x = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=line,
        outvar={"u": phi_0},
        invar=bc_min,
        batch_size=cfg.batch_size.bc_min
    )
    ode_domain.add_constraint(bc_min_x, "bc_min")

    # BC at x = max_x
    bc_max_x = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=line,
        outvar={"u": 0.0},
        invar=bc_max,
        batch_size=cfg.batch_size.bc_max
    )
    ode_domain.add_constraint(bc_max_x, "bc_max")

    # ---------------------------------------
    # Interior PDE Constraint (biased sampling)
    # ---------------------------------------
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=line,
        outvar={"custom_pde": 0},
        batch_size=cfg.batch_size.interior,
        loss_weight_fn=rhs_weight,  # emphasize RHS
        invar_fn=lambda n: {"x": rhs_bias_sampler(n).reshape(-1, 1)}  # biased sampling
    )
    ode_domain.add_constraint(interior, "interior")

    # ---------------------------------------
    # Validator (for plotting / evaluation)
    # ---------------------------------------
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

    # ---------------------------------------
    # Solver
    # ---------------------------------------
    slv = Solver(cfg, ode_domain)
    slv.solve()

    # ---------------------------------------
    # Plot analytical vs predicted
    # ---------------------------------------
    model = slv.domain.constraints["interior"].nodes[-1].evaluate
    x_test = np.linspace(min_x, max_x, 200).reshape(-1, 1)
    pred = slv.infer({"x": x_test})["u"]

    plt.figure(figsize=(7, 5))
    plt.plot(x_test, u_true, label="Analytical", lw=2)
    plt.plot(x_test, pred, "--", label="PINN Prediction", lw=2)
    plt.yscale("log")
    plt.xlabel("x")
    plt.ylabel(r"$\phi(x)$")
    plt.title(r"Neutron Flux $\phi(x)$ (Log Scale)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    run()

