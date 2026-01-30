import math
import numpy as np
import sympy
from sympy import Symbol, Function, Lambda, Eq
import physicsnemo.sym

# PhysicsNeMo v25.03 imports
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.key import Key
from physicsnemo.sym.geometry.primitives_1d import Line1D
from physicsnemo.sym.domain.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.node import Node
from physicsnemo.sym.geometry.parameterization import Parameterization
from physicsnemo.sym.eq.pde import PDE
from diffusion_equation import DiffusionEquation1D, VacuumBoundary, ReflectiveBoundary


# Config from physics nemo
@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    ext_lengt_bc = True
    D = 1.0
    a = 5
    # Sigma_a and Q as symbols to vary parameters
    Sigma_a = Symbol("Sigma_a")
    Q = Symbol("Q")
    a_ext = a + 3 * 0.7104 * D


    
    # Normalisation between [0, 1] (LOG SCALE)
    Sigma_a_hat = Symbol("Sigma_a_hat")
    Q_hat = Symbol("Q_hat")

    Sigma_a_min = 1e-3
    Sigma_a_max = 10.0
    Q_min = 1e-3
    Q_max = 10.0
    
    Sigma_a_expr = Sigma_a_min * (Sigma_a_max / Sigma_a_min) ** Sigma_a_hat
    Q_expr = Q_min * (Q_max / Q_min) ** Q_hat


    # mapping nodes to map normalized for PINN

    mapping_nodes = [
    Node.from_sympy(Sigma_a_expr, "Sigma_a"),
    Node.from_sympy(Q_expr, "Q"),
    ]


    # Ranges set from parameterized values ([0,1])
    param_ranges = {
        Sigma_a_hat: (0.0, 1.0),
        Q_hat: (0.0, 1.0),
    }
    pr = Parameterization(param_ranges)

    # Uses original (transformed) variable for BCs
    de = DiffusionEquation1D(u="u", D=D, Sigma_a=Sigma_a, Q=Q)
    vb = VacuumBoundary(u="u", D=D, extrapolated_length=ext_lengt_bc)
    rb = ReflectiveBoundary(u="u", D=D)

    # Uses normalized variable for network
    diffusion_net = instantiate_arch(
        input_keys=[Key("x"), Key("Sigma_a_hat"), Key("Q_hat")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = (
        mapping_nodes
        + de.make_nodes()
        + vb.make_nodes()
        + rb.make_nodes()
        + [diffusion_net.make_node(name="diffusion_network")]
    )

    # Defining geometry as 1D line with extrapolated length
    x = Symbol("x")
    geo = Line1D(0, a_ext)
    domain = Domain()

    # boundary conditions
    LB = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"reflective_boundary": 0},
        batch_size=cfg.batch_size.LB,
        criteria=Eq(x, 0),
        parameterization=pr,
    )
    domain.add_constraint(LB, "LB")

    RB = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"vacuum_boundary": 0},
        batch_size=cfg.batch_size.RB,
        criteria=Eq(x, a_ext),
        parameterization=pr,
    )
    domain.add_constraint(RB, "RB")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"diffusion_equation_u": 0},
        batch_size=cfg.batch_size.interior,
        parameterization=pr,
    )
    domain.add_constraint(interior, "interior")

    # add validation data (define analytical solution symbollically)
    X = np.linspace(0, a_ext, 101)[:, None]

    def analytical_solution(x, D, a_ext, Sigma_a, Q):
        L = np.sqrt(D / Sigma_a)
        denom = 2.0 * np.cosh(a_ext / L)

        return (
            -L**2 * Q * np.exp(x / L) / denom
            + L**2 * Q
            + (L**2 * Q * np.exp(a_ext / L) / denom - L**2 * Q)
            * np.exp(a_ext / L)
            * np.exp(-x / L)
        )

    # Validation grid 
    X = np.linspace(0, a_ext, 101)[:, None]


    # Set values of Sa and Q to validate (can this be automated?)
    i = 0
    for Sa_val in [0.05, 1.0, 2.0]:
        for Q_val in [0.05, 1.0, 2.0]:

            u_true = analytical_solution(
                X.flatten(),
                D,
                a_ext,
                Sa_val,
                Q_val,
            )

            Sigma_hat_val = np.log(Sa_val / Sigma_a_min) / np.log(Sigma_a_max / Sigma_a_min)
            Q_hat_val = np.log(Q_val / Q_min) / np.log(Q_max  / Q_min)
            validator = PointwiseValidator(
                nodes=nodes,
                invar={
                    "x": X,
                    "Sigma_a_hat": np.full_like(X, Sigma_hat_val),
                    "Q_hat": np.full_like(X, Q_hat_val)
                },
                true_outvar={
                    "u": u_true.reshape(-1, 1),
                },
                batch_size=256,
            )

            domain.add_validator(
                validator,
                f"val_Sa_{Sa_val}_Q_{Q_val}_{i}",
            )
            i += 1

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()

if __name__ == "__main__":
    run()
