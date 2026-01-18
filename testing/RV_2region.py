""" 1D neutron diffusion equation
    Ref: Duderstadt
"""

from physicsnemo.sym.eq.pde import PDE

import numpy as np
from sympy import Symbol, Eq

import physicsnemo.sym
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_1d import Line1D
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)

# from physicsnemo.sym.eq.pdes.wave_equation import WaveEquation
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key
from diffusion_equation import (
    DiffusionEquation1D,
    VacuumBoundary,
    ReflectiveBoundary,
    InterfaceDiffusion1D,
)
from physicsnemo.sym.eq.pdes.diffusion import Diffusion
from physicsnemo.sym.node import Node
from physicsnemo.sym.domain.parameterization import Parameterization




@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # make list of nodes to unroll graph on
    ext_lengt_bc = True
    D1 = 1.0
    Sigma_a1 = Symbol("Sigma_a1")  # 0.01
    D2 = 0.8
    Sigma_a2 = Symbol("Sigma_a2")  # 0.1
    a1 = 5.0
    a2 = 10.0
    Q = Symbol("Q")  # 1.0

    if ext_lengt_bc:
        a_ext = a2 + 3 * 0.7104 * D2
    else:
        a_ext = a2
    # a_ext = 2 * a

    # Normalization [0,1]
    Sigma_a1_hat = Symbol("Sigma_a1_hat")
    Sigma_a2_hat = Symbol("Sigma_a2_hat")
    Q_hat = Symbol("Q_hat")

    Sigma_a1_max = 0.5
    Sigma_a2_max = 0.1
    Q_max = 1.5

    Sigma_a1_expr = Sigma_a1_hat * Sigma_a1_max
    Sigma_a2_expr = Sigma_a2_hat * Sigma_a2_max
    Q_expr = Q_hat * Q_max

    # mapping nodes to map normalized for PINN
    mapping_nodes = [
        Node.from_sympy(Sigma_a1_expr, "Sigma_a1"),
        Node.from_sympy(Sigma_a2_expr, "Sigma_a2"),
        Node.from_sympy(Q_expr, "Q"),
    ]

    de1 = DiffusionEquation1D(u="u1", D=D1, Sigma_a=Sigma_a1, Q=Q)
    de2 = DiffusionEquation1D(u="u2", D=D2, Sigma_a=Sigma_a2, Q=0)
    de_in = InterfaceDiffusion1D(u1="u1", u2="u2", D1=D1, D2=D2)
    vb = VacuumBoundary(u="u2", D=D2, extrapolated_length=ext_lengt_bc)
    rb = ReflectiveBoundary(u="u1", D=D1)

    # Ranges set from parameterized values ([0,1])
    param_ranges = {
        Sigma_a1_hat: (0.0, 1.0),
        Sigma_a2_hat: (0.0, 1.0),
        Q_hat: (0.0, 1.0),
    }
    pr = Parameterization(param_ranges)

    diffusion_net_u1 = instantiate_arch(
        input_keys=[Key("x"), Key("Sigma_a1_hat"), Key("Sigma_a2_hat"), Key("Q_hat")],
        output_keys=[Key("u1")],
        cfg=cfg.arch.fully_connected,
    )

    diffusion_net_u2 = instantiate_arch(
        input_keys=[Key("x"), Key("Sigma_a1_hat"), Key("Sigma_a2_hat"), Key("Q_hat")],
        output_keys=[Key("u2")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = (
        mapping_nodes
        + de1.make_nodes()
        + de2.make_nodes()
        + de_in.make_nodes()
        + vb.make_nodes()
        + rb.make_nodes()
        + [diffusion_net_u1.make_node(name="diffusion_network_u1")]
        + [diffusion_net_u2.make_node(name="diffusion_network_u2")]
    )

    # make geometry
    x = Symbol("x")
    geo1 = Line1D(point_1=0, point_2=a1)
    geo2 = Line1D(point_1=a1, point_2=a_ext)

    # make domain
    domain = Domain()

    # boundary condition
    LB = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo1,
        outvar={"reflective_boundary": 0},
        batch_size=cfg.batch_size.LB,
        criteria=Eq(x, 0),
        parameterization=pr,
    )
    domain.add_constraint(LB, "LB")

    IB = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo1,
        outvar={
            "flux_continuity": 0,
            "current_continuity": 0,
        },
        batch_size=cfg.batch_size.IB,
        criteria=Eq(x, a1),
        parameterization=pr,
    )
    domain.add_constraint(IB, "IB")

    RB = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo2,
        outvar={"vacuum_boundary": 0},
        batch_size=cfg.batch_size.RB,
        criteria=Eq(x, a_ext),
        parameterization=pr,
    )
    domain.add_constraint(RB, "RB")

    # interior 1
    interior1 = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo1,
        outvar={"diffusion_equation_u1": 0},
        bounds={x: (0, a1)},
        batch_size=cfg.batch_size.interior1,
        quasirandom=True,
        parameterization=pr,
    )
    domain.add_constraint(interior1, "interior1")

    # interior 2
    interior2 = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo2,
        outvar={"diffusion_equation_u2": 0},
        bounds={x: (a1, a_ext)},
        batch_size=cfg.batch_size.interior2,
        quasirandom=True,
        parameterization=pr,
    )
    domain.add_constraint(interior2, "interior2")

    # add validation data
    X1 = np.linspace(0, a1, 200)[:, None]
    X2 = np.linspace(a1, a_ext, 200)[:, None]

    def analytical_solution_1(X1, D1, D2, a_ext, Sigma_a1, Sigma_a2, Q, a1):
        L1 = np.sqrt(D1 / Sigma_a1)
        L2 = np.sqrt(D2 / Sigma_a2)
        u1 = D2*L1**3*Q*(np.exp(2*a1/L2) + np.exp(2*a_ext/L2))*np.exp(a1/L1)*np.exp(X1/L1)/(-D1*L2*np.exp(2*a1/L2) +
           D1*L2*np.exp(2*a_ext/L2) + D1*L2*np.exp(2*a1/L2 + 2*a1/L1) - D1*L2*np.exp(2*a_ext/L2 + 2*a1/L1) -
           D2*L1*np.exp(2*a1/L2) - D2*L1*np.exp(2*a_ext/L2) - D2*L1*np.exp(2*a1/L2 + 2*a1/L1) -
           D2*L1*np.exp(2*a_ext/L2 + 2*a1/L1)) + D2*L1**3*Q*(np.exp(2*a1/L2) +
           np.exp(2*a_ext/L2))*np.exp(a1/L1)*np.exp(-X1/L1)/(-D1*L2*np.exp(2*a1/L2) + D1*L2*np.exp(2*a_ext/L2) +
           D1*L2*np.exp(2*a1/L2 + 2*a1/L1) - D1*L2*np.exp(2*a_ext/L2 + 2*a1/L1) - D2*L1*np.exp(2*a1/L2) -
           D2*L1*np.exp(2*a_ext/L2) - D2*L1*np.exp(2*a1/L2 + 2*a1/L1) - D2*L1*np.exp(2*a_ext/L2 + 2*a1/L1)) + L1**2*Q
        return u1

    i = 0
    for Sa1_val in [0.2, 0.4, 2.0]:
        for Q_val in [0.5, 0.8, 1.2]:
            for Sa2_val in [0.04, 0.08]:
                u1 = analytical_solution_1(
                    X1.flatten(), D1, D2, a_ext, Sa1_val, Sa2_val, Q_val, a1
                )

                validator = PointwiseValidator(
                    nodes=nodes,
                    invar={
                        "x": X1,
                        "Sigma_a1_hat": np.full_like(X1, Sa1_val / Sigma_a1_max),
                        "Sigma_a2_hat": np.full_like(X1, Sa2_val / Sigma_a2_max),
                        "Q_hat": np.full_like(X1, Q_val / Q_max),
                    },
                    true_outvar={"u1": u1.reshape(-1, 1)},
                    batch_size=256,
                )

                domain.add_validator(
                    validator,
                    f"S1_val_Sa_{Sa1_val}_Q_{Q_val}_{i}",
                )
                i += 1

    def analytical_solution_2(X2, D1, D2, a_ext, Sigma_a1, Sigma_a2, Q, a2):
        L1 = np.sqrt(D1 / Sigma_a1)
        L2 = np.sqrt(D2 / Sigma_a2)
        u2 = D1*L1**2*L2*Q*(1 - np.exp(2*a1/L1))*np.exp(-X2/L2)*np.exp((a1 + 2*a_ext)/L2)/(-D1*L2*np.exp(2*a1/L2) +
         D1*L2*np.exp(2*a_ext/L2) + D1*L2*np.exp(2*a1*(1/L2 + 1/L1)) - D1*L2*np.exp(2*a_ext/L2 + 2*a1/L1) -
         D2*L1*np.exp(2*a1/L2) - D2*L1*np.exp(2*a_ext/L2) - D2*L1*np.exp(2*a1*(1/L2 + 1/L1)) -
         D2*L1*np.exp(2*a_ext/L2 + 2*a1/L1)) - D1*L1**2*L2*Q*(np.exp(2*a1/L1) - 1)*np.exp(a1/L2)*np.exp(X2/L2)/(D1*L2*np.exp(2*a1/L2) -
         D1*L2*np.exp(2*a_ext/L2) - D1*L2*np.exp(2*a1/L2 + 2*a1/L1) + D1*L2*np.exp(2*a_ext/L2 + 2*a1/L1) +
         D2*L1*np.exp(2*a1/L2) + D2*L1*np.exp(2*a_ext/L2) + D2*L1*np.exp(2*a1/L2 + 2*a1/L1) +
         D2*L1*np.exp(2*a_ext/L2 + 2*a1/L1))
        return u2

    i = 0
    for Sa1_val in [0.2, 0.4, 2.0]:
        for Q_val in [0.5, 0.8, 1.2]:
            for Sa2_val in [0.04, 0.08]:
                u2 = analytical_solution_2(
                    X2.flatten(), D1, D2, a_ext, Sa1_val, Sa2_val, Q_val, a2
                )

                validator = PointwiseValidator(
                    nodes=nodes,
                    invar={
                        "x": X2,
                        "Sigma_a1_hat": np.full_like(X2, Sa1_val / Sigma_a1_max),
                        "Sigma_a2_hat": np.full_like(X2, Sa2_val / Sigma_a2_max),
                        "Q_hat": np.full_like(X2, Q_val / Q_max),
                    },
                    true_outvar={"u2": u2.reshape(-1, 1)},
                    batch_size=256,
                )

                domain.add_validator(
                    validator,
                    f"S2_val_Sa_{Sa1_val}_Q_{Q_val}_{i}",
                )
                i += 1

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
