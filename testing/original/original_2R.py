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
from diffusion_equation import DiffusionEquation1D, VacuumBoundary, ReflectiveBoundary, InterfaceDiffusion1D
from physicsnemo.sym.eq.pdes.diffusion import Diffusion


@physicsnemo.sym.main(config_path="conf", config_name="config_dif_rv_2r")
def run(cfg: PhysicsNeMoConfig) -> None:
    # make list of nodes to unroll graph on
    ext_lengt_bc = True
    D1 = 1.0
    Sigma_a1 = 0.01
    D2 = 0.8
    Sigma_a2 = 0.1
    a1 = 5.0
    a2 = 10.0
    Q = 1.0
    if ext_lengt_bc:
        a_ext = a2 + 3 * 0.7104 * D2
    else:
        a_ext = a2
    # a_ext = 2 * a

    de1 = DiffusionEquation1D(u="u1", D=D1, Sigma_a=Sigma_a1, Q=Q)
    de2 = DiffusionEquation1D(u="u2", D=D2, Sigma_a=Sigma_a2, Q=0)
    de_in = InterfaceDiffusion1D(u1="u1", u2="u2", D1=D1, D2=D2)
    vb = VacuumBoundary(u="u2", D=D2, extrapolated_length=ext_lengt_bc)
    rb = ReflectiveBoundary(u="u1", D=D1)

    diffusion_net_u1 = instantiate_arch(
        input_keys=[Key("x")],
        output_keys=[Key("u1")],
        cfg=cfg.arch.fully_connected,
    )

    diffusion_net_u2 = instantiate_arch(
        input_keys=[Key("x")],
        output_keys=[Key("u2")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = (
            de1.make_nodes() +
            de2.make_nodes() +
            de_in.make_nodes() +
            vb.make_nodes() +
            rb.make_nodes() +
            [diffusion_net_u1.make_node(name="diffusion_network_u1")] +
            [diffusion_net_u2.make_node(name="diffusion_network_u2")]
    )

    # add constraints to solver
    # make geometry
    x = Symbol("x")
    geo1 = Line1D(point_1=0,  point_2=a1)
    geo2 = Line1D(point_1=a1, point_2=a_ext)

    # make domain
    domain = Domain()

    # boundary condition
    LB = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo1,
        outvar={"reflective_boundary": 0},
        batch_size=cfg.batch_size.LB,
        lambda_weighting={"reflective_boundary": 100.0},
        criteria=Eq(x, 0)
    )
    domain.add_constraint(LB, "LB")

    IB = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo1,
        outvar={"flux_continuity": 0,
                "current_continuity": 0},
        batch_size=cfg.batch_size.IB,
        lambda_weighting={"flux_continuity": 100.0, "current_continuity": 100.0},
        criteria=Eq(x, a1)
    )
    domain.add_constraint(IB, "IB")

    RB = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo2,
        outvar={"vacuum_boundary": 0},
        batch_size=cfg.batch_size.RB,
        lambda_weighting={"vacuum_boundary": 100.0},
        criteria=Eq(x, a_ext)
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
    )
    domain.add_constraint(interior2, "interior2")

    # add validation data
    deltaX = 0.01
    X1 = np.arange(0, a1, deltaX)
    X2 = np.arange(a1, a_ext, deltaX)
    L1 = np.sqrt(D1/Sigma_a1)
    L2 = np.sqrt(D2/Sigma_a2)

    Q = Q/D1

    u1 = D2*L1**3*Q*(np.exp(2*a1/L2) + np.exp(2*a_ext/L2))*np.exp(a1/L1)*np.exp(X1/L1)/(-D1*L2*np.exp(2*a1/L2) +
           D1*L2*np.exp(2*a_ext/L2) + D1*L2*np.exp(2*a1/L2 + 2*a1/L1) - D1*L2*np.exp(2*a_ext/L2 + 2*a1/L1) -
           D2*L1*np.exp(2*a1/L2) - D2*L1*np.exp(2*a_ext/L2) - D2*L1*np.exp(2*a1/L2 + 2*a1/L1) -
           D2*L1*np.exp(2*a_ext/L2 + 2*a1/L1)) + D2*L1**3*Q*(np.exp(2*a1/L2) +
           np.exp(2*a_ext/L2))*np.exp(a1/L1)*np.exp(-X1/L1)/(-D1*L2*np.exp(2*a1/L2) + D1*L2*np.exp(2*a_ext/L2) +
           D1*L2*np.exp(2*a1/L2 + 2*a1/L1) - D1*L2*np.exp(2*a_ext/L2 + 2*a1/L1) - D2*L1*np.exp(2*a1/L2) -
           D2*L1*np.exp(2*a_ext/L2) - D2*L1*np.exp(2*a1/L2 + 2*a1/L1) - D2*L1*np.exp(2*a_ext/L2 + 2*a1/L1)) + L1**2*Q
    # a_l = -a_ext
    # a_r = a_ext
    # if ext_lengt_bc:
    #     u = -B**2 * Q * np.exp(x/B)/(2*np.cosh(a_r/B)) + B**2*Q + (B**2*Q*np.exp(a_r/B)/(2*np.cosh(a_r/B))
    #                                                                - B**2*Q)*np.exp(a_r/B)*np.exp(-x/B)
    #     # u = (B**2*Q - B**2*Q*np.exp(X/B)/(np.exp(a_l/B) + np.exp(a_r/B)) +
    #     #      (-B**2*Q + B**2*Q*np.exp(a_r/B)/(np.exp(a_l/B) + np.exp(a_r/B)))*np.exp(a_r/B)*np.exp(-X/B))
    #     # u = B * Q * (np.exp(2 * a_ext/B) - np.exp(2 * X/B)) * np.exp(-X/B)/(2 * D * (np.exp(2 * a_ext/B) + 1))
    # else:
    #     u = B * Q * (4 * D * np.exp(2*a_ext/B) + B * np.exp(2 * a_ext/B) +
    #                  (4 * D - B) * np.exp(2 * X/B)) * np.exp(-X/B)/(2 * D * (4 * D * np.exp(2*a_ext/B)
    #                                                                          - 4 * D + B * np.exp(2 * a_ext/B) + B))
    # u = (0.5 * Q * B/D) * (np.exp(-X/B) - np.exp(-(a_ext - X)/B))/(1 + np.exp(-a_ext/B))
    # X = np.meshgrid(x)
    X1 = np.expand_dims(X1.flatten(), axis=-1)
    u1 = np.expand_dims(u1.flatten(), axis=-1)
    print("*************************************")
    print(X1.shape)
    print("*************************************")
    # T = np.expand_dims(T.flatten(), axis=-1)
    # u = np.zeros(shape=X.shape)
    invar_numpy = {"x": X1}
    outvar_numpy = {"u1": u1}
    validator = PointwiseValidator(
        nodes=nodes, invar=invar_numpy, true_outvar=outvar_numpy, batch_size=128
    )
    domain.add_validator(validator, "validator_1")

    u2 = D1*L1**2*L2*Q*(1 - np.exp(2*a1/L1))*np.exp(-X2/L2)*np.exp((a1 + 2*a_ext)/L2)/(-D1*L2*np.exp(2*a1/L2) +
         D1*L2*np.exp(2*a_ext/L2) + D1*L2*np.exp(2*a1*(1/L2 + 1/L1)) - D1*L2*np.exp(2*a_ext/L2 + 2*a1/L1) -
         D2*L1*np.exp(2*a1/L2) - D2*L1*np.exp(2*a_ext/L2) - D2*L1*np.exp(2*a1*(1/L2 + 1/L1)) -
         D2*L1*np.exp(2*a_ext/L2 + 2*a1/L1)) - D1*L1**2*L2*Q*(np.exp(2*a1/L1) - 1)*np.exp(a1/L2)*np.exp(X2/L2)/(D1*L2*np.exp(2*a1/L2) -
         D1*L2*np.exp(2*a_ext/L2) - D1*L2*np.exp(2*a1/L2 + 2*a1/L1) + D1*L2*np.exp(2*a_ext/L2 + 2*a1/L1) +
         D2*L1*np.exp(2*a1/L2) + D2*L1*np.exp(2*a_ext/L2) + D2*L1*np.exp(2*a1/L2 + 2*a1/L1) +
         D2*L1*np.exp(2*a_ext/L2 + 2*a1/L1))
    X2 = np.expand_dims(X2.flatten(), axis=-1)
    u2 = np.expand_dims(u2.flatten(), axis=-1)
    invar_numpy = {"x": X2}
    outvar_numpy = {"u2": u2}
    validator = PointwiseValidator(
        nodes=nodes, invar=invar_numpy, true_outvar=outvar_numpy, batch_size=128
    )
    domain.add_validator(validator, "validator_2")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()

if __name__ == "__main__":
    run()
