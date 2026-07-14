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
from physicsnemo.sym.domain.constraint import PointwiseConstraint
from physicsnemo.sym.geometry.parameterization import Parameterization


@physicsnemo.sym.main(config_path="conf", config_name="config_dif_rv_2r")
def run(cfg: PhysicsNeMoConfig) -> None:
    # make list of nodes to unroll graph on
    ext_lengt_bc = True
    D1 = 1.0
    Sigma_a1 = Symbol("Sigma_a1") # parametric
    D2 = 0.8
    Sigma_a2 = 0.1
    a1 = 5.0
    a2 = 10.0
    Q = Symbol("Q") # parametric
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

    param_ranges = {
        Q: (0.0, 1.0),
        Sigma_a1 : (0.0, 0.1)  #ranges set
    }
    pr = Parameterization(param_ranges)
    
    diffusion_net_u1 = instantiate_arch(
        input_keys=[Key("x"), Key("Q"), Key("Sigma_a1")],
        output_keys=[Key("u1")],
        cfg=cfg.arch.fully_connected,
    )

    diffusion_net_u2 = instantiate_arch(
        input_keys=[Key("x"), Key("Q"), Key("Sigma_a1")],
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

    # boundary conditions
    LB = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo1,
        outvar={"reflective_boundary": 0},
        batch_size=cfg.batch_size.LB,
        lambda_weighting={"reflective_boundary": 50.0},
        criteria=Eq(x, 0),
        parameterization=pr
    )
    domain.add_constraint(LB, "LB")

    IB = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo1,
        outvar={"flux_continuity": 0,
                "current_continuity": 0},
        batch_size=cfg.batch_size.IB,
        lambda_weighting={"flux_continuity": 50.0, "current_continuity": 50.0},
        criteria=Eq(x, a1),
        parameterization=pr
    )
    domain.add_constraint(IB, "IB")

    RB = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo2,
        outvar={"vacuum_boundary": 0},
        batch_size=cfg.batch_size.RB,
        lambda_weighting={"vacuum_boundary": 50.0},
        criteria=Eq(x, a_ext),
        parameterization=pr
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
        parameterization=pr
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
        parameterization=pr
    )
    domain.add_constraint(interior2, "interior2")

######################################################################################################
    
    # add validation data and analytical solutions
    deltaX = 0.01
    X1 = np.arange(0, a1, deltaX)
    X2 = np.arange(a1, a_ext, deltaX)

    #Q = Q/D1  #should it change now Q symbolic?

    # Midpoint values to solve at 0, a1/2 and a1 as well as then a1, (a1 + a_ext)/2, a_ext
    all_x_u1, all_u1_vals, all_Q_u1, all_Sigma_a1_u1 = [], [], [], []
    all_x_u2, all_u2_vals, all_Q_u2, all_Sigma_a1_u2 = [], [], [], []


    ###### Analytical solutions

    def analytical_solution_1(X1, D1, D2, a_ext, Sigma_a1, Sigma_a2, Q, a1):
        Q=Q/D1 # changes for each Q
        L1 = np.sqrt(D1/Sigma_a1)
        L2 = np.sqrt(D2/Sigma_a2)
        u1 = D2*L1**3*Q*(np.exp(2*a1/L2) + np.exp(2*a_ext/L2))*np.exp(a1/L1)*np.exp(X1/L1)/(-D1*L2*np.exp(2*a1/L2) +
               D1*L2*np.exp(2*a_ext/L2) + D1*L2*np.exp(2*a1/L2 + 2*a1/L1) - D1*L2*np.exp(2*a_ext/L2 + 2*a1/L1) -
               D2*L1*np.exp(2*a1/L2) - D2*L1*np.exp(2*a_ext/L2) - D2*L1*np.exp(2*a1/L2 + 2*a1/L1) -
               D2*L1*np.exp(2*a_ext/L2 + 2*a1/L1)) + D2*L1**3*Q*(np.exp(2*a1/L2) +
               np.exp(2*a_ext/L2))*np.exp(a1/L1)*np.exp(-X1/L1)/(-D1*L2*np.exp(2*a1/L2) + D1*L2*np.exp(2*a_ext/L2) +
               D1*L2*np.exp(2*a1/L2 + 2*a1/L1) - D1*L2*np.exp(2*a_ext/L2 + 2*a1/L1) - D2*L1*np.exp(2*a1/L2) -
               D2*L1*np.exp(2*a_ext/L2) - D2*L1*np.exp(2*a1/L2 + 2*a1/L1) - D2*L1*np.exp(2*a_ext/L2 + 2*a1/L1)) + L1**2*Q
        return u1

    j = 0
    # Cycle through specified vals for Q and Sigma_a1
    for Q_val in [0.2, 0.5, 0.7]: 
        for Sigma_a1_val in [0.02, 0.05, 0.08]:
            u1 = analytical_solution_1(
                        X1.flatten(), D1, D2, a_ext, Sigma_a1_val, Sigma_a2, Q_val, a1
                    )
            validator = PointwiseValidator(
                nodes=nodes, invar={
                    "x": X1.reshape(-1,1), "Q": np.full((len(X1),1), Q_val), "Sigma_a1": np.full((len(X1),1), Sigma_a1_val)
                },
                true_outvar={"u1":u1.reshape(-1,1)}, #potentially u1.reshape(-1, 1)
                batch_size=128
            )
            domain.add_validator(validator, f"validator_1_{j}")
            j+=1

            # Anchors / midpoints fof u1 (only calcs points at LHS, center/crossover and RHS)
            x_pts = np.array([0.0, a1/4, a1/2, 3*a1/4, a1])
            u1_pts = analytical_solution_1(x_pts, D1, D2, a_ext, Sigma_a1_val, Sigma_a2, Q_val, a1)
            for x_pt, u_pt in zip(x_pts, u1_pts):
                all_x_u1.append([x_pt])
                all_Q_u1.append([Q_val])
                all_Sigma_a1_u1.append([Sigma_a1_val])
                all_u1_vals.append([float(u_pt)])


    # Repeat all for U2 (second reigon analytical solution)

    def analytical_solution_2(X2, D1, D2, a_ext, Sigma_a1, Sigma_a2, Q, a1):
        Q=Q/D1 # changes for each Q
        L1 = np.sqrt(D1/Sigma_a1)
        L2 = np.sqrt(D2/Sigma_a2)
        u2 = D1*L1**2*L2*Q*(1 - np.exp(2*a1/L1))*np.exp(-X2/L2)*np.exp((a1 + 2*a_ext)/L2)/(-D1*L2*np.exp(2*a1/L2) +
             D1*L2*np.exp(2*a_ext/L2) + D1*L2*np.exp(2*a1*(1/L2 + 1/L1)) - D1*L2*np.exp(2*a_ext/L2 + 2*a1/L1) -
             D2*L1*np.exp(2*a1/L2) - D2*L1*np.exp(2*a_ext/L2) - D2*L1*np.exp(2*a1*(1/L2 + 1/L1)) -
             D2*L1*np.exp(2*a_ext/L2 + 2*a1/L1)) - D1*L1**2*L2*Q*(np.exp(2*a1/L1) - 1)*np.exp(a1/L2)*np.exp(X2/L2)/(D1*L2*np.exp(2*a1/L2) -
             D1*L2*np.exp(2*a_ext/L2) - D1*L2*np.exp(2*a1/L2 + 2*a1/L1) + D1*L2*np.exp(2*a_ext/L2 + 2*a1/L1) +
             D2*L1*np.exp(2*a1/L2) + D2*L1*np.exp(2*a_ext/L2) + D2*L1*np.exp(2*a1/L2 + 2*a1/L1) +
             D2*L1*np.exp(2*a_ext/L2 + 2*a1/L1))
        return u2

    i = 0
    for Q_val in [0.2, 0.5, 0.7]:
        for Sigma_a1_val in [0.02, 0.05, 0.08]:
            u2 = analytical_solution_2(
                        X2.flatten(), D1, D2, a_ext, Sigma_a1_val, Sigma_a2, Q_val, a1
                    )
    
            validator = PointwiseValidator(
                nodes=nodes, invar={
                    "x": X2.reshape(-1,1), "Q": np.full((len(X2),1), Q_val), "Sigma_a1": np.full((len(X2),1), Sigma_a1_val)
                },
                true_outvar={"u2":u2.reshape(-1,1)}, #potentially u2.reshape(-1, 1)
                batch_size=128
            )
            domain.add_validator(validator, f"validator_2_{i}")
            i+=1

            x_pts = np.array([a1, a1 + (a_ext-a1)/3, a1 + 2*(a_ext-a1)/3, a_ext])
            u2_pts = analytical_solution_2(x_pts, D1, D2, a_ext, Sigma_a1_val, Sigma_a2, Q_val, a1)
            for x_pt, u_pt in zip(x_pts, u2_pts):
                all_x_u2.append([x_pt])
                all_Q_u2.append([Q_val])
                all_Sigma_a1_u2.append([Sigma_a1_val])
                all_u2_vals.append([float(u_pt)])

    data_constraint_u1 = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar={"x": np.array(all_x_u1), "Q": np.array(all_Q_u1), "Sigma_a1": np.array(all_Sigma_a1_u1)},
        outvar={"u1": np.array(all_u1_vals)},
        batch_size=len(all_u1_vals),
        lambda_weighting={"u1": 100.0}
    )

    domain.add_constraint(data_constraint_u1, "anchor_u1")

    data_constraint_u2 = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar={"x": np.array(all_x_u2), "Q": np.array(all_Q_u2), "Sigma_a1": np.array(all_Sigma_a1_u2)},
        outvar={"u2": np.array(all_u2_vals)},
        batch_size=len(all_u2_vals),
        lambda_weighting={"u2": 100.0}
    )
    domain.add_constraint(data_constraint_u2, "anchor_u2")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()

if __name__ == "__main__":
    run()
