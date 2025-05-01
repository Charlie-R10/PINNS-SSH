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
import random

class NeutronDiffusionNonMult1D(PDE):
    def __init__(self, D, Sa, S0):
        x = Symbol("x")

        input_variables = {"x": x}

        u = Function("u")(*input_variables)

        # set equations
        L_square = D / Sa
        coef = -1/L_square
        self.equations = {}
        # self.equations["fick"] = (0.25 * u - 0.5 * D * u.diff(x))
        self.equations["custom_pde"] = (u.diff(x, 2) + coef * u)
        self.equations["a_bc"] = (S0/2) + (D*(u.diff(x,1)))



@modulus.sym.main(config_path="ode_conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    D = random.randint(1, 20)
    Sa = random.uniform(0.005, 0.1)
    S0 = random.uniform(0, 10)
    a = random.randint(1, 40)


    print(f"D = {D}")
    print(f"Sa = {Sa}")
    print(f"S0 = {S0}")
    print(f"a = {a}")
    L_square = D / Sa
    L = math.sqrt(L_square)

    ode = NeutronDiffusionNonMult1D(D, Sa, S0)

    x = Symbol("x")

    # Creating net
    custom_net = instantiate_arch(
        input_keys=[Key("x")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = ode.make_nodes() + [custom_net.make_node(name="ode_network")]

    # Defining geometry
    a_ex = a + 0.7104 * 3 * D
    min_x = 0
    max_x = a_ex
    line = Line1D(min_x, max_x)
    ode_domain = Domain()

    # Boundary condition
    phi_0 = S0 * L * (1 - math.exp(-2 * a_ex / L)) / (2 * D * (1 + math.exp(-2 * a_ex / L)))
    bc_min_x = PointwiseBoundaryConstraint(nodes=nodes,
                                           geometry=line,
                                           outvar={"u": phi_0},
                                           criteria=sympy.Eq(x, min_x),
                                           batch_size=cfg.batch_size.bc_min)
    ode_domain.add_constraint(bc_min_x, "bc_min")

    bc_max_x = PointwiseBoundaryConstraint(nodes=nodes,
                                           geometry=line,
                                           outvar={"u": 0},
                                           criteria=sympy.Eq(x, max_x),
                                           batch_size=cfg.batch_size.bc_max)
    ode_domain.add_constraint(bc_max_x, "bc_max")


  #  bc_lhs = PointwiseBoundaryConstraint(nodes=nodes,
   #                                        geometry=line,
    #                                       outvar={"a_bc": 0},
    #                                       criteria=sympy.Eq(x, min_x),
    #                                       batch_size=cfg.batch_size.bc_max)
    #ode_domain.add_constraint(bc_lhs, "bc_lhs")


    # Interior
    interior = PointwiseInteriorConstraint(nodes=nodes,
                                           geometry=line,
                                           outvar={"custom_pde": 0},
                                           batch_size=cfg.batch_size.interior)
    ode_domain.add_constraint(interior, "interior")

    # Add inferencer
    points = np.linspace(0, 1, 101).reshape(101, 1)
    inferencer = PointwiseInferencer(nodes=nodes, invar={"x": points}, output_names=["u"], batch_size=1024, plotter=InferencerPlotter())
    ode_domain.add_inferencer(inferencer, "inf_data")


    # Add validation data
    #dx = 0.01
    #X = np.arange(min_x, max_x, dx)
    # # X = np.meshgrid(x)
    # # X = np.expand_dims(X.flatten(), axis=-1)
    #u = -2 * np.cos(2*X) + 10 * np.sin(2*X)
    #invar_numpy = {"x": X}
    #outvar_numpy = {"u": u}
    #validator = PointwiseValidator(nodes=nodes, invar=invar_numpy, true_outvar=outvar_numpy, batch_size=128)
    #ode_domain.add_validator(validator)

    #phi = S0 * L * (math.exp(-x/L) - math.exp(- a_ex / L)) / (2 * D * (1 + math.exp(-2 * a_ex / L)))



    # make solver
    slv = Solver(cfg, ode_domain)

    # start solver
    slv.solve()

if __name__ == '__main__':
    run()
