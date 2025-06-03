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



# Setup of class with 1d NDE
class NeutronDiffusionNonMult1D(PDE):
    def __init__(self, D, Sa):
        x = Symbol("x")
        D = Symbol("D")
        Sa = Symbol("Sa")
        input_variables = {"x": x, "D": D, "Sa": Sa}
        u = Function("u")(*input_variables)

        # set equations
        L_square = D / Sa
        coef = -1/L_square
        self.equations = {}
        # self.equations["fick"] = (0.25 * u - 0.5 * D * u.diff(x))
        self.equations["custom_pde"] = (u.diff(x, 2) + coef * u)

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
    L_square = D / Sa
    L = math.sqrt(L_square)

    ode = NeutronDiffusionNonMult1D(D, Sa)

    x = Symbol("x")

    # Creating net
    custom_net = instantiate_arch(
        input_keys=[Key("x"), Key("s0"), Key("D"), Key("Sa")], # All input keys parameterized
        output_keys=[Key("u")]
    )

    nodes = ode.make_nodes() + [custom_net.make_node(name="ode_network")]

    # Defining geometry
    a = 1.
    a_ex = a + 0.7104 * 3 * D
    min_x = 0
    max_x = a_ex # extrapolated length

    line = Line1D(min_x, max_x)
    ode_domain = Domain()

# Boundary condition of analytical solution at LHS
    L_sym = sympy.sqrt(D_sym / Sa_sym)
    a_ex = a + 0.7104 * 3 * D_sym
    phi_0 = s0_sym * L_sym * (1 - sympy.exp(-2 * a_ex / L_sym)) / (2 * D_sym * (1 + sympy.exp(-2 * a_ex / L_sym)))
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

    # Add inferencer
    points = np.linspace(0, 1, 101).reshape(101, 1)
    inferencer = PointwiseInferencer(nodes=nodes, invar={"x": points, "s0": np.full_like(points, 15.0)}, output_names=["u"], batch_size=1024, plotter=InferencerPlotter())
    ode_domain.add_inferencer(inferencer, "inf_data")


    # Add validator for analytical solution
    # Calc analytical solution first
    def analytical_solution(x, s0, D, a_ex):
        L = math.sqrt(D / 0.005)  # or pass Sa as a variable if needed
        numerator = np.sinh((a_ex - x) / L)
        denominator = np.cosh(a_ex / L)
        return (s0 * L / (2 * D)) * (numerator / denominator)

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

    # make solver
    slv = Solver(cfg, ode_domain)

    # start solver
    slv.solve()

if __name__ == '__main__':
    run()
