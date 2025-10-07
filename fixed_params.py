import math
import numpy as np
import sympy
from sympy import Symbol, Function
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
        coef = -1/L_square
        self.equations = {}
        # self.equations["fick"] = (0.25 * u - 0.5 * D * u.diff(x))
        self.equations["custom_pde"] = (u.diff(x, 2) + coef * u)

# Config from PhysicsNeMo
@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:

    D = 1 / (3 * 1.5)
    Sa = 18
    S0 = 1
    L_square = D / Sa
    L = math.sqrt(L_square)

    ode = NeutronDiffusionNonMult1D(D, Sa)

    x = Symbol("x")

    # Creating net
    custom_net = instantiate_arch(
        input_keys=[Key("x")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = ode.make_nodes() + [custom_net.make_node(name="ode_network")]

    # Defining geometry
    a = 1.
    a_ex = a + 0.7104 * 3 * D
    min_x = 0
    max_x = a_ex

    line = Line1D(min_x, max_x)
    ode_domain = Domain()

    numerator_phi0 = numpy.sinh((a_ex) / (2 * L))
    denominator_phi0 = numpy.cosh(a_ex / (2 * L))
    phi_0 = ((s0 * L) / (2 * D)) * (numerator_phi0 / denominator_phi0)

    # Boundary condition at x = 0
    bc_min_x = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=line,
        outvar={"u": phi_0},
        criteria=sympy.Eq(x, 0),
        batch_size=cfg.batch_size.bc_min
    )
    ode_domain.add_constraint(bc_min_x, "bc_min")

    # Boundary condition at x = a_ex/2
    bc_max_x = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=line,
        outvar={"u": 0},
        criteria=sympy.Eq(x, max_x),
        batch_size=cfg.batch_size.bc_max
    )
    ode_domain.add_constraint(bc_max_x, "bc_max")

    # Interior PDE residual
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=line,
        outvar={"neutron_diffusion_equation": 0},
        batch_size=cfg.batch_size.interior
    )
    ode_domain.add_constraint(interior, "interior")

    # Add inferencer
    points = np.linspace(0, 1, 101).reshape(101, 1)
    inferencer = PointwiseInferencer(nodes=nodes, invar={"x": points}, output_names=["u"], batch_size=1024, plotter=InferencerPlotter())
    ode_domain.add_inferencer(inferencer, "inf_data")


    # make solver
    slv = Solver(cfg, ode_domain)

    # start solver
    slv.solve()

if __name__ == '__main__':
    run()
    slv = Solver(cfg, ode_domain)
    slv.solve()


if __name__ == '__main__':
    run()
