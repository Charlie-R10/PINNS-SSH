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
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.node import Node
from physicsnemo.sym.geometry.parameterization import Parameterization
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.models.fourier_net import FourierNetArch

# Define custom PDE in class - use physics nemo PDE class as parent
class NDequation(PDE):
    def __init__(self, D, Sa):
        x = Symbol("x")
        u = Function("u")(x)

        # set equations
        L_square = D / Sa
        coef = -1 / L_square
        self.equations = {}
        self.equations["neutron_diffusion_equation"] = u.diff(x, 2) + coef * u

# Config from physics nemo
@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:


    # define parameters for D and Sa (fixed)
    D = 1 / (3 * 1.5)
    Sa = 0.005
    L_square = D / Sa
    L = math.sqrt(L_square)

    # Create parametric range for S0
    s0_sym = Symbol("s0")
    param_ranges = {s0_sym: (10.0, 20.0)}
    pr = Parameterization(param_ranges) 

    # Create ode equation from above class
    ode = NDequation(D, Sa)
    x = Symbol("x")

    # Create network
    custom_net = FourierNetArch(
      input_keys=[Key("x"), Key("s0")],
      output_keys=[Key("u")],
      frequencies=[1.0, 2.0, 4.0, 8.0],  # You can tune this
      layer_size=128,
      nr_layers=8,
      activation_fn="tanh"
    )

    # Form nodes - one from ode and one from neural net
    nodes = ode.make_nodes() + [custom_net.make_node(name="ode_network")]

    # Defining geometry as 1D line with extrapolated length 
    a = 1.
    a_ex = a + 0.7104 * 3 * D
    min_x = 0
    max_x = a_ex
    line = Line1D(min_x, max_x)
    ode_domain = Domain()

    # LHS boundary condition (uses analytical solution = 0 for loss)
    phi_0 = s0_sym * L * (1 - math.exp(-2 * a_ex / L)) / (2 * D * (1 + math.exp(-2 * a_ex / L)))
    bc_min_x = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=line,
        outvar={"u": phi_0},
        criteria=sympy.Eq(x, min_x),
        batch_size=cfg.batch_size.bc_min,
        parameterization=pr
    )
    ode_domain.add_constraint(bc_min_x, "bc_min")

    # Boundary condition that phi = 0 at RHS (extrapolated length)
    bc_max_x = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=line,
        outvar={"u": 0},
        criteria=sympy.Eq(x, max_x),
        batch_size=cfg.batch_size.bc_max,
        parameterization=pr
    )
    ode_domain.add_constraint(bc_max_x, "bc_max")

    # Interior loss function for neutron diffusion equation
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=line,
        outvar={"neutron_diffusion_equation": 0},
        batch_size=cfg.batch_size.interior,
        parameterization=pr
    )
    ode_domain.add_constraint(interior, "interior")

    # Inferencer setup
    points = np.linspace(0, 1, 101).reshape(-1, 1)
    inferencer = PointwiseInferencer(
        nodes=nodes,
        invar={"x": points, "s0": np.full_like(points, 15.0)},
        output_names=["u"],
        batch_size=1024
    )
    ode_domain.add_inferencer(inferencer, "inf_data")

    # Validator with calculated analytical solution (equation from Stacey)
    # Function to calculate analytical solution with parameters as inputs
    def analytical_solution(x, s0, D, a_ex):
        L = math.sqrt(D / 0.005)
        numerator = np.sinh((a_ex - x) / L)
        denominator = np.cosh(a_ex / L)
        return (s0 * L / (2 * D)) * (numerator / denominator)

    # Range of s0 values as validators inside loop
    s0_values = np.arange(10, 20.01, 0.5)
    for s0_val in s0_values:
        s0_array = np.full_like(points, s0_val)
        u_true = analytical_solution(points.flatten(), s0_val, D, a_ex) # Uses function to calc analytical phi

        # Validator to calculate error
        validator = PointwiseValidator(
            nodes=nodes,
            invar={"x": points, "s0": s0_array},
            true_outvar={"u": u_true.reshape(-1, 1)},
            batch_size=1024
        )
        ode_domain.add_validator(validator, f"validator_s0_{s0_val:.1f}") # Outputs different vtp files for s0 values (named)

    # Run solver
    solver = Solver(cfg, ode_domain)
    solver.solve(lbfgs=True)

if __name__ == '__main__':
    run()
