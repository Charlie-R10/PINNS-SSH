import math
import numpy as np
import sympy
from sympy import Symbol, Function, Lambda
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



# Define custom PDE in class - use physics nemo PDE class as parent
class NDequation(PDE):
    def __init__(self):
        x = Symbol("x")
        D = Symbol("D")
        Sa = Symbol("Sa")
        input_variables = {"x": x, "D": D, "Sa": Sa}
        u = Function("u")(*input_variables)

        # set equations
        L_square = D / Sa
        coef = -1/L_square
        self.equations = {}
        self.equations["neutron_diffusion_equation"] = u.diff(x, 2) + coef * u
        
# Config from physics nemo
@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:

    D_sym = Symbol("D")
    Sa_sym = Symbol("Sa")
    s0_sym = Symbol("s0")
    param_ranges = {
        s0_sym: (10.0, 20.0),
        D_sym: (0.1, 1.0),
        Sa_sym: (0.001, 0.01)
    }
    pr = Parameterization(param_ranges)
    

    ode = NDequation()

    x = Symbol("x")

    # Creating net
    custom_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("s0"), Key("D"), Key("Sa")], # All input keys parameterized
        output_keys=[Key("u")]
    )
    
    # Form nodes - one from ode and one from neural net
    nodes = ode.make_nodes() + [custom_net.make_node(name="ode_network")]

    # Defining geometry as 1D line with extrapolated length 
    a = 1.
    min_x = 0
    a_ex = a + 0.7104 * 3 * D_sym
    max_x = a_ex # extrapolated length

    line = Line1D(min_x, 2.5)
    ode_domain = Domain()

    # LHS boundary condition (uses analytical solution = 0 for loss)
    L_sym = sympy.sqrt(D_sym / Sa_sym)
    phi_0 = s0_sym * L_sym * (1 - sympy.exp(-2 * a_ex / L_sym)) / (2 * D_sym * (1 + sympy.exp(-2 * a_ex / L_sym)))


    bc_min_x = PointwiseBoundaryConstraint(nodes=nodes,
                                           geometry=line,
                                           outvar={"u": phi_0},
                                           criteria=sympy.Eq(x, min_x),
                                           batch_size=cfg.batch_size.bc_min,
                                           parameterization=pr) #only BC that contains S0, therefore only BC needs parameterizaion?
    ode_domain.add_constraint(bc_min_x, "bc_min")

    # Boundary condition that phi = 0 at RHS (extrapolated length)
    bc_max_x = PointwiseBoundaryConstraint(nodes=nodes,
                                           geometry=line,
                                           outvar={"u": 0},
                                           criteria=sympy.Eq(x, max_x),
                                           batch_size=cfg.batch_size.bc_max,
                                           parameterization=pr)
    ode_domain.add_constraint(bc_max_x, "bc_max")

    # Interior loss function for neutron diffusion equation
    interior = PointwiseInteriorConstraint(nodes=nodes,
                                           geometry=line,
                                           outvar={"neutron_diffusion_equation": 0},
                                           batch_size=cfg.batch_size.interior,
                                           parameterization=pr)
    ode_domain.add_constraint(interior, "interior")

    # Add inferencer - removed for now 
    #points = np.linspace(0, 1, 101).reshape(101, 1)
    #inferencer = PointwiseInferencer(nodes=nodes, invar={"x": points, "s0": np.full_like(points, 15.0)}, output_names=["u"], batch_size=1024, plotter=InferencerPlotter())
    #ode_domain.add_inferencer(inferencer, "inf_data")


    # Validator with calculated analytical solution (equation from Stacey)
    # Function to calculate analytical solution with parameters as inputs
    def analytical_solution(x, s0, D, a_ex, Sa):
        L = math.sqrt(D / Sa) 
        numerator = np.sinh((a_ex - x) / L)
        denominator = np.cosh(a_ex / L)
        return (s0 * L / (2 * D)) * (numerator / denominator)

    # Validator loop for s0, D and Sa - 3 values each for now as validation parameters
    for s0_val in [10, 15, 20]:
        for D_val in [0.1, 0.5, 1.0]:
            for Sa_val in [0.001, 0.005, 0.01]:
                L_val = math.sqrt(D_val / Sa_val)
                a_ex = a + 0.7104 * 3 * D_val

                # Analytical Solution calculated from inputted values
                u_true = analytical_solution(points.flatten(), s0_val, D_val, a_ex, Sa_val)
                
                # u_true = ( s0_val * L_val * (1 - np.exp(-2 * a_ex / L_val)) /(2 * D_val * (1 + np.exp(-2 * a_ex / L_val))) *(np.cosh((points - a) / L_val) / np.cosh(a_ex / L_val))
                

                # Validator to calculate error
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
