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



# Define custom PDE in class - uses physics nemo PDE class as parent
class NDequation(PDE):
    def __init__(self):
        x = Symbol("x")
        Sa = Symbol("Sa")
        input_variables = {"x": x, "Sa": Sa}
        u = Function("u")(*input_variables)

        # set equations
        D = 1 / (3 * 1.5)
        L_square = D / Sa
        coef = -1/L_square
        self.equations = {}
        self.equations["neutron_diffusion_equation"] = u.diff(x, 2) + coef * u
        
# Config from physics nemo
@physicsnemo.sym.main(config_path="conf", config_name="config2")
def run(cfg: PhysicsNeMoConfig) -> None:

    D = 1 / (3 * 1.5)
    Sa_sym = Symbol("Sa")
    s0_sym = Symbol("s0")
    param_ranges = {
        s0_sym: (10.0, 20.0),
        Sa_sym: (0.001, 0.01)
    }
    pr = Parameterization(param_ranges)
    

    ode = NDequation()

    x = Symbol("x")

    # Creating net
    width = cfg.hparams.width
    depth = cfg.hparams.depth
    activation = cfg.hparams.activation

    custom_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("s0"), Key("Sa")],
        output_keys=[Key("u")],
        layer_size=width,
        nr_layers=depth,
        activation_fn=activation
    )
    
    # Form nodes - one from ode and one from neural net
    nodes = ode.make_nodes() + [custom_net.make_node(name="ode_network")]

    # Defining geometry as 1D line with extrapolated length 
    a = 1.
    min_x = 0
    a_ex = a + 0.7104 * 3 * D
    max_x = a_ex/2 # extrapolated length


    line = Line1D(min_x, max_x)
    ode_domain = Domain()

    # LHS boundary condition (uses analytical solution = 0 for loss)
    L_sym = sympy.sqrt(D / Sa_sym)
    numerator_phi0 = sympy.sinh((a_ex) / (2 * L_sym))
    denominator_phi0 = sympy.cosh(a_ex / (2 * L_sym))
    phi_0 = ((s0_sym * L_sym) / (2 * D)) * (numerator_phi0 / denominator_phi0)
    bc_min_x = PointwiseBoundaryConstraint(nodes=nodes,
                                           geometry=line,
                                           outvar={"u": phi_0},
                                           criteria=sympy.Eq(x, 0),
                                           batch_size=cfg.batch_size.bc_min,
                                           parameterization=pr) 
    ode_domain.add_constraint(bc_min_x, "bc_min")

    # Boundary condition that phi = 0 at a_ex/2 (extrapolated length/2)
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

    points = np.linspace(0, max_x, 101).reshape(101, 1)


    # Validator with calculated analytical solution (equation from Stacey)
    # Function to calculate analytical solution with parameters as inputs
    def analytical_solution(x, s0, D, a_ex, Sa):
        L = math.sqrt(D / Sa) 
        numerator = np.sinh((a_ex - 2*x) / (2 * L))
        denominator = np.cosh(a_ex / (2 * L))
        return (s0 * L / (2 * D)) * (numerator / denominator)

    # Validator loop for s0, D and Sa - 3 values each for now as validation parameters
    for s0_val in [10, 12, 15, 17, 20]:
          for Sa_val in [0.001, 0.005, 0.01]:
                L_val = math.sqrt(D / Sa_val)
                a_ex = a + 0.7104 * 3 * D

                # Analytical Solution calculated from inputted values
                u_true = analytical_solution(points.flatten(), s0_val, D, a_ex, Sa_val)
                

                # Validator to calculate error
                validator = PointwiseValidator(
                    nodes=nodes,
                    invar={"x": points, "s0": np.full_like(points, s0_val), "Sa": np.full_like(points, Sa_val)},
                    true_outvar={"u": u_true.reshape(-1, 1)},
                    batch_size=1024)

                ode_domain.add_validator(validator, f"validator_s0_{s0_val}_Sa_{Sa_val}")

    # make solver
    slv = Solver(cfg, ode_domain)

    # start solver
    slv.solve()

      rel_l2_all = []
    for name, validator in slv.domain.validators.items():
        errs = validator.get_errors()
        if "relative_l2" in errs:
            rel_l2_all.append(float(errs["relative_l2"]))

    # fall back 
    val_metric = float(np.mean(rel_l2_all)) if rel_l2_all else 1e9

    metric_path.parent.mkdir(parents=True, exist_ok=True)
    with metric_path.open("w") as f:
        json.dump({"val_error": val_metric}, f)
    print(f"[Dual_parametric] wrote metric {val_metric:.4e} to {metric_path}")

if __name__ == '__main__':
    run()
