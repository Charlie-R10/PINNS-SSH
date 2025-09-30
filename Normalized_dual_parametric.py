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


# Define custom PDE 
class NDequation(PDE):
    def __init__(self):
        x = Symbol("x")
        Sa = Symbol("Sa")
        input_variables = {"x": x, "Sa": Sa}
        u = Function("u")(*input_variables)

        # set equations
        D = 1 / (3 * 1.5)
        L_square = D / Sa
        coef = -1 / L_square
        self.equations = {}
        self.equations["neutron_diffusion_equation"] = u.diff(x, 2) + coef * u


# Config from PhysicsNeMo
@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:

    D = 1 / (3 * 1.5)
    Sa_sym = Symbol("Sa")
    s0_sym = Symbol("s0")
    param_ranges = {
        s0_sym: (0, 10),
        Sa_sym: (0, 10)
    }
    pr = Parameterization(param_ranges)

    ode = NDequation()
    x = Symbol("x")

    # Geometry
    a = 1.
    min_x = 0
    a_ex = a + 0.7104 * 3 * D
    max_x = a_ex / 2
    line = Line1D(min_x, max_x)
    ode_domain = Domain()

    # -------------------------
    # Create neural network
    # -------------------------
    custom_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("s0"), Key("Sa")],
        output_keys=[Key("u")]  # NN predicting normalized flux not normal
    )

    # Input transform (normalize inputs)
    def input_transform(invar):
        invar_new = {}
        invar_new["x"] = invar["x"] / max_x         # map x → [0,1]
        invar_new["s0"] = invar["s0"] / 10        # map s0 → [0,1]
        invar_new["Sa"] = invar["Sa"] / 10       # map Sa → [0,1]
        return invar_new

    # Output transform (rescale back to dimensional)
    def output_transform(invar, outvar):
        D_val = 1 / (3 * 1.5)
        Sa_val = invar["Sa"] * 10    # undo normalization - mapping
        s0_val = invar["s0"] * 10
        L_val = sympy.sqrt(D_val / Sa_val)
        phi_ref = (s0_val * L_val) / (2 * D_val)
        outvar_new = {}
        outvar_new["u"] = outvar["u"] * phi_ref
        return outvar_new

    custom_net.input_transform = input_transform
    custom_net.output_transform = output_transform


    # Check normalization working as it should
    print("Check input normalization:")
    sample = {"x": np.array([0.0, max_x]), "s0": np.array([0, 20]), "Sa": np.array([0, 20])}
    print(input_transform(sample))


    # Form nodes (PDE nodes + network node)
    nodes = ode.make_nodes() + [custom_net.make_node(name="ode_network")]

    # -------------------------
    # Constraints
    # -------------------------
    L_sym = sympy.sqrt(D / Sa_sym)
    numerator_phi0 = sympy.sinh((a_ex) / (2 * L_sym))
    denominator_phi0 = sympy.cosh(a_ex / (2 * L_sym))
    phi_0 = ((s0_sym * L_sym) / (2 * D)) * (numerator_phi0 / denominator_phi0)

    # Boundary condition at x = 0
    bc_min_x = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=line,
        outvar={"u": phi_0},
        criteria=sympy.Eq(x, 0),
        batch_size=cfg.batch_size.bc_min,
        parameterization=pr
    )
    ode_domain.add_constraint(bc_min_x, "bc_min")

    # Boundary condition at x = a_ex/2
    bc_max_x = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=line,
        outvar={"u": 0},
        criteria=sympy.Eq(x, max_x),
        batch_size=cfg.batch_size.bc_max,
        parameterization=pr
    )
    ode_domain.add_constraint(bc_max_x, "bc_max")

    # Interior PDE residual
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=line,
        outvar={"neutron_diffusion_equation": 0},
        batch_size=cfg.batch_size.interior,
        parameterization=pr
    )
    ode_domain.add_constraint(interior, "interior")

    # -------------------------
    # Validators
    # -------------------------
    points = np.linspace(0, max_x, 101).reshape(101, 1)

    def analytical_solution(x, s0, D, a_ex, Sa):
        L = math.sqrt(D / Sa)
        numerator = np.sinh((a_ex - 2 * x) / (2 * L))
        denominator = np.cosh(a_ex / (2 * L))
        return (s0 * L / (2 * D)) * (numerator / denominator)

    i = 0
    for s0_val in [1, 4, 7, 9]:
        for Sa_val in [1, 4, 7, 9]:
            L_val = math.sqrt(D / Sa_val)
            a_ex = a + 0.7104 * 3 * D

            u_true = analytical_solution(points.flatten(), s0_val, D, a_ex, Sa_val)

            validator = PointwiseValidator(
                nodes=nodes,
                invar={"x": points,
                       "s0": np.full_like(points, s0_val),
                       "Sa": np.full_like(points, Sa_val)},
                true_outvar={"u": u_true.reshape(-1, 1)},
                batch_size=1024
            )
            ode_domain.add_validator(validator, f"validator_s0_{s0_val}_Sa_{i}")
            i += 1

    # -------------------------
    # Solver
    # -------------------------
    slv = Solver(cfg, ode_domain)
    slv.solve()


if __name__ == '__main__':
    run()

