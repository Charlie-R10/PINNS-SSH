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

# Helper class for normalization
class Normalizer:
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi
        self.range = hi - lo + 1e-12
        
    def norm(self, x):
        return (x - self.lo) / self.range
    
    def unnorm(self, y):
        return y * self.range + self.lo

# Define custom PDE in class
class NDequation(PDE):
    def __init__(self):
        x = Symbol("x")
        Sa = Symbol("Sa")
        u = Function("u")(x, Sa)

        # diffusion coefficient
        D = 1 / (3 * 1.5)
        coef = -Sa / D  # u'' + coef * u = 0
        self.equations = {"neutron_diffusion_equation": u.diff(x, 2) + coef * u}

@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # Physical constants
    D = 1 / (3 * 1.5)
    a = 1.0
    a_ex = a + 0.7104 * 3 * D
    max_x = a_ex / 2

    # Define symbolic parameters
    x_sym, s0_sym, Sa_sym = Symbol("x"), Symbol("s0"), Symbol("Sa")

    # Real parameter bounds
    s0_min, s0_max = 1e3, 5e6
    Sa_min, Sa_max = 100, 1000
    x_min, x_max = 0.0, max_x

    # Create normalizers
    s0_norm = Normalizer(s0_min, s0_max)
    Sa_norm = Normalizer(Sa_min, Sa_max)
    x_norm = Normalizer(x_min, x_max)

    # Use normalized parameter ranges [0,1] in PhysicsNeMo
    param_ranges = {s0_sym: (0, 1), Sa_sym: (0, 1)}
    pr = Parameterization(param_ranges)

    # Instantiate PDE and network
    ode = NDequation()
    custom_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("s0"), Key("Sa")],
        output_keys=[Key("u")]
    )
    nodes = ode.make_nodes() + [custom_net.make_node(name="ode_network")]

    # Geometry and domain
    line = Line1D(x_min, x_max)
    ode_domain = Domain()

    # Boundary constraints (symbolic outvar uses raw symbols, normalization handled by pr)
    L_sym = sympy.sqrt(D / Sa_sym)
    phi0 = ((s0_sym * L_sym) / (2 * D)) * (sympy.sinh(a_ex / (2 * L_sym)) / sympy.cosh(a_ex / (2 * L_sym)))
    bc_min = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=line,
        outvar={"u": phi0}, criteria=sympy.Eq(x_sym, x_min),
        batch_size=cfg.batch_size.bc_min, parameterization=pr
    )
    bc_max = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=line,
        outvar={"u": 0}, criteria=sympy.Eq(x_sym, x_max),
        batch_size=cfg.batch_size.bc_max, parameterization=pr
    )
    ode_domain.add_constraint(bc_min, "bc_min")
    ode_domain.add_constraint(bc_max, "bc_max")

    # Interior PDE constraint
    interior = PointwiseInteriorConstraint(
        nodes=nodes, geometry=line,
        outvar={"neutron_diffusion_equation": 0},
        batch_size=cfg.batch_size.interior, parameterization=pr
    )
    ode_domain.add_constraint(interior, "interior")

    # Define sample points and normalize
    points = np.linspace(x_min, x_max, 101).reshape(-1, 1)
    points_n = x_norm.norm(points)

    # Analytical solution for validation
    def analytical_solution(x, s0, D, a_ex, Sa):
        L = math.sqrt(D / Sa)
        return (s0 * L / (2 * D)) * (np.sinh((a_ex - 2*x) / (2*L)) / np.cosh(a_ex / (2*L)))

    # Validator loop over parameter values
    i = 0
    for s0_val in [1e3, 1e4, 1e5, 2e6, 5e6]:
        for Sa_val in [100, 400, 800]:
            # Compute true solution and normalize
            u_true = analytical_solution(points.flatten(), s0_val, D, a_ex, Sa_val)
            u_max = np.max(np.abs(u_true)) + 1e-8
            u_n = (u_true / u_max).reshape(-1, 1)
            # Normalize inputs
            s0_n = s0_norm.norm(s0_val)
            Sa_n = Sa_norm.norm(Sa_val)
            validator = PointwiseValidator(
                nodes=nodes,
                invar={
                    "x": points_n,
                    "s0": np.full_like(points_n, s0_n),
                    "Sa": np.full_like(points_n, Sa_n)
                },
                true_outvar={"u": u_n}, batch_size=1024
            )
            ode_domain.add_validator(validator, f"validator_{i}")
            i += 1

    # Solve
    slv = Solver(cfg, ode_domain)
    slv.solve()

if __name__ == '__main__':
    run()
