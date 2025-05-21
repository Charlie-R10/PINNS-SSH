import math
import numpy as np
import sympy
from sympy import Symbol, Function

import torch
import physicsnemo as pn  # PhysicsNEMO main package

from physicsnemo.domain import Domain
from physicsnemo.geometry import Line1D
from physicsnemo.pde import PDE
from physicsnemo.constraint import DirichletBC, PDEConstraint
from physicsnemo.networks import FullyConnectedNet
from physicsnemo.solver import Solver


# Define PDE for neutron diffusion
class NeutronDiffusionNonMult1D(PDE):
    def __init__(self, D, Sa):
        super().__init__()
        self.D = D
        self.Sa = Sa

    def residual(self, inputs, outputs):
        x = inputs["x"]
        u = outputs["u"]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        L_square = self.D / self.Sa
        coef = -1 / L_square
        return u_xx + coef * u


def analytical_solution(x, s0, D, a_ex):
    L = math.sqrt(D / 0.005)
    numerator = np.sinh((a_ex - x) / L)
    denominator = np.cosh(a_ex / L)
    return (s0 * L / (2 * D)) * (numerator / denominator)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    D = 1 / (3 * 1.5)
    Sa = 0.005

    a = 1.0
    a_ex = a + 0.7104 * 3 * D
    min_x = 0.0
    max_x = a_ex

    # Create geometry and domain
    line = Line1D(min_x, max_x)
    domain = Domain()

    # Instantiate PDE object
    pde = NeutronDiffusionNonMult1D(D, Sa)

    # Instantiate neural network
    net = FullyConnectedNet(input_dim=1, output_dim=1, num_hidden_layers=4, num_neurons_per_layer=64).to(device)

    # Add PDE constraint to domain
    pde_constraint = PDEConstraint(
        pde=pde,
        net=net,
        geometry=line,
        batch_size=1024,
        device=device
    )
    domain.add_constraint(pde_constraint)

    # Boundary condition at x = 0, Dirichlet BC: u = phi_0
    # You can parametrize s0 outside the solver, here we fix s0 to a value for example

    s0_val = 15.0
    L_square = D / Sa
    L = math.sqrt(L_square)
    phi_0 = s0_val * L * (1 - math.exp(-2 * a_ex / L)) / (2 * D * (1 + math.exp(-2 * a_ex / L)))

    def bc_func(x):
        # Boundary condition returns u=phi_0 at x=0
        return torch.full((x.shape[0], 1), phi_0, device=device)

    bc_min = DirichletBC(
        net=net,
        geometry=line,
        outvar="u",
        batch_size=64,
        criteria=lambda x: torch.isclose(x[:, 0], torch.tensor(min_x, device=device)),
        func=
