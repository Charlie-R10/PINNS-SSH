from sympy import Symbol, Function, Number
from physicsnemo.sym.eq.pde import PDE


class DiffusionEquation1D(PDE):
    """
    Diffusion equation 1D
    The equation is given as an example for implementing
    your own PDE.

    Parameters
    ==========
    D : [float, str]
        Diffusion coefficient
    Sigma_a : float
        Absorption cross-section
    """

    name = "DiffusionEquation1D"

    def __init__(self, u="u", D=1.0, Sigma_a=0.1, Q=0.):
        self._u = u
        # coordinates
        x = Symbol("x")

        # make input variables
        input_variables = {"x": x}

        # make u function
        assert type(u) == str, "u needs to be string"
        u = Function(u)(*input_variables)

        # Diffusion and absorption cross-sections
        if type(D) is str:
            D = Function(D)(*input_variables)
        elif type(D) in [float, int]:
            D = Number(D)

        if type(Sigma_a) is str:
            Sigma_a = Function(Sigma_a)(*input_variables)
        elif type(Sigma_a) in [float, int]:
            Sigma_a = Number(Sigma_a)

        if type(Q) is str:
            Q = Function(Q)(*input_variables)
        elif type(Q) in [float, int]:
            Q = Number(Q)

        # set equations
        self.equations = {}
        # self.equations["diffusion_equation"] = u.diff(t, 2) - (c**2 * u.diff(x)).diff(x)
        self.equations["diffusion_equation_" + self._u] = (
                D * (u.diff(x)).diff(x) - Sigma_a * u + Q
        )


# define plane source boundary conditions
class PlaneSourceBoundary(PDE):
    """
    Plane source boundary condition for neutron diffusion equation
    Ref: https://www.nuclear-power.com/nuclear-power/reactor-physics/neutron-diffusion-theory/diffusion-equation/

    Parameters
    ==========
    u : str
        The dependent variable.
    D : float,
        Diffusion coefficient
    Q : float
        Plane source
    """

    name = "PlaneSourceBoundary"

    def __init__(self, u="u", D=1.0, Q=1.0):
        # set params
        self.u = u

        # coordinates
        x = Symbol("x")

        # make input variables
        input_variables = {"x": x}

        # Scalar function
        assert type(u) == str, "u needs to be string"
        u = Function(u)(*input_variables)

        # Diffusion coefficient and neutron source
        if type(D) is str:
            D = Function(D)(*input_variables)
        elif type(D) in [float, int]:
            D = Number(D)
        else:
            raise ValueError("D must be either symbol or number but type of D is {}".format(type(D)))

        if type(Q) is str:
            raise ValueError("Q can not be string")
        elif type(Q) in [float, int]:
            Q = Number(Q)

        # set equations
        self.equations = {}
        self.equations["plane_source_boundary"] = (
            # 0.25 * u + 0.5 * D * u.diff(x) - 0.5 * Q
            -D * u.diff(x) - 0.5 * Q
        )


# define vacuum boundary conditions
class VacuumBoundary(PDE):
    """
    Vacuum boundary condition for neutron diffusion equation
    Ref: https://www.nuclear-power.com/nuclear-power/reactor-physics/neutron-diffusion-theory/diffusion-equation/

    Parameters
    ==========
    u : str
        The dependent variable.
    D : float,
        Diffusion coefficient
    """

    name = "VacuumBoundary"

    def __init__(self, u="u", D=1.0, extrapolated_length=True):
        # set params
        self.u = u

        # coordinates
        x = Symbol("x")

        # make input varialbles
        input_variables = {"x": x}

        # Scalar function
        assert type(u) == str, "u needs to be string"
        u = Function(u)(*input_variables)

        # Diffusion coefficient
        if type(D) is str:
            D = Function(D)(*input_variables)
        elif type(D) in [float, int]:
            D = Number(D)

        # set equations
        self.equations = {}
        if extrapolated_length:
            self.equations["vacuum_boundary"] = (
                u
            )
        else:
            self.equations["vacuum_boundary"] = (
                0.25 * u + 0.5 * D * u.diff(x)
            )


# define reflective boundary condition
class ReflectiveBoundary(PDE):
    """
    Reflective boundary condition for neutron diffusion equation
    Ref: https://www.nuclear-power.com/nuclear-power/reactor-physics/neutron-diffusion-theory/diffusion-equation/

    Parameters
    ==========
    u : str
        The dependent variable.
    D : float,
        Diffusion coefficient
    """

    name = "ReflectiveBoundary"

    def __init__(self, u="u", D=1.0):
        # set params
        self.u = u

        # coordinates
        x = Symbol("x")

        # make input variables
        input_variables = {"x": x}

        # Scalar function
        assert type(u) == str, "u needs to be string"
        u = Function(u)(*input_variables)

        # Diffusion coefficient
        if type(D) is str:
            D = Function(D)(*input_variables)
        elif type(D) in [float, int]:
            D = Number(D)

        # set equations
        self.equations = {}
        self.equations["reflective_boundary"] = (
            -D * u.diff(x)
        )


# define interface conditions for multiregion problems
class InterfaceDiffusion1D(PDE):
    """
    Interface boundary condition for 1D1G neutron diffusion equation
    Ref: https://www.nuclear-power.com/nuclear-power/reactor-physics/neutron-diffusion-theory/diffusion-equation/

    Parameters
    ==========
    u : str
        The dependent variable.
    D : float,
        Diffusion coefficient
    """

    name = "Interface"

    def __init__(self, u1="u1", u2="u2", D1=1.0, D2=1.0):
        self.u1 = u1
        self.u2 = u2

        # input variable
        x = Symbol("x")
        input_variables={"x": x}

        # scalar functions
        assert type(u1) == str, "u1 needs to be string"
        assert type(u2) == str, "u2 needs to be string"
        u1 = Function(u1)(*input_variables)
        u2 = Function(u2)(*input_variables)

        # Diffusion coefficients
        if type(D1) is str:
            D1 = Function(D1)(*input_variables)
        elif type(D1) in [float, int]:
            D1 = Number(D1)
        else:
            raise ValueError("D1 must be either symbol or number but type of D1 is {}".format(type(D1)))

        if type(D2) is str:
            D2 = Function(D2)(*input_variables)
        elif type(D2) in [float, int]:
            D2 = Number(D2)
        else:
            raise ValueError("D2 must be either symbol or number but type of D2 is {}".format(type(D2)))

        # set equations
        self.equations = {}
        # Flux continuity equation
        self.equations["flux_continuity"] = (
            u1 - u2
        )
        self.equations["current_continuity"] = (
            D1*u1.diff(x) - D2*u2.diff(x)
        )


class DiffusionEquation2D(PDE):
    """
    Diffusion equation for 2D geometry
    The equation is given as an example for implementing
    your own PDE.

    Parameters
    ==========
    D : float
        Diffusion coefficient
    Sigma_a : float
        Absorption cross-section
    """

    name = "Diffusion2D"

    def __init__(self, u="u", D=1.0, Sigma_a=0.1, Q=0.):
        self._u = u
        # coordinates
        x = Symbol("x")
        y = Symbol("y")

        # make input variables
        input_variables = {"x": x, "y": y}

        # make u function
        assert type(u) == str, "u needs to be string"
        u = Function(u)(*input_variables)

        # Diffusion and absorption cross-sections
        if type(D) is str or type(Sigma_a) is str:
            raise ValueError("D and Sigma_a must be numbers")
        elif type(D) in [float, int] and type(Sigma_a) in [float, int] and type(Q) in [float, int]:
            D = Number(D)
            Sigma_a = Number(Sigma_a)
            Q = Number(Q)

        # set equations
        self.equations = {}
        # self.equations["diffusion_equation"] = u.diff(t, 2) - (c**2 * u.diff(x)).diff(x)
        self.equations["diffusion_equation_" + self._u] = (
            (D * u.diff(y)).diff(y) + (D * u.diff(x)).diff(x) - Sigma_a * u + Q
        )


class VacuumBoundary2D(PDE):
    """
    Vacuum boundary condition for neutron diffusion equation
    Ref: https://www.nuclear-power.com/nuclear-power/reactor-physics/neutron-diffusion-theory/diffusion-equation/

    Parameters
    ==========
    u : str
        The dependent variable.
    D : float,
        Diffusion coefficient
    """

    name = "VacuumBoundary"

    def __init__(self, u="u", D=1.0, extrapolated_length=True):
        # set params
        self.u = u

        # coordinates
        x = Symbol("x")
        y = Symbol("y")

        norm_x = Symbol("norm_x")
        norm_y = Symbol("norm_y")

        # make input varialbles
        input_variables = {"x": x, "y": y}

        # Scalar function
        assert type(u) == str, "u needs to be string"
        u = Function(u)(*input_variables)

        # Diffusion coefficient
        if type(D) is str:
            raise ValueError("D can not be string")
        elif type(D) in [float, int]:
            D = Number(D)

        # set equations
        self.equations = {}
        if extrapolated_length:
            self.equations["vacuum_boundary"] = (
                u
            )
        else:
            self.equations["vacuum_boundary"] = (
                0.25 * u + 0.5 * D * (u.diff(x) * norm_x + u.diff(y) * norm_y)
            )
