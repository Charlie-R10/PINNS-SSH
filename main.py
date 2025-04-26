import vtk
import numpy as np
import pandas as pd

# Create a reader
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(r"C:\Users\charl\OneDrive\Desktop\Uni work\4th year project\VTP_extractor\inf_data.vtp")

D = 0.5
Sa = 0.01
S0 = 20
a = 1
L = np.sqrt(D / Sa)
a_ex = a + (0.7104*3*D)

phi_values = []

x_vals = np.linspace(0, 1, 101)

for x in x_vals:
    #phi = ((S0 * L) / (2 * D)) * (np.sinh((a - 2 * x) / (2 * L)) / np.cosh(a / (2 * L))
    #phi = S0 * L * (1 - np.exp(-2 * a_ex / L)) / (2 * D * (1 + np.exp(-2 * a_ex / L)))  # change to sympy.exp

    exp_term = np.exp(-2 * a_ex / L)
    numerator = S0 * L * (np.cosh(x / L) - exp_term * np.cosh((2 * a_ex - x) / L))
    denominator = 2 * D * (1 + exp_term)
    phi = numerator / denominator
    phi_values.append(phi)

# Read the file
reader.Update()

# Get the output
polydata = reader.GetOutput()

# Access point data
point_data = polydata.GetPointData()

# Access ID points
id_points = polydata.GetPoints()

# Get the predicted u data and points
u_data_array = point_data.GetArray("u")

# Convert u data to numpy array
u_data = np.array([u_data_array.GetValue(i) for i in range(u_data_array.GetNumberOfTuples())])

id_points_array = np.array([id_points.GetPoint(i) for i in range(id_points.GetNumberOfPoints())])

# Extract the first column
id_point_final = id_points_array[:, 0]

# Create a DataFrame to store the data
data = pd.DataFrame({
    'D': D,
    'Sa': Sa,
    'S0': S0,
    'a': a,
    'u_data': u_data,
    'u_predicted': np.array(phi_values),
    'ID points': id_point_final
})

# Write the DataFrame to an Excel file
data.to_excel('output.xlsx', index=False)