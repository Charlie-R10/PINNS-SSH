import vtk
import numpy as np
import pandas as pd

# Create a reader
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(r"C:\Users\charl\OneDrive\Desktop\Uni work\4th year project\VTP_extractor\inf_data.vtp") #change this to correct filepath

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
    'u_data': u_data,
    'ID points': id_point_final
})

# Write the DataFrame to an Excel file
data.to_excel('output.xlsx', index=False)
