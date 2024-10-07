# %%


# import numpy as np
# import trimesh
# import matplotlib.pyplot as plt
# from pyglet import gl
# import pyglet

# # Load the GLB file
# mesh = trimesh.load(r"C:\Users\victo\OneDrive - BGU\Nina\stones.glb")


# # Load the GLB file as a scene
# scene = trimesh.load(r"C:\Users\victo\OneDrive - BGU\Nina\stones.glb")  # Replace with your GLB file path

# # Iterate through the geometries in the scene
# for geometry in scene.geometry.values():
#     if hasattr(geometry, 'visual'):
#         print("Visual Properties:")
#         print(geometry.visual)

#         # Check for materials
#         if hasattr(geometry.visual, 'materials'):
#             materials = geometry.visual.materials
#             for material in materials:
#                 if 'baseColorFactor' in material:
#                     print("Base Color Factor:", material['baseColorFactor'])


import trimesh
import numpy as np
import matplotlib.pyplot as plt

# Load the GLB file as a scene
scene = trimesh.load(r"C:\Users\victo\OneDrive - BGU\Nina\stones.glb")   # Replace with your GLB file path

# Iterate through the geometries in the scene
for i,geometry in enumerate(scene.geometry.values()):
    if hasattr(geometry, 'visual'):
        print(i,"Visual Properties:")
        print(geometry.visual)

geometry.visual.to_color().vertex_colors[:,0].shape







# # Calculate the surface area
# surface_area = mesh.area
# print(f'Surface Area: {surface_area}')
# print(mesh.volume / mesh.convex_hull.volume)


# # Check if the mesh is valid
# if not mesh.is_empty:
#     # Show the mesh in a viewer
#     mesh.show()
# else:
#     print("The mesh is empty or invalid.")



# %%


# import trimesh

# # Load a 3D model from a file
# scene = trimesh.load(r"C:\Users\victo\OneDrive - BGU\Nina\stones.glb")


# # Check if the scene is valid
# if not scene.is_empty:
#     # Iterate through each geometry in the scene
#     for name, geometry in scene.geometry.items():
#         print(f"Geometry Name: {name}")
#         # Access the vertices of the geometry
#         vertices = geometry.vertices
#         print("Coordinates of the mesh vertices:")
#         for i, vertex in enumerate(vertices):
#             print(f"Vertex {i}: {vertex}")
# else:
#     print("The scene is empty or invalid.")

# %%
import trimesh
import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
SAVE = False

%matplotlib qt

# Load a 3D model from a file
scene = trimesh.load(r"C:\Users\victo\OneDrive - BGU\Nina\stones.glb")

# Check if the scene is valid
if not scene.is_empty:
    # Iterate through each geometry in the scene
    for name, geometry in scene.geometry.items():
        print(f"Geometry Name: {name}")
        # Access the vertices of the geometry
        vertices = geometry.vertices
        faces = geometry.faces
        colors = geometry.visual.to_color().vertex_colors
        colors_norm = colors/colors.max(axis=0)

        tr = .75
        select_idx = (colors_norm[:,0]<tr) & (colors_norm[:,1]<tr) & (colors_norm[:,2]<tr)
        vertices_in = vertices[select_idx]
        colors_norm_in = colors_norm[select_idx]

        vertices_out = vertices[~select_idx]
        colors_norm_out = colors_norm[~select_idx]
        
        # Plot the vertices
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(vertices_in[:, 0], vertices_in[:, 1], vertices_in[:, 2], c="r", s=6, alpha=.6)  # s is the size of the points
        sc = ax.scatter(vertices_out[:, 0], vertices_out[:, 1], vertices_out[:, 2], c="b", s=6, alpha=.6) 
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        drange = .12
        ax.set_xlim(.3 - drange, .3 + drange)
        # ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(0.36 - drange, 0.36 + drange)
        plt.show()

        # break

# # Set aspect ratio to 1
#         max_range = np.array([vertices[:, 0].max() - vertices[:, 0].min(),
#                               vertices[:, 1].max() - vertices[:, 1].min(),
#                               vertices[:, 2].max() - vertices[:, 2].min()]).max() / 2.0

#         mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
#         mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
#         mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5

#         ax.set_xlim(mid_x - max_range, mid_x + max_range)
#         ax.set_ylim(mid_y - max_range, mid_y + max_range)
#         ax.set_zlim(mid_z - max_range, mid_z + max_range)
    if SAVE:
        # Animation function to rotate the view
        def update(frame):
            ax.view_init(elev=120, azim=frame)  # Change the azimuth angle
            return sc,

            # Create the animation
        frames = array = np.concatenate(
            (np.linspace(-50, -150, num=20), 
            np.linspace(-150, -50, num=20))
            )
        ani = FuncAnimation(fig, update, frames=frames, blit=True)

        # Save the animation as a GIF
        ani.save(r"C:\Users\victo\OneDrive - BGU\Nina\stones_rotating_point_cloud.gif", writer='pillow', fps=5)  

        
else:
    print("The scene is empty or invalid.")

# %%
plt.hist(colors_norm[:,0],30, alpha=.3)
plt.hist(colors_norm[:,1], 30,alpha=.3)
plt.hist(colors_norm[:,2], 30,alpha=.3)

# %%
colors_norm<0.7
# %%
# Here we find stones as conncted components 


from scipy.spatial import distance
from scipy.sparse.csgraph import connected_components

# Define the distance threshold
distance_threshold = 0.03

# Calculate the pairwise distance matrix
dist_matrix = distance.pdist(vertices_in, metric='euclidean')
dist_matrix = distance.squareform(dist_matrix)

# Create a binary adjacency matrix based on the distance threshold
adjacency_matrix = (dist_matrix < distance_threshold).astype(int)

# Find connected components
n_components, labels = connected_components(adjacency_matrix, directed=False, return_labels=True)

# Print the results
print(f'Number of connected components: {n_components}')
# %%
# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Count the number of points in each component
component_sizes = np.bincount(labels)
# Select labels of components with more than 100 points
large_components_labels = np.where(component_sizes > 100)[0]

# Define a color map for different components
colors = plt.cm.get_cmap('hsv', len(large_components_labels))

# Plot each component with a different color
for i, label in enumerate(large_components_labels):
    component_idx = labels == label
    component_points = vertices_in[component_idx]
    ax.scatter(component_points[:, 0], component_points[:, 1], component_points[:, 2], 
                color=colors(i), label=f'Component {label}', s=6, alpha=.1)

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Connected Components')
# ax.legend()


# Set aspect ratio to 1
max_range = np.array([vertices[:, 0].max() - vertices[:, 0].min(),
                        vertices[:, 1].max() - vertices[:, 1].min(),
                        vertices[:, 2].max() - vertices[:, 2].min()]).max() / 2.0

mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# %%

# Animation function to rotate the view
def update(frame):
    ax.view_init(elev=frame, azim=-90)  # Change the azimuth angle
    return sc,

    # Create the animation
frames = array = np.concatenate(
    (np.linspace(0, -60, num=20), 
    np.linspace(-60, 0, num=20))
    )
ani = FuncAnimation(fig, update, frames=frames, blit=True)

# Save the animation as a GIF
ani.save(r"C:\Users\victo\OneDrive - BGU\Nina\stones_rotating_point_cloud_1.gif", writer='pillow', fps=5)  

# %%

%matplotlib qt
vertex_index_map = {original_idx: new_idx for new_idx, original_idx in enumerate(np.where(select_idx)[0])}

# Select corresponding faces
# A face is part of the selected vertices if all its vertices are in the selected indices
selected_faces = []
for face in faces:
    if all(vertex in vertex_index_map for vertex in face):
        # Map the original vertex indices to the new indices
        new_face = [vertex_index_map[vertex] for vertex in face]
        selected_faces.append(new_face)


# Convert selected_faces to a numpy array
selected_faces = np.array(selected_faces)

# Create a new mesh with the selected vertices and faces
filtered_mesh = trimesh.Trimesh(vertices=vertices_in, faces=selected_faces)

# Optionally, visualize the filtered mesh
filtered_mesh.show()


# selected_vertices_mask  = vertices_in[labels == large_components_labels[0]]
# selected_faces = faces[np.all(selected_vertices_mask[faces], axis=1)]