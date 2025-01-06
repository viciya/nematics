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

# EXAMPLE OF LOADING A GLB FILE
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



# %%
import trimesh
import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
SAVE = False

def select_faces(faces, select_idx):
        # --- Get the indices of the selected vertices
    selected_vertex_indices = np.where(select_idx)[0]
    selected_faces = []
    for face in faces:
        # Check if all vertices in the face are in the selected vertices
        if np.all(np.isin(face, selected_vertex_indices)):
            # Map the original face indices to the new indices
            new_face = [np.where(selected_vertex_indices == idx)[0][0] for idx in face]
            selected_faces.append(new_face)

    # Convert to a numpy array
    return np.array(selected_faces)

%matplotlib qt

# Load a 3D model from a file
scene = trimesh.load(r"C:\Users\victo\OneDrive - BGU\Nina\stones.glb")
# scene = trimesh.load(r"C:\Users\victo\OneDrive - BGU\Nina\Stone_set\1_2_2025.glb")
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

        tr0,tr1,tr2 = .83, .83, .78 # RGB colors
        select_idx = (colors_norm[:,0]<tr0) & (colors_norm[:,1]<tr1) & (colors_norm[:,2]<tr2)

        # Select the vertices and faces based on the condition
        vertices_in = vertices[select_idx]
        faces_in = select_faces(faces, select_idx)
        colors_norm_in = colors_norm[select_idx]

        vertices_out = vertices[~select_idx]
        colors_norm_out = colors_norm[~select_idx]
        
        # Plot the vertices
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # sc = ax.scatter(vertices_in[:, 0], vertices_in[:, 1], vertices_in[:, 2], c=colors_norm_in[:,:3])
        sc = ax.scatter(vertices_in[:, 0], vertices_in[:, 1], vertices_in[:, 2], c="r", s=3, alpha=.3)  # s is the size of the points
        sc = ax.scatter(vertices_out[:, 0], vertices_out[:, 1], vertices_out[:, 2], c="b", s=3, alpha=.3) 
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        drange = .25
        # ax.set_xlim(.3 - drange, .3 + drange)
        # ax.set_ylim(mid_y - max_range, mid_y + max_range)
        # ax.set_zlim(0.36 - drange, 0.36 + drange)
        ax.view_init(elev=-30, azim=90) 
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
plt.hist(colors_norm[:,0],30, alpha=.3, color="r",label='Red')
plt.hist(colors_norm[:,1], 30,alpha=.3, color="g",label='Green')
plt.hist(colors_norm[:,2], 30,alpha=.3, color="b",label='Blue')
plt.legend()


# %%
# Here we find stones as connected components 


from scipy.spatial import distance
from scipy.sparse.csgraph import connected_components

# Define the distance threshold
distance_threshold = 0.02

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
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

# Count the number of points in each component
component_sizes = np.bincount(labels)
# Select labels of components with more than 100 points
large_components_labels = np.where(component_sizes > 300)[0]

# Define a color map for different components
colors = plt.cm.get_cmap('hsv', len(large_components_labels))

objects = []
# Plot each component with a different color
for i, label in enumerate(large_components_labels[:]):
    component_idx = labels==label
    component_points = vertices_in[component_idx]
    mesh = trimesh.Trimesh(vertices=component_points, faces=select_faces(faces_in, component_idx))
    mesh.visual.vertex_colors =  colors_norm_in[component_idx]

    objects.append(mesh)
    area = mesh.area
    ax.scatter(component_points[:, 0], component_points[:, 1], component_points[:, 2], 
                color=colors(i), label='%s: %1.5s'%(label, str(area)), s=6, alpha=.3)
    
    
    ax.text(np.mean(component_points[:, 0]), np.mean(component_points[:, 1]), np.mean(component_points[:, 2]), 
            '%s: %1.5s'%(label, str(area)))#, color=colors(i))
    

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Connected Components')

ax.legend()
ax.set_ylim(0, .6)
ax.set_aspect("equal")
# Set aspect ratio to 1
# max_range = np.array([vertices[:, 0].max() - vertices[:, 0].min(),
#                         vertices[:, 1].max() - vertices[:, 1].min(),
#                         vertices[:, 2].max() - vertices[:, 2].min()]).max() / 2.0

# mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
# mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
# mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5

# ax.set_xlim(mid_x - max_range, mid_x + max_range)
# ax.set_ylim(mid_y - max_range, mid_y + max_range)
# ax.set_zlim(mid_z - max_range, mid_z + max_range)
#  %%
objects[-1].show()
# %%

# Animation function to rotate the view
def update(frame):
    # ax.view_init(elev=frame, azim=-90)  # Change the azimuth angle
    ax.view_init(frame, -30, 90)
    return sc,

    # Create the animation
frames = array = np.concatenate(
    (np.linspace(0, -120, num=20), 
    np.linspace(-120, 0, num=20))
    )
ani = FuncAnimation(fig, update, frames=frames, blit=True)

# Save the animation as a GIF
ani.save(r"C:\Users\victo\OneDrive - BGU\Nina\stones_rotating_point_cloud_2.gif", writer='pillow', fps=5)  

# %%

%matplotlib qt

# Select faces that reference the selected vertices
selected_faces = []
for face in faces:
    # Check if all vertices in the face are in the selected vertices
    if np.all(np.isin(face, selected_vertex_indices)):
        # Map the original face indices to the new indices
        new_face = [np.where(selected_vertex_indices == idx)[0][0] for idx in face]
        selected_faces.append(new_face)

# Convert to a numpy array
faces_in = np.array(selected_faces)


# Create a new mesh with the selected vertices and faces
filtered_mesh = trimesh.Trimesh(vertices=vertices_in, faces=faces_in)

# Optionally, visualize the filtered mesh
filtered_mesh.show()


# selected_vertices_mask  = vertices_in[labels == large_components_labels[0]]
# selected_faces = faces[np.all(selected_vertices_mask[faces], axis=1)]