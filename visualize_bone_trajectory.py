import os
import sys
import glob
import numpy as np
import trimesh
import argparse



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class VisualizerMatplotlib:

    def __init__(self, cmap, savename, limit):
        self.cmap = cmap
        self.savename = savename
        self.limit = limit
        self.setup()

    def setup(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        limit = self.limit
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-limit, limit])
        self.ax = ax

    def plot_3d_flow(self, points, colors=None):
        x, y, z = points.T

        if colors is None:
            flow_mag = np.arange(len(x))
            colors = plt.cm.get_cmap(self.cmap)(flow_mag / flow_mag.max())

        # ax.quiver(x[:-1], y[:-1], z[:-1], x[1:]-x[:-1], y[1:]-y[:-1], z[1:]-z[:-1],
        #           color=colors, length=1, normalize=False)

        for i in range(len(x)-1):
            # ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]], color=colors[i])
            self.ax.scatter(x[i], y[i], z[i], color=colors[i], marker='o')

        
        
    def plot(self, array):
        
        flow_mag = np.arange(array.shape[1])
        colors = plt.cm.get_cmap(self.cmap)(flow_mag / flow_mag.max())

        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=plt.Normalize(vmin=0, vmax=flow_mag.max()))
        sm.set_array([])

        cbar = plt.colorbar(sm)
        cbar.set_label('Flow Magnitude')

        for i in range(len(array)):
            self.plot_3d_flow(array[i], colors=colors)

        plt.savefig(f"minitest/{self.savename}_plt_view0.png")
        self.ax.view_init(elev=90, azim=0)
        plt.savefig(f"minitest/{self.savename}_plt_view1.png")
        self.ax.view_init(elev=45, azim=0)
        plt.savefig(f"minitest/{self.savename}_plt_view2.png")

# Example usage:
# Replace `your_points` with your actual points in the format (N, 3)
# your_points = np.array(
#     [[1,1,1],[1,1,0.9],[1,0.8,0.8],[1,0.6,0.6],[0.6,0.35,0.35],[0.5,0.3,0.3],[0.2,0.2,0.2],[0.1,0.1,0.1]])
# plot_3d_flow(your_points)

class VisualizerTrimesh:

    def __init__(self, cmap, savename):
        self.cmap = cmap
        self.savename = savename
        self.radius = 1e-4
        self.vis_mesh = []

    def plot_3d_flow(self, points, colors=None):
        if colors is None:
            pass
        
        for i in range(1, len(points)):
            # segment = np.stack([points[i-1], points[i]])
            # line = trimesh.creation.cylinder(
            #     self.radius,
            #     segment=segment,
            #     sections=5,
            #     vertex_colors=colors[i]
            # )
            # self.vis_mesh.append(line)
        
            sphere = trimesh.creation.uv_sphere(radius=self.radius*10, count=[4,4])
            sphere.visual.vertex_colors = colors[i]
            sphere.vertices += points[i]

            self.vis_mesh.append(sphere)    

    def plot(self, array):

        flow_mag = np.arange(array.shape[1])
        colors = plt.cm.get_cmap(self.cmap)(flow_mag / flow_mag.max())

        for i in range(len(array)):
            self.plot_3d_flow(array[i], colors)
        
        meshes_fin = trimesh.util.concatenate(self.vis_mesh)
        meshes_fin.export(f"visualize/{self.savename}.obj")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='meao_ver1.13/a-eagle', 
                                help="root directory to be visualized")
    args = parser.parse_args()
    
    seqname = args.root.split('/')[-1]
    root_dir = f"logdir/{args.root}/npy"
    bone_loc_paths = sorted(glob.glob(os.path.join(root_dir, "*_loc.npy")))
    # bone_os_paths = glob.glob(os.path.join(root_dir, "*_os.npy"))
    if len(bone_loc_paths) == 0:
        raise ValueError(f"Invalid bone path : ", root_dir)
    
    bone_loc_matrix = np.zeros((len(bone_loc_paths), 3))
    
    bone_loc_mat = np.zeros((len(bone_loc_paths), 25, 3))
    for idx, path in enumerate(bone_loc_paths):
        bone_loc_mat[idx] = np.load(path)
    

    # plot some bones
    bone_select_idx = [i for i in range(25)]

    if bone_select_idx is not None:
        bone_loc_mat = bone_loc_mat[:, bone_select_idx, :]

    input_bone_loc_mat = bone_loc_mat.transpose(1,0,2)

    # # color list = ['RdYlGn', 'hsv', 'YlOrRd']
    # # visualizer = VisualizerMatplotlib(cmap='RdYlGn', 
    # #                                 savename=f'{seqname}_{str(bone_select_idx)}',
    # #                                 limit=0.05)
    # # visualizer.plot(input_bone_loc_mat)

    savename_ = "_".join(args.root.split('/'))
    visualizer = VisualizerTrimesh(cmap='YlGn', savename=f'{savename_}_{bone_select_idx}')
    visualizer.plot(input_bone_loc_mat)

    # plot each bones

    # for idx in bone_select_idx:
    #     bone_loc_mat_each = bone_loc_mat[:, [idx], :]

    #     input_bone_loc_mat = bone_loc_mat_each.transpose(1,0,2)
    #     savename_ = "_".join(args.root.split('/'))
    #     visualizer = VisualizerTrimesh(cmap='YlGn', savename=f'{savename_}_each_{idx}')
    #     visualizer.plot(input_bone_loc_mat)
