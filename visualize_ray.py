# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from absl import flags, app
import sys
sys.path.insert(0,'third_party')
import numpy as np
import torch
import os
import glob
import cv2
import trimesh
from scipy.spatial.transform import Rotation as R
import imageio

from utils.io import save_vid, str_to_frame, save_bones
from utils.colors import label_colormap
from nnutils.train_utils import v2s_trainer
import matplotlib
from matplotlib import pyplot as plt
opts = flags.FLAGS
                

# TODO : color palettte

def palette():
    return ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

def save_visualize(rendered_seq, aux_seq, save_path, mode='matplotlib', ray_select_mode='slide', vis_2d=True, animate=False, vis_ray_origin=False):
    """
    visualize ray & bone & mesh -> Check how BANMo warps points
    """

    # parsing
    # print("Parsing : rendered_seq & aux_seq")
    save_path += '_' + ray_select_mode

    ray_rest = rendered_seq['ray_canonical']    # List[torch.Tensor(rnd_frame_chunk, img_size, img_size, 128 * 3), ...]
    ray_cam = rendered_seq['ray_camera']        # List[torch.Tensor(rnd_frame_chunk, img_size, img_size, 128 * 3), ...]
    ray_origin = rendered_seq['ray_origin']     # List[torch.Tensor(rnd_frame_chunk, img_size, img_size, 128 * 3), ...]
    ray_alphas = rendered_seq['alphas']
    skinning_weight = rendered_seq['skin_backward']

    img = rendered_seq['img']                   # List[torch.Tensor(rnd_frame_chunk, img_size, img_size, 3), ...]
    mask = rendered_seq['sil']                  # List[torch.Tensor(rnd_frame_chunk, img_size, img_size, 1), ...]

    mesh_rest = aux_seq['mesh_rest']
    scale = (mesh_rest.vertices.max(0) - mesh_rest.vertices.min(0)).max()
    mesh_cam = aux_seq['mesh']
    bone_rest = aux_seq['bone_rest']
    bone_cam = aux_seq['bone']
    obj_bound = aux_seq['obj_bound']
    # print("-"*10)     
    
    # type / shape check
    print("Type / Shape Check")
    # print(type(skinning_weight), len(skinning_weight), skinning_weight[0].shape)  # (rnd_frame_chunk, img_size, img_size, 128, 25)
    # exit()
    # print(type(mesh_rest))
    # print(type(mesh_cam), len(mesh_cam))    # frame 수 만큼
    # print(type(bone_rest), bone_rest.shape) 
    # print(type(bone_cam), len(bone_cam), bone_cam[0].shape)     # frame 수 * bone 수 * 10 (=parameter 수)
    # print(type(ray_rest), len(ray_cam), ray_cam[0].shape)   # 7 / (3, img size, img size, 384) 앞에 3이 아마 banmo flag 의 rnd_frame_chunk 인 것 같고 ... 384 = 128 x 3
    # print(type(ray_cam), len(ray_cam), ray_cam[0].shape)
    # print(type(ray_origin), len(ray_origin), ray_origin[0].shape, ray_origin[-1].shape) # ray 랑 shape 같음
    # print(type(obj_bound))  # trimesh.cahcing.TrackedArray
    # print(np.asarray(obj_bound))
    # print("ray alphas", type(ray_alphas), len(ray_alphas), ray_alphas[0].shape)
    # print("-"*10)

    # exit()

    # Tensor manipulation
    ray_rest = torch.cat(ray_rest, dim=0)
    ray_cam = torch.cat(ray_cam, dim=0)
    ray_origin = torch.cat(ray_origin, dim=0)
    ray_alphas = torch.cat(ray_alphas, dim=0)
    skinning_weight = torch.cat(skinning_weight, dim=0)
    bone_cam = np.array(bone_cam)

    img = torch.cat(img, dim=0).cpu()
    mask = torch.cat(mask, dim=0).cpu()
    nonzero = torch.nonzero(mask, as_tuple=True)    # (batch, row=y, col=x)
    img_size = img.shape[-2]

    valid_x = np.asarray(nonzero[2])
    valid_y = np.asarray(nonzero[1])

    ray_start_idx = img.shape[-2] - 5
    ray_end_idx = img.shape[-2] + 5
    ray_slide = 400

    # FIXME
    if ray_select_mode == 'mask_range':
        # to visualize neighboring pixels
        # ray_rest = ray_rest[ray_start_idx:ray_end_idx]
        # ray_cam = ray_cam[ray_start_idx:ray_end_idx]
        valid_x = valid_x[ray_start_idx:ray_end_idx]
        valid_y = valid_y[ray_start_idx:ray_end_idx]
        save_path += f'_from{ray_start_idx}to{ray_end_idx}'

    if ray_select_mode == 'xrange':
        # CUSTOM    # use for drawing one horizontal line 
        row_offset = 20
        start = 30
        end = 31
        valid_x = np.arange(start, end, dtype=np.uint8)
        valid_y = np.ones(len(valid_x), dtype=np.uint8) * row_offset
        save_path += f'_y{row_offset}_xfrom{start}to{end}'
        # print(valid_x, valid_y)
    
    if ray_select_mode == 'yrange':
        # CUSTOM    # use for drawing one horizontal line 
        col_offset = 28
        start = 10
        end = 30
        valid_y = np.arange(start, end, dtype=np.uint8)
        valid_x = np.ones(len(valid_y), dtype=np.uint8) * col_offset
        save_path += f'_x{col_offset}_yfrom{start}to{end}'
        # print(valid_x, valid_y)

    elif ray_select_mode == 'slide':
        # to visualize scattered lay
        # ray_rest = ray_rest[::ray_slide]
        # ray_cam = ray_cam[::ray_slide]
        valid_x = valid_x[::ray_slide]
        valid_y = valid_y[::ray_slide]
        save_path += f'_{ray_slide}'

    elif ray_select_mode == 'select':
        select_list = [[int(img.shape[-2] // 2)], [int(img.shape[-2] // 2)]]    # [[x1, x2, ...], [y1, y2, ...]]
        valid_x = np.asarray(select_list[0])
        valid_y = np.asarray(select_list[1])

    if mode == 'matplotlib':
        raise NotImplementedError()
        # visualize_plt(mesh_rest, mesh_cam, bone_rest, bone_cam, ray_rest, ray_cam, ray_origin, ray_alphas,
        #               valid=(valid_x, valid_y), animate=animate, save_path=save_path)
    
    elif mode == 'trimesh':
        visualize_trimesh(mesh_rest, mesh_cam, bone_rest, bone_cam, ray_rest, ray_cam, ray_origin, ray_alphas,
                          skinning_weight=skinning_weight,
                          valid=(valid_x, valid_y), scale=scale, save_path=save_path, vis_ray_origin=vis_ray_origin)

    if vis_2d:
        print("Visualize 2D mask & sampled pixel")
        visualize_2d(mask, valid_x, valid_y, img_size, save_path)


def visualize_plt(mesh_rest, mesh_cam, bone_rest, bone_cam, ray_rest, ray_cam, ray_origin, valid, animate, save_path):
    """
    Plot by matplotlib.pyplot
    Just use for debugging or roughly checking the scene.
    
    # TODO (0710)
    ray visualization using mask (reformatting)
    """
    # from mpl_toolkits.mplot3d import Axes3D
    # from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    # from matplotlib import animation

    plt.clf()
    fig = plt.figure(figsize=(14, 8))

    xyz_camera_space_xslice = ray_cam[:, :, 0]
    xyz_camera_space_yslice = ray_cam[:, :, 1]
    xyz_camera_space_zslice = ray_cam[:, :, 2]

    xyz_canonical_space_xslice = ray_rest[:, :, 0]
    xyz_canonical_space_yslice = ray_rest[:, :, 1]
    xyz_canonical_space_zslice = ray_rest[:, :, 2]
    
    x_axis_boundary_cam = (np.max(xyz_camera_space_xslice), np.min(xyz_camera_space_xslice))
    y_axis_boundary_cam = (np.max(xyz_camera_space_yslice), np.min(xyz_camera_space_yslice))
    z_axis_boundary_cam = (np.max(xyz_camera_space_zslice), np.min(xyz_camera_space_zslice))

    x_axis_boundary_canonical = (np.max(xyz_canonical_space_xslice), np.min(xyz_canonical_space_xslice))
    y_axis_boundary_canonical = (np.max(xyz_canonical_space_yslice), np.min(xyz_canonical_space_yslice))
    z_axis_boundary_canonical = (np.max(xyz_canonical_space_zslice), np.min(xyz_canonical_space_zslice))
    
    x_axis_boundary = max(x_axis_boundary_cam[0], x_axis_boundary_canonical[0]), min(x_axis_boundary_cam[1], x_axis_boundary_canonical[1])
    y_axis_boundary = max(y_axis_boundary_cam[0], y_axis_boundary_canonical[0]), min(y_axis_boundary_cam[1], y_axis_boundary_canonical[1])
    z_axis_boundary = max(z_axis_boundary_cam[0], z_axis_boundary_canonical[0]), min(z_axis_boundary_cam[1], z_axis_boundary_canonical[1])
    
    scene_bounding_max_length = max(x_axis_boundary[1] - x_axis_boundary[0], y_axis_boundary[1] - y_axis_boundary[0], z_axis_boundary[1] - z_axis_boundary[0])
    
    ax_camera = fig.add_subplot(121, projection='3d')
    ax_canonical = fig.add_subplot(122, projection='3d')
    
    colormap = label_colormap()
    print("Visualizing Rays : Camera / Canonical")
    for idx, (x, y, z, x_cam, y_cam, z_cam) in enumerate(zip(xyz_canonical_space_xslice, xyz_canonical_space_yslice, xyz_canonical_space_zslice, xyz_camera_space_xslice, xyz_camera_space_yslice, xyz_camera_space_zslice)):
        ax_canonical.plot(x, y, z, c=colormap[idx]/255, marker=".")
        ax_camera.plot(x_cam, y_cam, z_cam, c=colormap[idx]/255, marker=".")

    print("Visualizing Ray origin")
    for pts in ray_origin:
        ax_camera.plot(xs=pts[0], ys=pts[1], zs=pts[2])

    print("Visualize Bones")
    bone_dfm = bone_cam[0]
    # print(np.max(bone_dfm, axis=-1), np.min(bone_dfm, axis=-1))
    # print(x_axis_boundary, y_axis_boundary, z_axis_boundary)

    # print(bone_rest.shape)
    for one_bone_rest, one_bone_dfm in zip(bone_rest, bone_dfm):
        # print(one_bone_dfm.shape)
        # print(one_bone_rest.shape)
        center_rest = one_bone_rest[:3]
        center_dfm = one_bone_dfm[:3]
        ax_canonical.scatter(center_rest[0], center_rest[1], center_rest[2], marker='D')
        ax_camera.scatter(center_dfm[0], center_dfm[1], center_dfm[2], marker='^')
        
    plt.savefig(save_path)

def visualize_trimesh(mesh_rest, mesh_cam, bone_rest, bone_cam, ray_rest, ray_cam, ray_origin, ray_alphas, skinning_weight, valid, scale, save_path, vis_ray_origin=False):

    print("Trimesh : Visualize Rays")
    draw_ray_cam = draw_ray(ray_cam, valid, ray_alphas, skinning_weight=skinning_weight, ray_origin=ray_origin, scale=scale, save_path=save_path)
    draw_ray_canonical = draw_ray(ray_rest, valid, ray_alphas, skinning_weight=skinning_weight, scale=scale, save_path=save_path)
    
    print("Trimesh : Visualize Bones")
    save_bones(bone_rest, len_max=scale, path=save_path + '_bone_rest.obj')
    for idx in range(len(bone_cam)):
        save_bones(bone_cam[idx], len_max=scale, path=save_path + '_bone_cam.obj')
    
    print("Trimesh : Visualize Meshes")
    mesh_rest.export(save_path + '_mesh_rest.obj')
    for idx in range(len(mesh_cam)):
        mesh_cam[idx].export(save_path +'_mesh_cam.obj')
    
    

def visualize_2d(mask, valid_x, valid_y, img_size, save_path=''):
    plt.clf()
    fig = plt.figure(figsize=(8, 8))
    mask_vis = mask.cpu().numpy().reshape(img_size, img_size)
    plt.imshow(mask_vis)
    for x, y in zip(valid_x, valid_y):
        plt.scatter(x, y)
    plt.savefig('visualize/vis_2d.png' if save_path =='' else save_path + '_vis_2d.png')
    import cv2
    cv2.imwrite(save_path + '_vis_2d_cv2.png',np.uint8(mask_vis))
    
def draw_ray(ray, valid, ray_alphas, skinning_weight, ray_origin=None, scale=None, ray_split=1, save_path='', vis_ray_origin=False):
    """
    ray = [# frame, img_size, img_size, 3]
    valid = (valid_x, valid_y)
    ray_origin = [# frame, ?]
    ray_split = how many points to skip visualization

    visualize method
        1. cylinder (line)
        2. Sphere (Point)
    """

    def weight_pts(alpha, max_val, min_val):
        length = max_val - min_val
        standardized = (alpha - min_val) / length
        return np.digitize(standardized, [0.2, 0.4, 0.6, 0.8]) / 4

    valid_x, valid_y = valid
    colormap = palette()
    bone_colormap = label_colormap()
    valid_x = valid_x[::ray_split]
    valid_y = valid_y[::ray_split]

    ray_meshes = []
    ray_meshes_weight = []

    ray = ray.view(ray.shape[0], ray.shape[1], ray.shape[2], -1, 3).numpy()
    skinning_weight = skinning_weight.view(ray.shape[0], ray.shape[1], ray.shape[2], -1, 25).numpy()
    ray_cut_start = -1
    ray_cut_end = -1

    # visualize skinning weight distribution
    ray_cut_start = 65
    ray_cut_end = 75
    ray = ray[:, :, :, ray_cut_start:ray_cut_end]
    skinning_weight = skinning_weight[:, :, :, ray_cut_start:ray_cut_end]
    skinning_weight_ex = skinning_weight[0, valid_y[0], valid_x[0]]
    ex_color = matplotlib.cm.get_cmap('hsv') 
    plt.clf()
    fig = plt.figure(figsize=(8, 8))
    for i, weights in enumerate(skinning_weight_ex):
        plt.plot(weights, label=f'{i + ray_cut_start}', linestyle='--', marker='o', color=ex_color(i/len(skinning_weight_ex)))
    plt.legend()
    plt.savefig(f'{save_path}_ray{ray_cut_start}to{ray_cut_end}_skinning_weight_dist.png')

    skinning_weight = np.argmax(skinning_weight, axis=-1)
    alphas = ray_alphas.numpy()
    max_alpha = alphas.ravel().max()
    min_alpha = alphas.ravel().min()
    cylinder_radius = scale * 0.001
    pts_radius = scale * 0.005

    alpha_plot = []
    for idx, (x, y) in enumerate(zip(valid_x, valid_y)):
        ray_palette = colormap[idx % len(colormap)]
        ray_color = matplotlib.cm.get_cmap(ray_palette)

        ray_pxs = ray[0, y, x, :, :]
        alpha_pxs = alphas[0, y, x, :]
        weight = weight_pts(alpha_pxs, max_alpha, min_alpha)
        
        alpha_plot.append(alpha_pxs.ravel())
        
        ray_skin_color = skinning_weight[0, y, x, :]
        for i in range(len(ray_pxs)):
            if i > 0:
                cylinder_color = ray_color((weight[i] + weight[i-1])/2)
                segment = np.stack([ray_pxs[i-1], ray_pxs[i]])
                line = trimesh.creation.cylinder(
                    cylinder_radius,
                    segment=segment,
                    sections=5, vertex_colors=cylinder_color
                )
                ray_meshes.append(line)
            pts_color = ray_color(weight[i])
            sphere = trimesh.creation.uv_sphere(radius=pts_radius, count=[4, 4])
            sphere.visual.vertex_colors = pts_color
            sphere.vertices += ray_pxs[i]
            ray_meshes.append(sphere)

            sphere_skin = trimesh.creation.uv_sphere(radius=pts_radius, count=[4, 4])
            pts_skin_color = bone_colormap[ray_skin_color[i]]
            # pts_skin_color = np.tile(pts_skin_color[:, None], (1, len(sphere_skin.vertices), 1)).reshape(-1, 3)
            sphere_skin.visual.vertex_colors = pts_skin_color
            sphere_skin.vertices += ray_pxs[i]
            ray_meshes_weight.append(sphere_skin)

    alpha_plot = np.array(alpha_plot)
    for alpha_ray in alpha_plot:
        plt.plot(alpha_ray.ravel())
    plt.savefig(save_path + '_weight.png')
    plt.clf()

    if not isinstance(ray_origin, type(None)):
        save_path += f'_ray{ray_cut_start}to{ray_cut_end}_ray_camera.obj' if ray_cut_start > 0 else '_ray_camera.obj'
        if vis_ray_origin:
            ray_origin = ray_origin.cpu().numpy()[:, 0, 0, :]
            radius = scale * 0.02
            cmap = matplotlib.cm.get_cmap('cool')
            color_list = np.asarray(range(len(ray_origin)))/float(len(ray_origin)) 
            for idx, one_origin in enumerate(ray_origin):
                axis = trimesh.creation.axis(origin_size=radius, origin_color=cmap(color_list[idx]),
                                            axis_radius=radius*0.1, axis_length=radius*3)
                axis.visual.vertex_colors = cmap(color_list[idx])
                axis.vertices += one_origin
                ray_meshes.append(axis)
    else:
        save_path += f'_ray{ray_cut_start}to{ray_cut_end}_ray_canonical.obj' if ray_cut_start > 0 else '_ray_canonical.obj'

    meshes = trimesh.util.concatenate(ray_meshes)
    meshes.export(save_path)
    meshes_skinning = trimesh.util.concatenate(ray_meshes_weight)
    meshes_skinning.export(save_path.replace('.obj', '_skinning.obj'))

    return meshes
    # else:
    #     meshes = trimesh.util.concatenate(ray_meshes)
    #     meshes.export('visualize/ray_canonical.obj' if save_path == '' else save_path + '_ray_canonical.obj')
    #     return meshes

def main(_):

    ############## CUSTOM ###########
    mode = 'trimesh'                # support_list = ['matplotlib', 'trimesh']
    ray_select_mode = 'xrange'       # support = ['mask_range', 'xrange', 'yrange', 'slide', 'select']
    animate = False                 
    vis_ray_origin = False
    vis_2d = True 

    opts.frame_chunk = 1
    opts.rnd_frame_chunk =1
    opts.render_size = 64
    #################################
    print(opts)

    trainer = v2s_trainer(opts, is_eval=True)
    data_info = trainer.init_dataset()    
    trainer.define_model(data_info)

    ############## CUSTOM ###########
    trainer.model.img_size = 64
    savename = 'visualize/'
    savename += opts.model_path.split('logdir/')[-1].split('/params')[0]
    savename += '_epoch' + opts.model_path.split('params')[-1].split('.pth')[0]

    frame_start = 136
    frame_end = 137

    # if 'ft2' in opts.model_path:
    #     active_sampling = True
    #################################

    # print(savename)
    # exit()

    dynamic_mesh = opts.flowbw or opts.lbs
    idx_render = str_to_frame(opts.test_frames, data_info)
    # print(type(idx_render), idx_render)

    ########################### 
    chunk = opts.frame_chunk
    # for i in range(0, len(idx_render), chunk):
    for i in range(frame_start, frame_end, chunk):
        rendered_seq, aux_seq = trainer.vis_ray(idx_render=idx_render[i:i+chunk],
                                             dynamic_mesh=dynamic_mesh)

        save_visualize(rendered_seq, aux_seq, savename + '_' + str(i),
                       mode=mode, 
                       ray_select_mode=ray_select_mode,
                       animate=animate,
                       vis_2d=vis_2d,
                       vis_ray_origin=vis_ray_origin
                       )
        
        if i - frame_start > 3:
            break
        # break

if __name__ == '__main__':
    app.run(main)
