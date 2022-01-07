import torch
import numpy as np
import cv2

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_ply

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    TexturesVertex,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor
)

def set_renderer():
    # Setup
    device = torch.device("cuda:2")
    torch.cuda.set_device(device)

    # Initialize a camera.
    R, T = look_at_view_transform(1, 0, 0)
    R = torch.tensor([[[-1.],[ 0.],[ 0.]],[[ 0.],[ 1.],[ 0.]],[[ 0.],[ 0.],[-1.]]]).reshape(1,3,3)
    camera_tensor = torch.load('../results/horse_1_test/stage1_camera_tensor.pt')
    extrinsic_tensor = torch.load('../results/horse_1_test/stage1_extrinsic_tensor.pt')
    R_v1 = camera_tensor[..., :, 5:14][0][0].reshape(1, 3, 3)
    extrinsic_tensor_v1 = extrinsic_tensor[0].unsqueeze(0)
    T_v1 = extrinsic_tensor_v1[0][0:3, 3].unsqueeze(0) / extrinsic_tensor_v1[0][0:3, 3][1]
    T[0][0] = -0
    T[0][1] = -0.5
    T[0][2] = 100
    cameras = FoVOrthographicCameras(device=device, R=R_v1, T=T)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters.
    raster_settings = PointsRasterizationSettings(
        image_size=512,
        radius=0.003,
        points_per_pixel=10
    )

    # Create a points renderer by compositing points using an alpha compositor (nearer points
    # are weighted more heavily). See [1] for an explanation.
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )

    return renderer

def get_verts_rgb_colors(obj_path):
    rgb_colors = []

    f = open(obj_path)
    lines = f.readlines()
    for line in lines:
        ls = line.split(' ')
        if len(ls) == 7:
            rgb_colors.append(ls[-3:])

    return np.array(rgb_colors, dtype='float32')[None, :, :]

def get_ply_rgb_colors(obj_path):
    rgb_colors = []

    f = open(obj_path)
    lines = f.readlines()
    for i, line in enumerate(lines):
        if i>9:
            ls = line.split(' ')
            if len(ls) == 6:
                rgb_colors.append(ls[-3:])

    return np.array(rgb_colors, dtype='float32')[None, :, :]

def generate_video_from_obj(obj_path,ply_path, video_path, renderer):
    # Setup
    device = torch.device("cuda:2")
    torch.cuda.set_device(device)

    # Load ply file
    points = load_ply(ply_path)[0].clone().detach().unsqueeze(0).to(device)
    ply_rgb_colors = get_ply_rgb_colors(ply_path)
    ply_rgb_colors = torch.from_numpy(ply_rgb_colors).to(device)

    # Load obj file
    #mesh = load_objs_as_meshes([obj_path], device=device)
    #verts_rgb_colors = get_verts_rgb_colors(obj_path)
    #verts_rgb_colors = torch.from_numpy(verts_rgb_colors).to(device)
    #verts_list = mesh._verts_list

    # Set point cloud object
    point_cloud = Pointclouds(points=points / 50, features=ply_rgb_colors)
    images_w_tex = renderer(point_cloud)
    images_w_tex = np.clip(images_w_tex[0, ..., :3].cpu().numpy(), 0.0, 1.0)[:, :, ::-1] * 255
    cv2.imwrite(video_path, images_w_tex)


if __name__ == '__main__':
    obj_path = '../results/horse_1_test/stage1_pred_col.obj'
    ply_path = '../results/horse_1_test/stage1_pred_col.ply'
    video_path = '../results/horse_1_test/stage1_render_tester.jpg'
    renderer = set_renderer()

    generate_video_from_obj(obj_path, ply_path, video_path, renderer)