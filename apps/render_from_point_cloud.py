import torch
import numpy as np
import cv2

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor
)

def set_renderer():
    # Setup
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Initialize a camera.
    R, T = look_at_view_transform(dist=22.0, elev=0, azim=0, device=device)
    cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)

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

def generate_video_from_obj(obj_path, video_path, renderer):
    # Setup
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Load obj file
    verts_rgb_colors = get_verts_rgb_colors(obj_path)
    verts_rgb_colors = torch.from_numpy(verts_rgb_colors).to(device)
    mesh = load_objs_as_meshes([obj_path], device=device)

    # Set point cloud object
    verts_list = mesh._verts_list
    point_cloud = Pointclouds(points=verts_list[0].unsqueeze(0), features=verts_rgb_colors)

    # Render 2D image from point cloud
    images_w_tex = renderer(point_cloud)
    images_w_tex = np.clip(images_w_tex[0, ..., :3].cpu().numpy(), 0.0, 1.0)[:, :, ::-1] * 255
    cv2.imwrite(video_path, images_w_tex)


if __name__ == '__main__':
    obj_path = '../results/horse_2_test/stage2.obj'
    video_path = '../results/horse_2_test/stage2_render_tester.jpg'
    renderer = set_renderer()
    

    generate_video_from_obj(obj_path, video_path, renderer)