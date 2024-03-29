import torch
import numpy as np
import cv2

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    TexturesVertex
)

def set_renderer():
    # Setup
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Initialize an OpenGL perspective camera.
    R, T = look_at_view_transform(dist=2.0, elev=0, azim=0, device=device)
    cameras = FoVOrthographicCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=512, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
        bin_size = None, 
        max_faces_per_bin = None
    )

    lights = PointLights(device=device, location=((2.0, 2.0, 2.0),))

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
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
    textures = TexturesVertex(verts_features=verts_rgb_colors)
    #wo_textures = TexturesVertex(verts_features=torch.ones_like(verts_rgb_colors)*0.75)

    # Load obj
    mesh = load_objs_as_meshes([obj_path], device=device)

    # Set mesh
    verts_list = mesh._verts_list
    faces_list = mesh._faces_list
    mesh_w_tex = Meshes(verts_list, faces_list, textures)
    #mesh_wo_tex = Meshes(verts, faces, wo_textures)

    # create VideoWriter
    #fourcc = cv2. VideoWriter_fourcc(*'MP4V')
    #out = cv2.VideoWriter(video_path, fourcc, 20.0, (512,512))

    #for i in range(1):
        #R, T = look_at_view_transform(dist=2.0, elev=0, azim=0, device=device))
        #images_w_tex = renderer(mesh_w_tex, R=R, T=T)
        #images_w_tex = np.clip(images_w_tex[0, ..., :3].cpu().numpy(), 0.0, 1.0)[:, :, ::-1] * 255
        #images_wo_tex = renderer(mesh_wo_tex, R=R, T=T)
        #images_wo_tex = np.clip(images_wo_tex[0, ..., :3].cpu().numpy(), 0.0, 1.0)[:, :, ::-1] * 255
        #image = np.concatenate([images_w_tex, images_wo_tex], axis=1)
        #out.write(images_w_tex.astype('uint8'))
    #out.release()

    # save image
    R, T = look_at_view_transform(dist=2.0, elev=0, azim=0, device=device)
    cameras = FoVOrthographicCameras(device=device, R=R, T=T)
    images_w_tex = renderer(mesh_w_tex, cameras=cameras)
    images_w_tex = np.clip(images_w_tex[0, ..., :3].cpu().numpy(), 0.0, 1.0)[:, :, ::-1] * 255
    cv2.imwrite(video_path, images_w_tex)

if __name__ == '__main__':
    obj_path = '../results/horse_2_test/stage2.obj'
    video_path = '../results/horse_2_test/stage2_render_tester.jpg'
    renderer = set_renderer()
    

    generate_video_from_obj(obj_path, video_path, renderer)