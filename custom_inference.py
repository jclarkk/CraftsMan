import argparse
import numpy as np
import os
import torch
import trimesh
from PIL import Image
from huggingface_hub import hf_hub_download

from apps.utils import load_model
from mvdiffusion import image_process
from mvdiffusion.mvdiffusion_unclip import run_era3d_pipeline


def prepare_image(tensor):
    array = tensor.squeeze()  # Remove any singleton dimensions
    array = np.moveaxis(array, 0, -1)  # Change shape from (C, H, W) to (H, W, C)
    array = (array * 255).astype(np.uint8)  # Rescale and convert to uint8
    return array


def image2mesh(view_front: np.ndarray,
               view_right: np.ndarray,
               view_back: np.ndarray,
               view_left: np.ndarray,
               img_filename: str,
               output_folder: str,
               remesh: bool = False,
               texture: bool = False,
               target_face_count: int = 2000,
               scheluder_name: str = "DDIMScheduler",
               guidance_scale: int = 7.5,
               step: int = 50,
               seed: int = 4,
               octree_depth: int = 7):
    sample_inputs = {
        "mvimages": [[
            Image.fromarray(prepare_image(view_front)),
            Image.fromarray(prepare_image(view_right)),
            Image.fromarray(prepare_image(view_back)),
            Image.fromarray(prepare_image(view_left))
        ]]
    }

    ckpt_path = hf_hub_download(repo_id="wyysf/CraftsMan",
                                filename="image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6-aligned-vae/model.ckpt",
                                repo_type="model")
    config_path = hf_hub_download(repo_id="wyysf/CraftsMan",
                                  filename="image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6-aligned-vae/config.yaml",
                                  repo_type="model")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    global model
    model = load_model(ckpt_path, config_path, device)

    latents = model.sample(
        sample_inputs,
        sample_times=1,
        guidance_scale=guidance_scale,
        return_intermediates=False,
        steps=step,
        seed=seed
    )[0]

    # decode the latents to mesh
    box_v = 1.1
    mesh_outputs, _ = model.shape_model.extract_geometry(
        latents,
        bounds=[-box_v, -box_v, -box_v, box_v, box_v, box_v],
        octree_depth=octree_depth
    )
    # Verify folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Export the mesh only
    filepath = os.path.join(output_folder, f"{img_filename}_mesh.obj")

    assert len(mesh_outputs) == 1, "Only support single mesh output for gradio demo"
    mesh = trimesh.Trimesh(mesh_outputs[0][0], mesh_outputs[0][1])
    mesh.export(filepath, include_normals=True)

    if remesh:
        remeshed_filepath = os.path.join(output_folder, f"{img_filename}_remesh.obj")
        print("Remeshing with Instant Meshes...")
        command = f"apps/third_party/InstantMeshes {filepath} -f {target_face_count} -o {remeshed_filepath}"
        os.system(command)


def parse_args():
    parser = argparse.ArgumentParser(description='Reconstruction CLI Tool')
    parser.add_argument('--image', type=str, help='Input image file')
    parser.add_argument("--seed", type=int, default=9, help="Random seed for generating multi-view images", )
    parser.add_argument("--remesh", type=bool, default=False, help="Remesh the output mesh", )
    parser.add_argument("--target_face_count", type=int, default=2000, help="Target face count for remeshing", )
    parser.add_argument("--guidance_scale_3D", type=float, default=3, help="Guidance scale for 3D reconstruction", )
    parser.add_argument("--step_3D", type=int, default=50, help="Number of steps for 3D reconstruction", )
    parser.add_argument("--octree_depth", type=int, default=7, help="Octree depth for 3D reconstruction", )
    parser.add_argument('--output_folder', type=str, default='./output', help='Output folder')
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def generate_3d(args):
    # Get image filename without extension
    img_filename = os.path.splitext(os.path.basename(args.image))[0]

    # Clean BG first
    input_img = image_process.process(args.image)
    # Generate multi view diffusion
    normal_images, rgb_images = run_era3d_pipeline(input_img)

    front_image = rgb_images['front'].detach().cpu().numpy()
    right_image = rgb_images['right'].detach().cpu().numpy()
    back_image = rgb_images['back'].detach().cpu().numpy()
    left_image = rgb_images['left'].detach().cpu().numpy()

    image2mesh(
        front_image,
        right_image,
        back_image,
        left_image,
        img_filename,
        args.output_folder,
        args.remesh,
        args.texture,
        args.target_face_count,
        guidance_scale=args.guidance_scale_3D,
        step=args.step_3D,
        seed=args.seed,
        octree_depth=args.octree_depth
    )


if __name__ == '__main__':
    args = parse_args()

    generate_3d(args)
