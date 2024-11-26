import json

import base64

import os

import argparse
import requests
import trimesh

from craftsman import CraftsManPipeline
import torch

TEXTURE_SERVER_URL = "algodemo.sz.lightions.top:31025"


def parse_args():
    parser = argparse.ArgumentParser(description='Reconstruction CLI Tool')
    parser.add_argument('--image', type=str, help='Input image file')
    parser.add_argument("--seed", type=int, default=9, help="Random seed for generating multi-view images", )
    parser.add_argument("--remesh", type=bool, default=False, help="Remesh the output mesh", )
    parser.add_argument("--target_face_count", type=int, default=10000, help="Target face count for remeshing", )
    parser.add_argument("--guidance_scale_3D", type=float, default=3, help="Guidance scale for 3D reconstruction", )
    parser.add_argument("--step_3D", type=int, default=50, help="Number of steps for 3D reconstruction", )
    parser.add_argument("--mc_depth", type=int, default=8, help="Octree depth for 3D reconstruction", )
    parser.add_argument('--output_folder', type=str, default='./output', help='Output folder')
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def generate_3d(args):
    pipeline = CraftsManPipeline.from_pretrained(
        "craftsman3d/craftsman",
        device="cuda:0",
        torch_dtype=torch.float32)

    image_file = args.image

    # Check image is valid
    if not os.path.exists(image_file):
        raise FileNotFoundError(f"Image file not found: {image_file}")

    geo_file = os.path.join(args.output_folder, "geo.obj")
    textured_file = os.path.join(args.output_folder, "textured.glb")

    # Geo Inference
    mesh = pipeline(
        image_file,
        num_inference_steps=args.step_3D,
        guidance_scale=args.guidance_scale_3D,
        mc_depth=args.mc_depth,
    ).meshes[0]

    # Save output
    mesh.export(geo_file)

    if args.remesh:
        remeshed_filepath = os.path.join(args.output_folder, f"remesh.obj")
        print("Remeshing with Instant Meshes...")
        command = f"InstantMeshes {geo_file} -f {args.target_face_count} -o {remeshed_filepath}"
        os.system(command)

        geo_file = remeshed_filepath

    # Convert the mesh to glb
    geo_glb_file = os.path.join(args.output_folder, "geo.glb")
    trimesh.load_mesh(geo_file).export(geo_glb_file)

    ########## For texture generation, currently only support the API ##########
    with open(image_file, 'rb') as f:
        image_bytes = f.read()
    with open(geo_glb_file, 'rb') as f:
        mesh_bytes = f.read()
    request = {
        'png_base64_image': base64.b64encode(image_bytes).decode('utf-8'),
        'glb_base64_mesh': base64.b64encode(mesh_bytes).decode('utf-8'),
    }
    response = requests.post(
        url=f"http://{TEXTURE_SERVER_URL}/generate_texture",
        headers={'Content-Type': 'application/json'},
        data=json.dumps(request),
    ).json()
    mesh_bytes = base64.b64decode(response['glb_base64_mesh'])
    with open(textured_file, 'wb') as f:
        f.write(mesh_bytes)

    print(f"Generated textured mesh: {textured_file}")


if __name__ == '__main__':
    args = parse_args()

    generate_3d(args)
