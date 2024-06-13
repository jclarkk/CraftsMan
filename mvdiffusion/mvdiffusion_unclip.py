import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, List

import torch
import torch.utils.checkpoint
from PIL import Image
from accelerate.utils import set_seed
from einops import rearrange
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from mvdiffusion.data.single_image_dataset import SingleImageDataset
from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline

weight_dtype = torch.float16


def tensor_to_numpy(tensor):
    return tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()


@dataclass
class TestConfig:
    pretrained_model_name_or_path: str
    pretrained_unet_path: Optional[str]
    revision: Optional[str]
    validation_dataset: Dict
    save_dir: str
    seed: Optional[int]
    validation_batch_size: int
    dataloader_num_workers: int
    # save_single_views: bool
    save_mode: str
    local_rank: int

    pipe_kwargs: Dict
    pipe_validation_kwargs: Dict
    unet_from_pretrained_kwargs: Dict
    validation_guidance_scales: List[float]
    validation_grid_nrow: int
    camera_embedding_lr_mult: float

    num_views: int
    camera_embedding_type: str

    pred_type: str  # joint, or ablation
    regress_elevation: bool
    enable_xformers_memory_efficient_attention: bool

    cond_on_normals: bool
    cond_on_colors: bool

    regress_elevation: bool
    regress_focal_length: bool


def convert_to_numpy(tensor):
    return tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()


def save_image(tensor, fp):
    ndarr = convert_to_numpy(tensor)
    # pdb.set_trace()
    save_image_numpy(ndarr, fp)
    return ndarr


def save_image_numpy(ndarr, fp):
    im = Image.fromarray(ndarr)
    im.save(fp)


def generate_images(dataloader, pipeline, cfg: TestConfig, views):
    pipeline.set_progress_bar_config(disable=True)

    if cfg.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=pipeline.unet.device).manual_seed(cfg.seed)

    images_cond, pred_cat = [], defaultdict(list)

    normal_imgs = {}
    rgb_imgs = {}

    for _, batch in tqdm(enumerate(dataloader)):
        images_cond.append(batch['imgs_in'][:, 0])

        imgs_in = torch.cat([batch['imgs_in']] * 2, dim=0)
        num_views = imgs_in.shape[1]
        imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W")  # (B*Nv, 3, H, W)

        normal_prompt_embeddings, clr_prompt_embeddings = batch['normal_prompt_embeddings'], batch[
            'color_prompt_embeddings']
        prompt_embeddings = torch.cat([normal_prompt_embeddings, clr_prompt_embeddings], dim=0)
        prompt_embeddings = rearrange(prompt_embeddings, "B Nv N C -> (B Nv) N C")

        with torch.autocast("cuda"):
            # B*Nv images
            for guidance_scale in cfg.validation_guidance_scales:
                unet_out = pipeline(
                    imgs_in, None, prompt_embeds=prompt_embeddings,
                    generator=generator, guidance_scale=guidance_scale, output_type='pt', num_images_per_prompt=1,
                    **cfg.pipe_validation_kwargs
                )

                out = unet_out.images
                bsz = out.shape[0] // 2

                normals_pred = out[:bsz]
                images_pred = out[bsz:]

                pred_cat[f"cfg{guidance_scale:.1f}"].append(
                    torch.cat([normals_pred, images_pred], dim=-1))  # b, 3, h, w

                for i in range(bsz // num_views):
                    img_in_ = images_cond[-1][i].to(out.device)
                    vis_ = [img_in_]
                    for j in range(num_views):
                        view = views[j]
                        idx = i * num_views + j
                        normal = normals_pred[idx]
                        color = images_pred[idx]
                        vis_.append(color)
                        vis_.append(normal)

                        normal_imgs[view] = normal
                        rgb_imgs[view] = color
    torch.cuda.empty_cache()

    return normal_imgs, rgb_imgs


def load_era3d_pipeline(cfg):
    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained('./ckpt/MacLab-Era3D-512-6view', torch_dtype=weight_dtype)
    pipeline.unet.enable_xformers_memory_efficient_attention()
    if torch.cuda.is_available():
        pipeline.to('cuda:0')
    return pipeline


def run_era3d_pipeline(in_image, views=['front', 'front_right', 'right', 'back', 'left', 'front_left']):
    from mvdiffusion.utils.misc import load_config

    # Get current script directory
    script_dir = os.path.dirname(__file__)

    cfg = load_config(os.path.join(script_dir, 'test_unclip-512-6view.yaml'))
    schema = OmegaConf.structured(TestConfig)
    cfg = OmegaConf.merge(schema, cfg)

    if cfg.seed is not None:
        set_seed(cfg.seed)
    pipeline = load_era3d_pipeline(cfg)
    if torch.cuda.is_available():
        pipeline.to('cuda:0')

    # Get the  dataset
    validation_dataset = SingleImageDataset(
        **cfg.validation_dataset,
        single_image=in_image
    )
    # DataLoaders creation:
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=cfg.validation_batch_size, shuffle=False, num_workers=cfg.dataloader_num_workers
    )

    return generate_images(validation_dataloader, pipeline, cfg, views)
