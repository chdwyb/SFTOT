import os
import random
import argparse
import math
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F


def parse_args_unpaired_training(input_args=None):


    parser = argparse.ArgumentParser(description="CycleGAN-Turbo with optional SFTOT/WGAN losses.")

    parser.add_argument("--config", type=str, help="YAML defaults; explicit CLI options take precedence")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="GPU requires an explicit CLI selection")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--resume_checkpoint", type=str, help="Existing checkpoint file to resume from")
    parser.add_argument("--allow_weights_only_resume", action="store_true", help="Allow generator-only checkpoints with fresh discriminator and optimizer states")


    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")


    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_idt", default=1, type=float)
    parser.add_argument("--lambda_cycle", default=1, type=float)
    parser.add_argument("--lambda_cycle_lpips", default=10.0, type=float)
    parser.add_argument("--lambda_idt_lpips", default=1.0, type=float)


    parser.add_argument("--lambda_transport", type=float, default=0.1)
    parser.add_argument("--lambda_frequency", type=float, default=0.1)
    parser.add_argument("--lambda_identity_ssim", type=float, default=0.1)
    parser.add_argument("--lambda_gp", type=float, default=10.0)
    parser.add_argument("--fft_norm", choices=["backward", "ortho", "forward"], default="ortho")


    parser.add_argument("--dataset_folder", type=str)
    parser.add_argument("--train_img_prep")
    parser.add_argument("--val_img_prep")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--max_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)

    parser.add_argument("--data_layout", choices=["auto", "upstream", "split"], default="auto")
    parser.add_argument("--caption_src", help="Fallback only when fixed_prompt_a.txt is absent")
    parser.add_argument("--caption_tgt", help="Fallback only when fixed_prompt_b.txt is absent")
    parser.add_argument("--paired_eval", action="store_true", help="Explicitly enable paired RGB evaluation; never pixel supervision in training")
    parser.add_argument("--paired_eval_manifest", help="Optional JSON mapping for paired evaluation GT paths")


    parser.add_argument("--pretrained_model_name_or_path", default="stabilityai/sd-turbo")
    parser.add_argument("--lora_rank_unet", default=128, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)


    parser.add_argument("--viz_freq", type=int, default=20)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--tracker_project_name", type=str)
    parser.add_argument("--validation_steps", type=int, default=500,)
    parser.add_argument("--validation_num_images", type=int, default=-1, help="Number of images to use for validation. -1 to use all images.")
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--recovery_steps", type=int, default=500)


    parser.add_argument("--learning_rate", type=float, default=5e-6,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help=(
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.",)
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)


    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--gradient_checkpointing", action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")

    preliminary, _ = parser.parse_known_args(input_args)
    if preliminary.config:
        import yaml
        with open(preliminary.config, encoding="utf-8") as stream:
            defaults = yaml.safe_load(stream)
        if not isinstance(defaults, dict):
            parser.error("config must contain a YAML mapping")
        actions = {action.dest: action for action in parser._actions}
        unknown = set(defaults) - (set(actions) - {"help", "config"})
        if unknown:
            parser.error(f"Unknown config keys: {sorted(unknown)}")
        if "device" in defaults:
            parser.error("device must be selected explicitly on CLI, not in YAML")
        for name, value in defaults.items():
            action = actions[name]
            if value is None:
                continue
            if isinstance(action, argparse._StoreTrueAction):
                if not isinstance(value, bool):
                    parser.error(f"{name} must be a YAML boolean")
            elif action.type:
                try:
                    defaults[name] = action.type(value)
                except (TypeError, ValueError):
                    parser.error(f"Invalid config value for {name}: {value!r}")
            if action.choices and defaults[name] not in action.choices:
                parser.error(f"Invalid choice for {name}: {value!r}")
        parser.set_defaults(**defaults)
    args = parser.parse_args(input_args)
    for name in ("dataset_folder", "train_img_prep", "val_img_prep", "output_dir", "tracker_project_name"):
        if not getattr(args, name):
            parser.error(f"{name} is required; set it in YAML or on the CLI")
    for name in ("lambda_transport", "lambda_frequency", "lambda_identity_ssim", "lambda_gp"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"{name} must be finite and nonnegative")
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        parser.error("max_train_steps must be positive")
    if args.resume_checkpoint is not None and not os.path.isfile(args.resume_checkpoint):
        parser.error("resume_checkpoint must be an existing file")
    if args.allow_weights_only_resume and args.resume_checkpoint is None:
        parser.error("allow_weights_only_resume requires resume_checkpoint")
    return args


def build_transform(image_prep):

    if image_prep == "resized_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    elif image_prep == "resize_286_randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.Resize((286, 286), interpolation=Image.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep in ["resize_256", "resize_256x256"]:
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS)
        ])
    elif image_prep == "no_resize":
        T = transforms.Compose([])
    else:
        raise ValueError(f"Unsupported image preprocessing: {image_prep!r}")
    return T


def list_image_files(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory not found: {folder}")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
    return sorted(str(path) for path in folder.iterdir()
                  if path.is_file() and path.suffix.lower() in extensions)


def resolve_dataset_folders(dataset_folder, split, data_layout="auto"):

    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    if data_layout not in {"auto", "upstream", "split"}:
        raise ValueError("data_layout must be auto, upstream, or split")
    candidates = []
    if data_layout in {"auto", "upstream"}:
        candidates.append((os.path.join(dataset_folder, f"{split}_A"),
                           os.path.join(dataset_folder, f"{split}_B")))
    if data_layout in {"auto", "split"}:
        candidates.append((os.path.join(dataset_folder, split, "input"),
                           os.path.join(dataset_folder, split, "target")))
    for source, target in candidates:
        if os.path.isdir(source) and os.path.isdir(target):
            return source, target
        if os.path.exists(source) or os.path.exists(target):
            raise FileNotFoundError(f"Incomplete dataset domains: {source} and {target}")
    raise FileNotFoundError(f"No {split} domains found for layout {data_layout!r}: {candidates}")


class UnpairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, image_prep, tokenizer,
                 data_layout="auto", caption_src=None, caption_tgt=None):

        super().__init__()
        self.source_folder, self.target_folder = resolve_dataset_folders(dataset_folder, split, data_layout)
        self.tokenizer = tokenizer
        prompt_a = os.path.join(dataset_folder, "fixed_prompt_a.txt")
        prompt_b = os.path.join(dataset_folder, "fixed_prompt_b.txt")
        if os.path.isfile(prompt_a):
            with open(prompt_a, "r", encoding="utf-8-sig") as f:
                self.fixed_caption_src = f.read().strip()
        elif caption_src is not None:
            self.fixed_caption_src = caption_src
        else:
            raise FileNotFoundError(f"Missing {prompt_a}; provide caption_src as an explicit fallback")
        if not self.fixed_caption_src.strip():
            raise ValueError("The source-domain prompt must not be empty")
        self.input_ids_src = self.tokenizer(
            self.fixed_caption_src, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids
        if os.path.isfile(prompt_b):
            with open(prompt_b, "r", encoding="utf-8-sig") as f:
                self.fixed_caption_tgt = f.read().strip()
        elif caption_tgt is not None:
            self.fixed_caption_tgt = caption_tgt
        else:
            raise FileNotFoundError(f"Missing {prompt_b}; provide caption_tgt as an explicit fallback")
        if not self.fixed_caption_tgt.strip():
            raise ValueError("The target-domain prompt must not be empty")
        self.input_ids_tgt = self.tokenizer(
            self.fixed_caption_tgt, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        self.l_imgs_src = list_image_files(self.source_folder)
        self.l_imgs_tgt = list_image_files(self.target_folder)
        if not self.l_imgs_src or not self.l_imgs_tgt:
            raise ValueError("Both dataset domains must contain images")
        self.T = build_transform(image_prep)

    def __len__(self):

        return len(self.l_imgs_src) + len(self.l_imgs_tgt)

    def __getitem__(self, index):

        if index < len(self.l_imgs_src):
            img_path_src = self.l_imgs_src[index]
        else:
            img_path_src = random.choice(self.l_imgs_src)
        img_path_tgt = random.choice(self.l_imgs_tgt)
        with Image.open(img_path_src) as image:
            img_pil_src = image.convert("RGB")
        with Image.open(img_path_tgt) as image:
            img_pil_tgt = image.convert("RGB")
        img_t_src = F.to_tensor(self.T(img_pil_src))
        img_t_tgt = F.to_tensor(self.T(img_pil_tgt))
        img_t_src = F.normalize(img_t_src, mean=[0.5], std=[0.5])
        img_t_tgt = F.normalize(img_t_tgt, mean=[0.5], std=[0.5])
        return {
            "pixel_values_src": img_t_src,
            "pixel_values_tgt": img_t_tgt,
            "caption_src": self.fixed_caption_src,
            "caption_tgt": self.fixed_caption_tgt,
            "input_ids_src": self.input_ids_src,
            "input_ids_tgt": self.input_ids_tgt,
        }
