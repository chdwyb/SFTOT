import argparse
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms
from tqdm.auto import tqdm

from cyclegan_turbo import CycleGAN_Turbo
from restoration_metrics import PairedMetrics
from training_utils import build_transform, list_image_files


def main():
    parser = argparse.ArgumentParser(description="Restore one image or a folder with a trained SFTOT checkpoint.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input_image", help="Single input image")
    inputs.add_argument("--input_dir", help="Input image folder (non-recursive)")
    parser.add_argument("--checkpoint", required=True, help="Generator checkpoint written by training")
    parser.add_argument("--direction", choices=("a2b", "b2a"), required=True)
    parser.add_argument("--prompt", required=True, help="Target-domain prompt, matching the training prompt")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--image_prep", default="no_resize", choices=(
        "no_resize", "resize_256", "resize_256x256", "resize_512", "resize_512x512", "resized_crop_512"))
    parser.add_argument("--target_dir", help="Optional paired ground-truth folder for RGB PSNR/SSIM")
    parser.add_argument("--paired_eval_manifest", help="Optional JSON input-to-target filename mapping")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--pretrained_model_name_or_path", default="stabilityai/sd-turbo")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    args = parser.parse_args()

    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if args.paired_eval_manifest and not args.target_dir:
        parser.error("--paired_eval_manifest requires --target_dir")
    if args.target_dir and not args.input_dir:
        parser.error("--target_dir requires --input_dir")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if args.enable_xformers_memory_efficient_attention and device.type != "cuda":
        parser.error("xformers inference requires a CUDA device")
    paths = [Path(args.input_image)] if args.input_image else [Path(path) for path in list_image_files(args.input_dir)]
    if not paths or any(not path.is_file() for path in paths):
        parser.error("Input must contain at least one existing image")
    output_dir = Path(args.output_dir)
    if any(output_dir.resolve() == path.parent.resolve() for path in paths):
        parser.error("--output_dir must differ from the input directory")
    if args.target_dir and output_dir.resolve() == Path(args.target_dir).resolve():
        parser.error("--output_dir must differ from the ground-truth directory")
    # Preserve PNG names; append .png for other formats to avoid lossy re-encoding.
    destinations = [output_dir / (path.name if path.suffix.lower() == ".png" else path.name + ".png")
                    for path in paths]
    if len({str(path).casefold() for path in destinations}) != len(destinations):
        parser.error("Input filenames collide after PNG conversion; rename the conflicting inputs")
    if any(path.exists() for path in destinations):
        parser.error("Predictions already exist in --output_dir; choose a fresh output directory")
    transform = build_transform(args.image_prep)
    metrics = None
    if args.target_dir:
        metrics = PairedMetrics(args.input_dir, paths, list_image_files(args.target_dir),
                                transform, output_dir / "paired", args.paired_eval_manifest)

    torch.manual_seed(args.seed)
    model = CycleGAN_Turbo(
        args.checkpoint, device=device,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        local_files_only=args.local_files_only)
    model.eval()
    if args.enable_xformers_memory_efficient_attention:
        model.unet.enable_xformers_memory_efficient_attention()
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for path, destination in tqdm(list(zip(paths, destinations)), desc="Restoring"):
            with Image.open(path) as image:
                input_image = image.convert("RGB")
            input_img = transform(input_image)
            if input_img.width % 8 or input_img.height % 8:
                raise ValueError(f"Input dimensions must be divisible by 8: {path} ({input_img.size}); "
                                 "select --image_prep resize_256 or resize_512 to resize explicitly")
            x_t = transforms.ToTensor()(input_img)
            x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).to(device)
            output = model(x_t, direction=args.direction, caption=args.prompt)
            if output.shape != x_t.shape:
                raise ValueError(f"Output shape {tuple(output.shape)} does not match input {tuple(x_t.shape)}")
            output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
            output_pil.save(destination)
            if metrics is not None:
                metrics.add(path, x_t, output)
    if metrics is not None:
        report = metrics.finalize(step=0)
        mean = report["mean"]
        print(f"RGB PSNR: {mean['psnr_rgb']:.4f} dB; SSIM: {mean['ssim_rgb']:.6f}; "
              f"images: {report['images']}")
        print(f"Metrics saved to {output_dir / 'paired' / 'metrics.json'}")
    print(f"Saved {len(paths)} restored image(s) to {output_dir}")


if __name__ == "__main__":
    main()
