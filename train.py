import os
import json
import math
import time
from datetime import datetime, timezone
import gc
import copy
import lpips
import torch
import wandb
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from diffusers.optimization import get_scheduler
from cleanfid.fid import get_files_features, build_feature_extractor, frechet_distance
import vision_aided_loss
from model import make_1step_sched
from cyclegan_turbo import CycleGAN_Turbo, VAE_encode, VAE_decode, initialize_unet, initialize_vae
from training_utils import UnpairedDataset, build_transform, parse_args_unpaired_training, resolve_dataset_folders, list_image_files
from dino_struct import DinoStructureLoss
from sftot_losses import generator_wgan_loss, discriminator_wgan_terms, transport_terms, identity_structure
from restoration_metrics import PairedMetrics
from runtime_io import atomic_json_write, atomic_torch_save, RunLock
from training_state import load_checkpoint, load_generator_state, export_generator_state, pack_training_state, restore_training_state, restore_rng_state

_last_status = None

def extract_fid_features(folder, model, device):
    # clean-fid's extension globs duplicate files on case-insensitive filesystems.
    return get_files_features(list_image_files(folder), model=model, num_workers=0,
                              batch_size=8, device=device, mode='clean', verbose=True)

def clear_validation_features(vae_enc):
    for vae in (vae_enc.vae, vae_enc.vae_b2a):
        vae.encoder.current_down_blocks = None
        vae.decoder.incoming_skip_acts = None

def write_status(args, state, phase, step, started_at, **details):
    global _last_status
    payload = dict(state=state, phase=phase, pid=os.getpid(), step=step, target_steps=args.max_train_steps, updated_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.time() - started_at, **details)
    path = os.path.join(args.output_dir, 'status.json')
    _last_status = payload
    return atomic_json_write(path, payload)

def main(args):
    if args.gan_disc_type != 'vagan_clip':
        raise ValueError('Only the upstream vagan_clip discriminator is supported')
    if args.gradient_accumulation_steps != 1:
        raise ValueError('The upstream three-stage optimizer sequence requires gradient_accumulation_steps=1')
    if min(args.validation_steps, args.checkpointing_steps, args.recovery_steps, args.viz_freq, args.train_batch_size) <= 0:
        raise ValueError('Step intervals and batch size must be positive')
    if args.paired_eval and args.val_img_prep == 'resize_286_randomcrop_256x256_hflip':
        raise ValueError('Paired metrics require a deterministic validation transform')
    accelerator = Accelerator(cpu=args.device == 'cpu', mixed_precision='no', gradient_accumulation_steps=args.gradient_accumulation_steps, log_with=None if args.report_to == 'none' else args.report_to)
    if accelerator.num_processes != 1:
        raise ValueError('This trainer currently supports one process; use a single CPU or GPU')
    device = accelerator.device
    use_wgan = args.gan_loss_type == 'wgan'
    model_options = dict(device=device, pretrained_model_name_or_path=args.pretrained_model_name_or_path, local_files_only=args.local_files_only)
    started_at = time.time()
    if any(os.path.isfile(os.path.join(args.output_dir, name)) for name in ('metrics.jsonl', 'status.json', 'config.json')):
        raise FileExistsError('Use a new output directory for a new training run')
    resume_payload = load_checkpoint(args.resume_checkpoint) if args.resume_checkpoint else None
    resume_full = resume_payload is not None and resume_payload.get('format') == 'sftot_training_state'
    resume_step = int(resume_payload['step']) if resume_payload is not None else 0
    if resume_payload is not None and not resume_full and not args.allow_weights_only_resume:
        raise ValueError('This checkpoint has no discriminator/optimizer state; explicitly allow a weights-only continuation')
    if resume_payload is not None and not resume_full and args.lr_scheduler != 'constant':
        raise ValueError('Weights-only continuation currently requires the constant learning-rate schedule')
    if args.max_train_steps is not None and resume_step >= args.max_train_steps:
        raise ValueError('The target step must be greater than the checkpoint step')
    write_status(args, 'running', 'loading_models', resume_step, started_at)
    set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder='tokenizer', use_fast=False, local_files_only=args.local_files_only)
    noise_scheduler_1step = make_1step_sched(**model_options)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder='text_encoder', local_files_only=args.local_files_only).to(device)
    (unet, l_modules_unet_encoder, l_modules_unet_decoder, l_modules_unet_others) = initialize_unet(args.lora_rank_unet, return_lora_module_names=True, **model_options)
    (vae_a2b, vae_lora_target_modules) = initialize_vae(args.lora_rank_vae, return_lora_module_names=True, **model_options)
    weight_dtype = torch.float32
    vae_a2b.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    if args.gan_disc_type == 'vagan_clip':
        net_disc_a = vision_aided_loss.Discriminator(cv_type='clip', loss_type=None if use_wgan else args.gan_loss_type, device=device)
        net_disc_a.cv_ensemble.requires_grad_(False)
        net_disc_b = vision_aided_loss.Discriminator(cv_type='clip', loss_type=None if use_wgan else args.gan_loss_type, device=device)
        net_disc_b.cv_ensemble.requires_grad_(False)
    (crit_cycle, crit_idt) = (torch.nn.L1Loss(), torch.nn.L1Loss())
    if args.enable_xformers_memory_efficient_attention and device.type == 'cuda':
        unet.enable_xformers_memory_efficient_attention()
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    if args.allow_tf32 and device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
    unet.conv_in.requires_grad_(True)
    vae_b2a = copy.deepcopy(vae_a2b)
    params_gen = CycleGAN_Turbo.get_traininable_params(unet, vae_a2b, vae_b2a)
    vae_enc = VAE_encode(vae_a2b, vae_b2a=vae_b2a)
    vae_dec = VAE_decode(vae_a2b, vae_b2a=vae_b2a)
    if resume_payload is not None and not resume_full:
        load_generator_state(resume_payload, unet, vae_enc, vae_dec, expected_step=resume_step)
        del resume_payload
        resume_payload = None
        gc.collect()
    optimizer_gen = torch.optim.AdamW(params_gen, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    params_disc = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
    optimizer_disc = torch.optim.AdamW(params_disc, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    dataset_train = UnpairedDataset(dataset_folder=args.dataset_folder, image_prep=args.train_img_prep, split='train', tokenizer=tokenizer, data_layout=args.data_layout, caption_src=args.caption_src, caption_tgt=args.caption_tgt)
    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    if not len(train_dataloader):
        raise ValueError('The training dataset is empty')
    if args.max_train_steps is None:
        args.max_train_steps = args.max_train_epochs * len(train_dataloader)
    if args.max_train_steps <= 0:
        raise ValueError('The training budget must be positive')
    if resume_step >= args.max_train_steps:
        raise ValueError('The target step must be greater than the checkpoint step')
    num_epochs = math.ceil((args.max_train_steps - resume_step) / len(train_dataloader))
    T_val = build_transform(args.val_img_prep)
    (test_source_folder, test_target_folder) = resolve_dataset_folders(args.dataset_folder, 'test', args.data_layout)
    fixed_caption_src = dataset_train.fixed_caption_src
    fixed_caption_tgt = dataset_train.fixed_caption_tgt
    l_images_src_test = list_image_files(test_source_folder)
    l_images_tgt_test = list_image_files(test_target_folder)
    if len(l_images_src_test) < 2 or len(l_images_tgt_test) < 2 or args.validation_num_images in (0, 1):
        raise ValueError('Upstream FID validation needs at least two images in each test domain')
    if args.paired_eval:
        PairedMetrics(args.dataset_folder, l_images_src_test, l_images_tgt_test, T_val, os.path.join(args.output_dir, 'paired', 'preflight'), args.paired_eval_manifest)
    if accelerator.is_main_process:
        atomic_json_write(os.path.join(args.output_dir, 'config.json'), vars(args), required=True)
        if args.resume_checkpoint:
            atomic_json_write(os.path.join(args.output_dir, 'resume.json'), dict(source_checkpoint=os.path.abspath(args.resume_checkpoint), source_step=resume_step, mode='training_state' if resume_full else 'weights_only', discriminator_and_optimizer_reset=not resume_full, data_resume='new_shuffled_epoch', target_steps=args.max_train_steps, remaining_iterations=args.max_train_steps - resume_step), required=True)
    write_status(args, 'running', 'fid_reference', resume_step, started_at)
    if accelerator.is_main_process:
        feat_model = build_feature_extractor('clean', device, use_dataparallel=False)
        output_dir_ref = os.path.join(args.output_dir, 'fid_reference_a2b')
        os.makedirs(output_dir_ref, exist_ok=True)
        for index, _path in enumerate(tqdm(l_images_tgt_test)):
            outf = os.path.join(output_dir_ref, f'{index}.png')
            if not os.path.exists(outf):
                _img = T_val(Image.open(_path).convert('RGB'))
                _img.save(outf)
        ref_features = extract_fid_features(output_dir_ref, feat_model, device)
        (a2b_ref_mu, a2b_ref_sigma) = (np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False))
        output_dir_ref = os.path.join(args.output_dir, 'fid_reference_b2a')
        os.makedirs(output_dir_ref, exist_ok=True)
        for index, _path in enumerate(tqdm(l_images_src_test)):
            outf = os.path.join(output_dir_ref, f'{index}.png')
            if not os.path.exists(outf):
                _img = T_val(Image.open(_path).convert('RGB'))
                _img.save(outf)
        ref_features = extract_fid_features(output_dir_ref, feat_model, device)
        (b2a_ref_mu, b2a_ref_sigma) = (np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False))
    lr_scheduler_gen = get_scheduler(args.lr_scheduler, optimizer=optimizer_gen, num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes, num_training_steps=args.max_train_steps * accelerator.num_processes, num_cycles=args.lr_num_cycles, power=args.lr_power)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc, num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes, num_training_steps=args.max_train_steps * accelerator.num_processes, num_cycles=args.lr_num_cycles, power=args.lr_power)
    net_lpips = lpips.LPIPS(net='vgg')
    net_lpips.to(device)
    net_lpips.requires_grad_(False)
    fixed_a2b_tokens = tokenizer(fixed_caption_tgt, max_length=tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt').input_ids[0]
    fixed_a2b_emb_base = text_encoder(fixed_a2b_tokens.to(device).unsqueeze(0))[0].detach()
    fixed_b2a_tokens = tokenizer(fixed_caption_src, max_length=tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt').input_ids[0]
    fixed_b2a_emb_base = text_encoder(fixed_b2a_tokens.to(device).unsqueeze(0))[0].detach()
    del text_encoder, tokenizer
    (unet, vae_enc, vae_dec, net_disc_a, net_disc_b) = accelerator.prepare(unet, vae_enc, vae_dec, net_disc_a, net_disc_b)
    (net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc) = accelerator.prepare(net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc)
    resume_rng = None
    if resume_full:
        restored = restore_training_state(resume_payload, accelerator.unwrap_model(unet), accelerator.unwrap_model(vae_enc), accelerator.unwrap_model(vae_dec), accelerator.unwrap_model(net_disc_a), accelerator.unwrap_model(net_disc_b), optimizer_gen, optimizer_disc, lr_scheduler_gen, lr_scheduler_disc, expected_step=resume_step)
        resume_rng = restored['rng_state']
        del resume_payload, restored
        gc.collect()
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))
    first_epoch = 0
    global_step = resume_step
    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc='Steps', disable=not accelerator.is_local_main_process)
    for (name, module) in net_disc_a.named_modules():
        if 'attn' in name:
            module.fused_attn = False
    for (name, module) in net_disc_b.named_modules():
        if 'attn' in name:
            module.fused_attn = False
    training_started_at = time.time()
    if resume_rng is not None:
        restore_rng_state(resume_rng)
    print(f'Training from step {resume_step} to {args.max_train_steps}; resume_mode={"training_state" if resume_full else "weights_only" if args.resume_checkpoint else "new"}', flush=True)
    for epoch in range(first_epoch, num_epochs):
        for (step, batch) in enumerate(train_dataloader):
            write_status(args, 'running', 'training', global_step, started_at)
            l_acc = [unet, net_disc_a, net_disc_b, vae_enc, vae_dec]
            with accelerator.accumulate(*l_acc):
                img_a = batch['pixel_values_src'].to(dtype=weight_dtype)
                img_b = batch['pixel_values_tgt'].to(dtype=weight_dtype)
                bsz = img_a.shape[0]
                fixed_a2b_emb = fixed_a2b_emb_base.repeat(bsz, 1, 1).to(dtype=weight_dtype)
                fixed_b2a_emb = fixed_b2a_emb_base.repeat(bsz, 1, 1).to(dtype=weight_dtype)
                timesteps = torch.tensor([noise_scheduler_1step.config.num_train_timesteps - 1] * bsz, device=img_a.device).long()
                cyc_fake_b = CycleGAN_Turbo.forward_with_networks(img_a, 'a2b', vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_a2b_emb)
                cyc_rec_a = CycleGAN_Turbo.forward_with_networks(cyc_fake_b, 'b2a', vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_b2a_emb)
                loss_cycle_a = crit_cycle(cyc_rec_a, img_a) * args.lambda_cycle
                loss_cycle_a += net_lpips(cyc_rec_a, img_a).mean() * args.lambda_cycle_lpips
                cyc_fake_a = CycleGAN_Turbo.forward_with_networks(img_b, 'b2a', vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_b2a_emb)
                cyc_rec_b = CycleGAN_Turbo.forward_with_networks(cyc_fake_a, 'a2b', vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_a2b_emb)
                loss_cycle_b = crit_cycle(cyc_rec_b, img_b) * args.lambda_cycle
                loss_cycle_b += net_lpips(cyc_rec_b, img_b).mean() * args.lambda_cycle_lpips
                accelerator.backward(loss_cycle_a + loss_cycle_b, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()
                fake_a = CycleGAN_Turbo.forward_with_networks(img_b, 'b2a', vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_b2a_emb)
                fake_b = CycleGAN_Turbo.forward_with_networks(img_a, 'a2b', vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_a2b_emb)
                if use_wgan:
                    loss_gan_a = generator_wgan_loss(net_disc_a, fake_b) * args.lambda_gan
                    loss_gan_b = generator_wgan_loss(net_disc_b, fake_a) * args.lambda_gan
                else:
                    loss_gan_a = net_disc_a(fake_b, for_G=True).mean() * args.lambda_gan
                    loss_gan_b = net_disc_b(fake_a, for_G=True).mean() * args.lambda_gan
                loss_transport = fake_b.new_zeros(())
                loss_frequency = fake_b.new_zeros(())
                if use_wgan and (args.lambda_transport or args.lambda_frequency):
                    costs = transport_terms(img_a, fake_b, args.fft_norm)
                    loss_transport = costs['cost'] * args.lambda_transport
                    loss_frequency = costs['frequency'] * args.lambda_frequency
                accelerator.backward(loss_gan_a + loss_gan_b + loss_transport + loss_frequency, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()
                optimizer_disc.zero_grad()
                idt_a = CycleGAN_Turbo.forward_with_networks(img_b, 'a2b', vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_a2b_emb)
                loss_idt_a = crit_idt(idt_a, img_b) * args.lambda_idt
                loss_idt_a += net_lpips(idt_a, img_b).mean() * args.lambda_idt_lpips
                idt_b = CycleGAN_Turbo.forward_with_networks(img_a, 'b2a', vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_b2a_emb)
                loss_idt_b = crit_idt(idt_b, img_a) * args.lambda_idt
                loss_idt_b += net_lpips(idt_b, img_a).mean() * args.lambda_idt_lpips
                loss_identity_structure = idt_a.new_zeros(())
                if use_wgan and args.lambda_identity_ssim:
                    loss_identity_structure = args.lambda_identity_ssim * (identity_structure(img_b, idt_a) + identity_structure(img_a, idt_b))
                loss_g_idt = loss_idt_a + loss_idt_b + loss_identity_structure
                accelerator.backward(loss_g_idt, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()
                if use_wgan:
                    terms_a = discriminator_wgan_terms(net_disc_a, img_b, fake_b)
                    terms_b = discriminator_wgan_terms(net_disc_b, img_a, fake_a)
                    loss_D_A_fake = terms_a['fake'] * args.lambda_gan
                    loss_D_B_fake = terms_b['fake'] * args.lambda_gan
                    loss_D_A_real = -terms_a['real'] * args.lambda_gan
                    loss_D_B_real = -terms_b['real'] * args.lambda_gan
                    loss_gp = 0.5 * args.lambda_gan * args.lambda_gp * (terms_a['gp'] + terms_b['gp'])
                    loss_D = 0.5 * (loss_D_A_fake + loss_D_B_fake + loss_D_A_real + loss_D_B_real) + loss_gp
                    accelerator.backward(loss_D, retain_graph=False)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_disc, args.max_grad_norm)
                    optimizer_disc.step()
                    lr_scheduler_disc.step()
                    optimizer_disc.zero_grad()
                else:
                    loss_D_A_fake = net_disc_a(fake_b.detach(), for_real=False).mean() * args.lambda_gan
                    loss_D_B_fake = net_disc_b(fake_a.detach(), for_real=False).mean() * args.lambda_gan
                    loss_D_fake = (loss_D_A_fake + loss_D_B_fake) * 0.5
                    accelerator.backward(loss_D_fake, retain_graph=False)
                    if accelerator.sync_gradients:
                        params_to_clip = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer_disc.step()
                    lr_scheduler_disc.step()
                    optimizer_disc.zero_grad()
                    loss_D_A_real = net_disc_a(img_b, for_real=True).mean() * args.lambda_gan
                    loss_D_B_real = net_disc_b(img_a, for_real=True).mean() * args.lambda_gan
                    loss_D_real = (loss_D_A_real + loss_D_B_real) * 0.5
                    accelerator.backward(loss_D_real, retain_graph=False)
                    if accelerator.sync_gradients:
                        params_to_clip = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer_disc.step()
                    lr_scheduler_disc.step()
                    optimizer_disc.zero_grad()
            logs = {}
            logs['cycle_a'] = loss_cycle_a.detach().item()
            logs['cycle_b'] = loss_cycle_b.detach().item()
            logs['gan_a'] = loss_gan_a.detach().item()
            logs['gan_b'] = loss_gan_b.detach().item()
            logs['disc_a'] = loss_D_A_fake.detach().item() + loss_D_A_real.detach().item()
            logs['disc_b'] = loss_D_B_fake.detach().item() + loss_D_B_real.detach().item()
            logs['idt_a'] = loss_idt_a.detach().item()
            logs['idt_b'] = loss_idt_b.detach().item()
            if use_wgan:
                logs.update(transport=loss_transport.detach().item(), frequency=loss_frequency.detach().item(), identity_ssim=loss_identity_structure.detach().item(), gp=loss_gp.detach().item())
            if not all(math.isfinite(value) for value in logs.values()):
                raise FloatingPointError(f'Nonfinite training loss at data iteration {global_step + 1}')
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    eval_unet = accelerator.unwrap_model(unet)
                    eval_vae_enc = accelerator.unwrap_model(vae_enc)
                    eval_vae_dec = accelerator.unwrap_model(vae_dec)
                    if global_step % args.viz_freq == 1:
                        for tracker in accelerator.trackers:
                            if tracker.name == 'wandb':
                                viz_img_a = batch['pixel_values_src'].to(dtype=weight_dtype)
                                viz_img_b = batch['pixel_values_tgt'].to(dtype=weight_dtype)
                                log_dict = {'train/real_a': [wandb.Image(viz_img_a[idx].float().detach().cpu(), caption=f'idx={idx}') for idx in range(bsz)], 'train/real_b': [wandb.Image(viz_img_b[idx].float().detach().cpu(), caption=f'idx={idx}') for idx in range(bsz)]}
                                log_dict['train/rec_a'] = [wandb.Image(cyc_rec_a[idx].float().detach().cpu(), caption=f'idx={idx}') for idx in range(bsz)]
                                log_dict['train/rec_b'] = [wandb.Image(cyc_rec_b[idx].float().detach().cpu(), caption=f'idx={idx}') for idx in range(bsz)]
                                log_dict['train/fake_b'] = [wandb.Image(fake_b[idx].float().detach().cpu(), caption=f'idx={idx}') for idx in range(bsz)]
                                log_dict['train/fake_a'] = [wandb.Image(fake_a[idx].float().detach().cpu(), caption=f'idx={idx}') for idx in range(bsz)]
                                tracker.log(log_dict)
                                gc.collect()
                                if device.type == 'cuda':
                                    torch.cuda.empty_cache()
                    if global_step == resume_step + 1 or global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps:
                        write_status(args, 'running', 'checkpoint', global_step, started_at)
                        outf = os.path.join(args.output_dir, 'checkpoints', f'model_{global_step}.pkl')
                        sd = export_generator_state(eval_unet, eval_vae_enc, eval_vae_dec, global_step)
                        atomic_torch_save(sd, outf)
                        del sd
                        gc.collect()
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                    if global_step == 1 or global_step % args.validation_steps == 0 or global_step == args.max_train_steps:
                        _timesteps = torch.tensor([noise_scheduler_1step.config.num_train_timesteps - 1] * 1, device=device).long()
                        write_status(args, 'running', 'validation', global_step, started_at)
                        net_dino = DinoStructureLoss(device=device)
                        fid_output_dir = os.path.join(args.output_dir, f'fid-{global_step}/samples_a2b')
                        os.makedirs(fid_output_dir, exist_ok=True)
                        l_dino_scores_a2b = []
                        paired_metrics = PairedMetrics(args.dataset_folder, l_images_src_test, l_images_tgt_test, T_val, os.path.join(args.output_dir, 'paired', f'step_{global_step:08d}'), args.paired_eval_manifest) if args.paired_eval else None
                        for (idx, input_img_path) in enumerate(tqdm(l_images_src_test)):
                            if idx >= args.validation_num_images and args.validation_num_images > 0:
                                break
                            outf = os.path.join(fid_output_dir, f'{idx}.png')
                            with torch.no_grad():
                                input_img = T_val(Image.open(input_img_path).convert('RGB'))
                                img_a = transforms.ToTensor()(input_img)
                                img_a = transforms.Normalize([0.5], [0.5])(img_a).unsqueeze(0).to(device)
                                eval_fake_b = CycleGAN_Turbo.forward_with_networks(img_a, 'a2b', eval_vae_enc, eval_unet, eval_vae_dec, noise_scheduler_1step, _timesteps, fixed_a2b_emb[0:1])
                                eval_fake_b_pil = transforms.ToPILImage()(eval_fake_b[0] * 0.5 + 0.5)
                                eval_fake_b_pil.save(outf)
                                if paired_metrics is not None:
                                    paired_metrics.add(input_img_path, img_a, eval_fake_b)
                                a = net_dino.preprocess(input_img).unsqueeze(0).to(device)
                                b = net_dino.preprocess(eval_fake_b_pil).unsqueeze(0).to(device)
                                dino_ssim = net_dino.calculate_global_ssim_loss(a, b).item()
                                l_dino_scores_a2b.append(dino_ssim)
                                clear_validation_features(eval_vae_enc)
                                write_status(args, 'running', 'validation_a2b', global_step, started_at, completed_images=idx + 1, total_images=len(l_images_src_test))
                        dino_score_a2b = np.mean(l_dino_scores_a2b)
                        gen_features = extract_fid_features(fid_output_dir, feat_model, device)
                        (ed_mu, ed_sigma) = (np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False))
                        score_fid_a2b = frechet_distance(a2b_ref_mu, a2b_ref_sigma, ed_mu, ed_sigma)
                        print(f'step={global_step}, fid(a2b)={score_fid_a2b:.2f}, dino(a2b)={dino_score_a2b:.3f}')
                        fid_output_dir = os.path.join(args.output_dir, f'fid-{global_step}/samples_b2a')
                        os.makedirs(fid_output_dir, exist_ok=True)
                        l_dino_scores_b2a = []
                        for (idx, input_img_path) in enumerate(tqdm(l_images_tgt_test)):
                            if idx >= args.validation_num_images and args.validation_num_images > 0:
                                break
                            outf = os.path.join(fid_output_dir, f'{idx}.png')
                            with torch.no_grad():
                                input_img = T_val(Image.open(input_img_path).convert('RGB'))
                                img_b = transforms.ToTensor()(input_img)
                                img_b = transforms.Normalize([0.5], [0.5])(img_b).unsqueeze(0).to(device)
                                eval_fake_a = CycleGAN_Turbo.forward_with_networks(img_b, 'b2a', eval_vae_enc, eval_unet, eval_vae_dec, noise_scheduler_1step, _timesteps, fixed_b2a_emb[0:1])
                                eval_fake_a_pil = transforms.ToPILImage()(eval_fake_a[0] * 0.5 + 0.5)
                                eval_fake_a_pil.save(outf)
                                a = net_dino.preprocess(input_img).unsqueeze(0).to(device)
                                b = net_dino.preprocess(eval_fake_a_pil).unsqueeze(0).to(device)
                                dino_ssim = net_dino.calculate_global_ssim_loss(a, b).item()
                                l_dino_scores_b2a.append(dino_ssim)
                                clear_validation_features(eval_vae_enc)
                                write_status(args, 'running', 'validation_b2a', global_step, started_at, completed_images=idx + 1, total_images=len(l_images_tgt_test))
                        dino_score_b2a = np.mean(l_dino_scores_b2a)
                        gen_features = extract_fid_features(fid_output_dir, feat_model, device)
                        (ed_mu, ed_sigma) = (np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False))
                        score_fid_b2a = frechet_distance(b2a_ref_mu, b2a_ref_sigma, ed_mu, ed_sigma)
                        print(f'step={global_step}, fid(b2a)={score_fid_b2a}, dino(b2a)={dino_score_b2a:.3f}')
                        (logs['val/fid_a2b'], logs['val/fid_b2a']) = (score_fid_a2b, score_fid_b2a)
                        (logs['val/dino_struct_a2b'], logs['val/dino_struct_b2a']) = (dino_score_a2b, dino_score_b2a)
                        if paired_metrics is not None:
                            paired_report = paired_metrics.finalize(global_step)
                            print(f"step={global_step}, paired={paired_report['mean']}")
                        with open(os.path.join(args.output_dir, 'validation.jsonl'), 'a', encoding='utf-8') as stream:
                            json.dump({'step': global_step, **{key: float(value) for (key, value) in logs.items() if key.startswith('val/')}}, stream)
                            stream.write('\n')
                        del net_dino
                        gc.collect()
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
            if global_step == resume_step + 1 or global_step % args.recovery_steps == 0 or global_step == args.max_train_steps:
                write_status(args, 'running', 'recovery_checkpoint', global_step, started_at)
                generator_sd = export_generator_state(accelerator.unwrap_model(unet), accelerator.unwrap_model(vae_enc), accelerator.unwrap_model(vae_dec), global_step)
                recovery = pack_training_state(generator_sd, accelerator.unwrap_model(net_disc_a), accelerator.unwrap_model(net_disc_b), optimizer_gen, optimizer_disc, lr_scheduler_gen, lr_scheduler_disc, global_step, extra={'config': vars(args), 'data_resume': 'new_shuffled_epoch', 'source_checkpoint': args.resume_checkpoint, 'session_start_step': resume_step})
                atomic_torch_save(recovery, os.path.join(args.output_dir, 'checkpoints', 'training_latest.pt'))
                del recovery, generator_sd
                gc.collect()
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if accelerator.is_main_process:
                with open(os.path.join(args.output_dir, 'metrics.jsonl'), 'a', encoding='utf-8') as stream:
                    json.dump({'step': global_step, 'session_step': global_step - resume_step, 'g_updates': 3 * global_step, 'd_updates': (1 if use_wgan else 2) * global_step, 'elapsed_seconds': time.time() - started_at, **logs}, stream)
                    stream.write('\n')
            elapsed_training = time.time() - training_started_at
            write_status(args, 'running', 'training', global_step, started_at, session_start_step=resume_step, session_step=global_step - resume_step, g_updates=3 * global_step, d_updates=(1 if use_wgan else 2) * global_step, training_elapsed_seconds=elapsed_training, eta_seconds=elapsed_training / (global_step - resume_step) * (args.max_train_steps - global_step), losses={key: float(value) for (key, value) in logs.items()})
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break
    accelerator.end_training()
    write_status(args, 'completed', 'completed', global_step, started_at, g_updates=3 * global_step, d_updates=(1 if use_wgan else 2) * global_step)
if __name__ == '__main__':
    args = parse_args_unpaired_training()
    with RunLock(os.path.join(args.output_dir, '.train.lock')):
        try:
            main(args)
        except BaseException as error:
            if _last_status is not None:
                previous = _last_status
                write_status(args, 'failed', previous['phase'], previous['step'], time.time() - previous['elapsed_seconds'], error_type=type(error).__name__, error=str(error))
            raise
