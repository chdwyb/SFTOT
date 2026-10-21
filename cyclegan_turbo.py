import copy
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd


class VAE_encode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_encode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        return _vae.encode(x).latent_dist.sample() * _vae.config.scaling_factor


class VAE_decode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_decode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        assert _vae.encoder.current_down_blocks is not None
        _vae.decoder.incoming_skip_acts = _vae.encoder.current_down_blocks
        x_decoded = (_vae.decode(x / _vae.config.scaling_factor).sample).clamp(-1, 1)
        return x_decoded


def initialize_unet(rank, return_lora_module_names=False, device="cpu", pretrained_model_name_or_path="stabilityai/sd-turbo", local_files_only=False):
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", local_files_only=local_files_only).to(device)
    unet.requires_grad_(False)
    unet.train()
    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n: continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and "up_blocks" in n:
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break
    lora_conf_encoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder, lora_alpha=rank)
    lora_conf_decoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder, lora_alpha=rank)
    lora_conf_others = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_modules_others, lora_alpha=rank)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")
    unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
    if return_lora_module_names:
        return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others
    else:
        return unet


def initialize_vae(rank=4, return_lora_module_names=False, device="cpu", pretrained_model_name_or_path="stabilityai/sd-turbo", local_files_only=False):
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", local_files_only=local_files_only).to(device)
    vae.requires_grad_(False)
    vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
    vae.requires_grad_(True)
    vae.train()

    vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device).requires_grad_(True)
    vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device).requires_grad_(True)
    vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device).requires_grad_(True)
    vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device).requires_grad_(True)
    torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
    vae.decoder.ignore_skip = False
    vae.decoder.gamma = 1
    l_vae_target_modules = ["conv1","conv2","conv_in", "conv_shortcut",
        "conv", "conv_out", "skip_conv_1", "skip_conv_2", "skip_conv_3",
        "skip_conv_4", "to_k", "to_q", "to_v", "to_out.0",
    ]
    vae_lora_config = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_vae_target_modules)
    vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
    if return_lora_module_names:
        return vae, l_vae_target_modules
    else:
        return vae


class CycleGAN_Turbo(torch.nn.Module):
    def __init__(self, checkpoint_path, device="cpu",
                 pretrained_model_name_or_path="stabilityai/sd-turbo", local_files_only=False):
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Local checkpoint not found: {checkpoint_path}")
        sd = torch.load(checkpoint_path, map_location="cpu")
        self.validate_checkpoint(sd)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", local_files_only=local_files_only)
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", local_files_only=local_files_only).to(device)
        self.sched = make_1step_sched(device=device, pretrained_model_name_or_path=pretrained_model_name_or_path, local_files_only=local_files_only)
        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", local_files_only=local_files_only)
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", local_files_only=local_files_only)
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)

        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
        vae.decoder.ignore_skip = False
        self.unet, self.vae = unet, vae
        self.load_ckpt_from_state_dict(sd)
        self.timesteps = torch.tensor([self.sched.config.num_train_timesteps - 1], device=device).long()
        self.vae_enc.to(device)
        self.vae_dec.to(device)
        self.unet.to(device)

    @staticmethod
    def validate_checkpoint(sd):
        if not isinstance(sd, dict):
            raise ValueError("Checkpoint must contain a state dictionary")
        module_keys = ("l_target_modules_encoder", "l_target_modules_decoder",
                       "l_modules_others", "vae_lora_target_modules")
        state_keys = ("sd_encoder", "sd_decoder", "sd_other", "sd_unet_conv_in",
                      "sd_vae_enc", "sd_vae_dec")
        required = set(module_keys + state_keys + ("rank_unet", "rank_vae"))
        missing = required - sd.keys()
        if missing:
            raise ValueError(f"Incomplete project checkpoint; missing keys: {sorted(missing)}")
        for key in ("rank_unet", "rank_vae"):
            if type(sd[key]) is not int or sd[key] <= 0:
                raise ValueError(f"Checkpoint {key} must be a positive integer")
        for key in module_keys:
            value = sd[key]
            if (not isinstance(value, (list, tuple)) or not value
                    or any(not isinstance(name, str) or not name for name in value)
                    or len(value) != len(set(value))):
                raise ValueError(f"Checkpoint {key} must contain unique nonempty module names")
        for key in state_keys:
            value = sd[key]
            if (not isinstance(value, dict) or not value
                    or any(not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
                           for name, tensor in value.items())):
                raise ValueError(f"Checkpoint {key} must contain named tensors")

    def load_ckpt_from_state_dict(self, sd):
        self.validate_checkpoint(sd)
        lora_conf_encoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_target_modules_encoder"], lora_alpha=sd["rank_unet"])
        lora_conf_decoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_target_modules_decoder"], lora_alpha=sd["rank_unet"])
        lora_conf_others = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_modules_others"], lora_alpha=sd["rank_unet"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for adapter, key in (("default_encoder", "sd_encoder"),
                             ("default_decoder", "sd_decoder"),
                             ("default_others", "sd_other")):
            parameters = {
                name.replace(f".{adapter}.weight", ".weight"): parameter
                for name, parameter in self.unet.named_parameters()
                if "lora" in name and adapter in name
            }
            if parameters.keys() != sd[key].keys():
                raise ValueError(f"Checkpoint {key} does not match its adapter modules")
            with torch.no_grad():
                for name, parameter in parameters.items():
                    if parameter.shape != sd[key][name].shape:
                        raise ValueError(f"Checkpoint tensor shape mismatch: {key}.{name}")
                    parameter.copy_(sd[key][name])
        self.unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
        self.unet.conv_in.load_state_dict(sd["sd_unet_conv_in"])

        vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
        self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        self.vae.decoder.gamma = 1
        self.vae_b2a = copy.deepcopy(self.vae)
        self.vae_enc = VAE_encode(self.vae, vae_b2a=self.vae_b2a)
        self.vae_enc.load_state_dict(sd["sd_vae_enc"])
        self.vae_dec = VAE_decode(self.vae, vae_b2a=self.vae_b2a)
        self.vae_dec.load_state_dict(sd["sd_vae_dec"])

    @staticmethod
    def forward_with_networks(x, direction, vae_enc, unet, vae_dec, sched, timesteps, text_emb):
        B = x.shape[0]
        if direction not in ("a2b", "b2a"):
            raise ValueError("direction must be a2b or b2a")
        timesteps = torch.as_tensor(timesteps, device=x.device).reshape(-1)
        if timesteps.numel() == 1:
            timesteps = timesteps.expand(B)
        if timesteps.numel() != B:
            raise ValueError("timesteps must contain one value or one value per input image")
        if text_emb.shape[0] == 1:
            text_emb = text_emb.expand(B, -1, -1)
        if text_emb.shape[0] != B:
            raise ValueError("text embeddings must contain one prompt or one prompt per input image")
        x_enc = vae_enc(x, direction=direction).to(x.dtype)
        model_pred = unet(x_enc, timesteps, encoder_hidden_states=text_emb,).sample
        x_out = torch.stack([sched.step(model_pred[i], timesteps[i], x_enc[i], return_dict=True).prev_sample for i in range(B)])
        x_out_decoded = vae_dec(x_out, direction=direction)
        return x_out_decoded

    @staticmethod
    def get_traininable_params(unet, vae_a2b, vae_b2a):

        params_gen = list(unet.conv_in.parameters())
        unet.conv_in.requires_grad_(True)
        unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
        for n,p in unet.named_parameters():
            if "lora" in n and "default" in n:
                assert p.requires_grad
                params_gen.append(p)


        for n,p in vae_a2b.named_parameters():
            if "lora" in n and "vae_skip" in n:
                assert p.requires_grad
                params_gen.append(p)
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_1.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_2.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_3.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_4.parameters())


        for n,p in vae_b2a.named_parameters():
            if "lora" in n and "vae_skip" in n:
                assert p.requires_grad
                params_gen.append(p)
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_1.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_2.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_3.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_4.parameters())


        return list({id(p): p for p in params_gen if p.requires_grad}.values())

    def forward(self, x_t, direction, caption=None, caption_emb=None):
        if direction not in ("a2b", "b2a"):
            raise ValueError("direction must be a2b or b2a")
        if caption is None and caption_emb is None:
            raise ValueError("caption or caption_emb is required")
        if caption_emb is not None:
            caption_enc = caption_emb
        else:
            caption_tokens = self.tokenizer(caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt").input_ids.to(x_t.device)
            caption_enc = self.text_encoder(caption_tokens)[0].detach().clone()
        return self.forward_with_networks(x_t, direction, self.vae_enc, self.unet, self.vae_dec, self.sched, self.timesteps, caption_enc)
