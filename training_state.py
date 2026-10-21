import copy
import random
from pathlib import Path

import numpy as np
import torch
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

from cyclegan_turbo import CycleGAN_Turbo


_ADAPTERS = (
    ("default_encoder", "sd_encoder", "l_target_modules_encoder"),
    ("default_decoder", "sd_decoder", "l_target_modules_decoder"),
    ("default_others", "sd_other", "l_modules_others"),
)
_FORMAT = "sftot_training_state"
_VERSION = 1


def _step(value):
    if type(value) is not int or value < 0:
        raise ValueError("Checkpoint step must be a nonnegative integer")
    return value


def _cpu_copy(value, memo=None):
    if memo is None:
        memo = {}
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise ValueError("Checkpoint tensors must have strided layout")
        key = (value.device, value.dtype, value.untyped_storage().data_ptr(),
               value.storage_offset(), tuple(value.shape), tuple(value.stride()))
        if key not in memo:
            memo[key] = value.detach().to(device="cpu", copy=True)
        return memo[key]
    if isinstance(value, dict):
        result = type(value)((key, _cpu_copy(item, memo)) for key, item in value.items())
        if hasattr(value, "_metadata"):
            result._metadata = _cpu_copy(value._metadata, memo)
        return result
    if isinstance(value, list):
        return [_cpu_copy(item, memo) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item, memo) for item in value)
    return copy.deepcopy(value)


def _check_state(saved, current, label):
    if not isinstance(saved, dict) or saved.keys() != current.keys():
        missing = sorted(current.keys() - saved.keys()) if isinstance(saved, dict) else []
        unexpected = sorted(saved.keys() - current.keys()) if isinstance(saved, dict) else []
        raise ValueError(f"{label} state keys differ: missing={missing}, unexpected={unexpected}")
    for name, tensor in saved.items():
        expected = current[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != expected.shape
                or tensor.dtype != expected.dtype):
            raise ValueError(f"{label}.{name} tensor shape or dtype differs")


def _adapter_modules(model, adapter):
    names = {
        name for name, module in model.named_modules()
        if adapter in getattr(module, "lora_A", {})
        and adapter in getattr(module, "lora_B", {})
    }
    if not names:
        raise ValueError(f"No initialized LoRA modules for {adapter}")
    return names


def _resolve_targets(model, targets):
    if not isinstance(targets, (list, tuple, set)) or not targets:
        raise ValueError("Generator adapters require explicit target module names")
    internal = {"base_layer", "lora_A", "lora_B", "lora_dropout", "lora_embedding_A",
                "lora_embedding_B", "lora_magnitude_vector", "modules_to_save"}
    names = {name for name, _ in model.named_modules()
             if name and not internal.intersection(name.split("."))}
    resolved = set()
    for target in targets:
        matches = {name for name in names if name == target or name.endswith("." + target)}
        if not matches:
            raise ValueError(f"Checkpoint target does not match any base module: {target}")
        resolved.update(matches)
    return resolved


def _metadata(unet, vae_enc, vae_dec):
    if (vae_enc.vae is not vae_dec.vae or vae_enc.vae_b2a is not vae_dec.vae_b2a
            or vae_enc.vae_b2a is None or vae_enc.vae is vae_enc.vae_b2a):
        raise ValueError("Generator wrappers must share the same two directional VAEs")
    result = {}
    ranks = set()
    for adapter, _, module_key in _ADAPTERS:
        config = unet.peft_config[adapter]
        actual = _adapter_modules(unet, adapter)
        if _resolve_targets(unet, config.target_modules) != actual:
            raise ValueError(f"Configured targets differ from initialized adapter {adapter}")
        result[module_key] = sorted(actual)
        ranks.add(config.r)
    if len(ranks) != 1:
        raise ValueError("The three UNet adapters must have the same rank")
    result["rank_unet"] = ranks.pop()
    vae_configs = [vae.peft_config["vae_skip"] for vae in (vae_enc.vae, vae_enc.vae_b2a)]
    vae_targets = [_adapter_modules(vae, "vae_skip") for vae in (vae_enc.vae, vae_enc.vae_b2a)]
    if vae_configs[0].r != vae_configs[1].r or vae_targets[0] != vae_targets[1]:
        raise ValueError("The two VAE adapters must have matching ranks and targets")
    for vae, config, actual in zip((vae_enc.vae, vae_enc.vae_b2a), vae_configs, vae_targets):
        if _resolve_targets(vae, config.target_modules) != actual:
            raise ValueError("Configured VAE targets differ from its initialized adapter")
    result["rank_vae"] = vae_configs[0].r
    result["vae_lora_target_modules"] = sorted(vae_targets[0])
    return result


def _validate_generator(sd, unet, vae_enc, vae_dec, expected_step=None):
    CycleGAN_Turbo.validate_checkpoint(sd)
    step = _step(sd.get("step"))
    if expected_step is not None and step != _step(expected_step):
        raise ValueError(f"Checkpoint step {step} differs from expected step {expected_step}")
    for key, expected in _metadata(unet, vae_enc, vae_dec).items():
        if isinstance(expected, list):
            model = vae_enc.vae if key == "vae_lora_target_modules" else unet
            actual = sorted(_resolve_targets(model, sd[key]))
        else:
            actual = sd[key]
        if actual != expected:
            raise ValueError(f"Checkpoint {key} differs from the initialized generator")
    for adapter, state_key, _ in _ADAPTERS:
        _check_state(sd[state_key], get_peft_model_state_dict(unet, adapter_name=adapter), state_key)
    _check_state(sd["sd_unet_conv_in"], unet.conv_in.state_dict(), "sd_unet_conv_in")
    _check_state(sd["sd_vae_enc"], vae_enc.state_dict(), "sd_vae_enc")
    _check_state(sd["sd_vae_dec"], vae_dec.state_dict(), "sd_vae_dec")
    for key, tensor in sd["sd_vae_enc"].items():
        if not torch.equal(tensor, sd["sd_vae_dec"][key]):
            raise ValueError(f"Conflicting shared VAE tensors: {key}")
    for adapter, state_key, _ in _ADAPTERS:
        for key, tensor in sd["sd_unet_conv_in"].items():
            if f".{adapter}." in key:
                adapter_key = "conv_in." + key.replace(f".{adapter}.", ".")
                if adapter_key not in sd[state_key] or not torch.equal(tensor, sd[state_key][adapter_key]):
                    raise ValueError(f"Conflicting conv_in adapter tensor: {key}")
    return step


def load_generator_state(sd, unet, vae_enc, vae_dec, expected_step=None):
    step = _validate_generator(sd, unet, vae_enc, vae_dec, expected_step)
    for adapter, state_key, _ in _ADAPTERS:
        result = set_peft_model_state_dict(unet, sd[state_key], adapter_name=adapter)
        missing = [key for key in result.missing_keys if f".{adapter}." in key]
        if result.unexpected_keys or missing:
            raise ValueError(f"Unable to restore adapter {adapter}: {result}")
    unet.conv_in.load_state_dict(sd["sd_unet_conv_in"], strict=True)
    vae_enc.load_state_dict(sd["sd_vae_enc"], strict=True)
    vae_dec.load_state_dict(sd["sd_vae_dec"], strict=True)
    return step


def export_generator_state(unet, vae_enc, vae_dec, step):
    sd = _metadata(unet, vae_enc, vae_dec)
    sd["step"] = _step(step)
    for adapter, state_key, _ in _ADAPTERS:
        sd[state_key] = get_peft_model_state_dict(unet, adapter_name=adapter)
    sd["sd_unet_conv_in"] = unet.conv_in.state_dict()
    sd["sd_vae_enc"] = vae_enc.state_dict()
    sd["sd_vae_dec"] = vae_dec.state_dict()
    return _cpu_copy(sd)


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch": torch.get_rng_state().clone(),
        "cuda": [value.cpu().clone() for value in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_initialized() else [],
    }


def restore_rng_state(state):
    if not isinstance(state, dict) or set(state) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("Incomplete RNG state")
    cuda = state["cuda"]
    if not isinstance(cuda, list):
        raise ValueError("CUDA RNG state must be a list")
    if cuda and (not torch.cuda.is_initialized() or len(cuda) != torch.cuda.device_count()):
        raise ValueError("Initialize the original CUDA device topology before restoring CUDA RNG")
    if not isinstance(state["torch"], torch.Tensor) or state["torch"].dtype != torch.uint8:
        raise ValueError("Torch RNG state must be a byte tensor")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if cuda:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda])


def pack_training_state(generator_sd, disc_a, disc_b, optimizer_gen, optimizer_disc,
                        scheduler_gen, scheduler_disc, step, extra=None):
    step = _step(step)
    CycleGAN_Turbo.validate_checkpoint(generator_sd)
    if _step(generator_sd.get("step")) != step:
        raise ValueError("Generator and training checkpoint steps must match")
    if extra is not None and not isinstance(extra, dict):
        raise ValueError("extra must be a dictionary or None")
    state = {
        "format": _FORMAT,
        "version": _VERSION,
        "step": step,
        "generator": generator_sd,
        "disc_a": disc_a.state_dict(),
        "disc_b": disc_b.state_dict(),
        "optimizer_gen": optimizer_gen.state_dict(),
        "optimizer_disc": optimizer_disc.state_dict(),
        "scheduler_gen": scheduler_gen.state_dict(),
        "scheduler_disc": scheduler_disc.state_dict(),
        "rng_state": capture_rng_state(),
        "extra": {} if extra is None else extra,
    }
    return _cpu_copy(state)


def _check_optimizer(state, optimizer, label):
    if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
        raise ValueError(f"Incomplete {label} state")
    groups = state["param_groups"]
    if len(groups) != len(optimizer.param_groups):
        raise ValueError(f"{label} parameter group count differs")
    for saved, current in zip(groups, optimizer.param_groups):
        if len(saved["params"]) != len(current["params"]):
            raise ValueError(f"{label} parameter group size differs")
        for index, parameter in zip(saved["params"], current["params"]):
            for key, value in state["state"].get(index, {}).items():
                if isinstance(value, torch.Tensor) and key != "step" and value.shape != parameter.shape:
                    raise ValueError(f"{label} moment shape differs for parameter {index}: {key}")


def restore_training_state(state, unet, vae_enc, vae_dec, disc_a, disc_b,
                           optimizer_gen, optimizer_disc, scheduler_gen, scheduler_disc,
                           expected_step=None):
    required = {"format", "version", "step", "generator", "disc_a", "disc_b",
                "optimizer_gen", "optimizer_disc", "scheduler_gen", "scheduler_disc",
                "rng_state", "extra"}
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError("Incomplete full training state; inference checkpoints cannot fully resume training")
    if state["format"] != _FORMAT or state["version"] != _VERSION:
        raise ValueError("Unsupported training checkpoint format/version")
    step = _step(state["step"])
    if expected_step is not None and step != _step(expected_step):
        raise ValueError(f"Checkpoint step {step} differs from expected step {expected_step}")
    _validate_generator(state["generator"], unet, vae_enc, vae_dec, step)
    _check_state(state["disc_a"], disc_a.state_dict(), "disc_a")
    _check_state(state["disc_b"], disc_b.state_dict(), "disc_b")
    _check_optimizer(state["optimizer_gen"], optimizer_gen, "optimizer_gen")
    _check_optimizer(state["optimizer_disc"], optimizer_disc, "optimizer_disc")
    for key, scheduler in (("scheduler_gen", scheduler_gen), ("scheduler_disc", scheduler_disc)):
        if not isinstance(state[key], dict) or state[key].keys() != scheduler.state_dict().keys():
            raise ValueError(f"{key} state keys differ")
    load_generator_state(state["generator"], unet, vae_enc, vae_dec, step)
    disc_a.load_state_dict(state["disc_a"], strict=True)
    disc_b.load_state_dict(state["disc_b"], strict=True)
    optimizer_gen.load_state_dict(state["optimizer_gen"])
    optimizer_disc.load_state_dict(state["optimizer_disc"])
    scheduler_gen.load_state_dict(state["scheduler_gen"])
    scheduler_disc.load_state_dict(state["scheduler_disc"])
    optimizer_gen.zero_grad(set_to_none=True)
    optimizer_disc.zero_grad(set_to_none=True)
    return {"step": step, "extra": copy.deepcopy(state["extra"]),
            "rng_state": _cpu_copy(state["rng_state"])}


def load_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Local checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must contain a dictionary")
    return state
