import torch
from torch.nn import functional as F

def multilevel_scores(predictions):
    heads = []

    def collect(value):
        if isinstance(value, torch.Tensor):
            if value.ndim < 1 or value.shape[0] == 0 or value.numel() == 0:
                raise ValueError('Each critic head must have a nonempty batch dimension')
            heads.append(value.reshape(value.shape[0], -1).mean(dim=1))
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
        else:
            raise TypeError('Critic predictions must be tensors or nested lists/tuples')
    collect(predictions)
    if not heads or any((head.shape != heads[0].shape for head in heads)):
        raise ValueError('Critic heads must be nonempty and have matching batch sizes')
    return torch.stack(heads).sum(dim=0)

def generator_wgan_loss(critic, fake):
    return -multilevel_scores(critic(fake)).mean()

def gradient_penalty(critic, real, fake):
    if real.shape != fake.shape or real.ndim != 4:
        raise ValueError('WGAN-GP requires equally shaped BCHW real/fake images')
    with torch.autocast(device_type=real.device.type, enabled=False):
        epsilon = torch.rand(real.shape[0], 1, 1, 1, device=real.device, dtype=torch.float32)
        mixed = (epsilon * real.detach().float() + (1.0 - epsilon) * fake.detach().float()).requires_grad_(True)
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            scores = multilevel_scores(critic(mixed))
            gradients = torch.autograd.grad(scores.sum(), mixed, create_graph=True)[0]
        return (gradients.flatten(1).norm(2, dim=1) - 1.0).square().mean()

def discriminator_wgan_terms(critic, real, fake):
    return {'fake': multilevel_scores(critic(fake.detach())).mean(), 'real': multilevel_scores(critic(real.detach())).mean(), 'gp': gradient_penalty(critic, real, fake)}

def unit_range(image):
    return (image.float() + 1.0) * 0.5

def ssim(x, y, window_size=11, data_range=1.0):
    if x.shape != y.shape or x.ndim != 4:
        raise ValueError('SSIM requires equally shaped BCHW tensors')
    size = min(window_size, x.shape[-2], x.shape[-1])
    size -= (size + 1) % 2
    if size < 3:
        raise ValueError('SSIM requires spatial dimensions of at least 3')
    (x, y) = (x.float(), y.float())
    coordinate = torch.arange(size, device=x.device, dtype=x.dtype) - size // 2
    kernel = torch.exp(-coordinate.square() / (2.0 * 1.5 ** 2))
    kernel = kernel / kernel.sum()
    window = (kernel[:, None] * kernel[None, :]).expand(x.shape[1], 1, size, size)

    def average(value):
        return F.conv2d(value, window, groups=x.shape[1])
    (mean_x, mean_y) = (average(x), average(y))
    variance_x = (average(x.square()) - mean_x.square()).clamp_min(0)
    variance_y = (average(y.square()) - mean_y.square()).clamp_min(0)
    covariance = average(x * y) - mean_x * mean_y
    (c1, c2) = ((0.01 * data_range) ** 2, (0.03 * data_range) ** 2)
    score = (2 * mean_x * mean_y + c1) * (2 * covariance + c2) / ((mean_x.square() + mean_y.square() + c1) * (variance_x + variance_y + c2))
    return score.mean().clamp(-1.0, 1.0)

def frequency_cost(source, restored, fft_norm='ortho'):
    if fft_norm not in {'ortho', 'backward', 'forward'}:
        raise ValueError('fft_norm must be ortho, backward, or forward')
    residual = unit_range(source) - unit_range(restored)
    return torch.fft.fft2(residual, dim=(-2, -1), norm=fft_norm).abs().mean()

def transport_terms(source, restored, fft_norm='ortho'):
    (source01, restored01) = (unit_range(source), unit_range(restored))
    l1 = F.l1_loss(source01, restored01)
    structure = 1.0 - ssim(source01, restored01)
    return {'cost': l1 + structure, 'l1': l1, 'structure': structure, 'frequency': frequency_cost(source, restored, fft_norm)}

def identity_structure(image, identity):
    return 1.0 - ssim(unit_range(image), unit_range(identity))
