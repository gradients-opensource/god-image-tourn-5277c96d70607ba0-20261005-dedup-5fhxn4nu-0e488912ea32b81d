"""torch.randint(0, 256, shape, dtype=uint8, generator=<fresh, manual_seed(seed)>) as a CUDA device with a GIVEN
number of SMs would produce it - so the evaluator's stochastic rounding can be reproduced on another GPU.

Why. comfy.float.stochastic_rounding draws its rounding noise with torch.randint on the evaluator's GPU. PyTorch's
CUDA kernel (aten/src/ATen/native/cuda/DistributionTemplates.h, torch 2.9.1) launches
    block = 256 threads,  grid = min(multiProcessorCount * (maxThreadsPerMultiProcessor / 256), ceil(n / 256))
and, in a grid-stride loop, thread t makes its k-th curand4() call (Philox4x32-10, curand_init(seed, subsequence=t,
offset=0)) and writes its four 32-bit outputs to elements  t + T*(4k + i), i = 0..3,  T = 256 * grid.  So the value
of an element depends on the SM count: the validator trains on an H100 (132 SMs) but evaluates on an A100 (108 SMs)
(G.O.D validator/tournament/gpu_requirements.py: IMAGETASK -> H100_1X; validator/evaluation: gpu_models ["A100"]),
and a pattern drawn on the training GPU is not the evaluator's pattern for any tensor above ~220 k elements.

Philox4x32-10 (curand_philox4x32_x.h): counter c = (c0, c1, c2, c3), key k = (k0, k1);
    round: (hi0, lo0) = mulhilo(0xD2511F53, c0); (hi1, lo1) = mulhilo(0xCD9E8D57, c2);
           c = (hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0);   k += (0x9E3779B9, 0xBB67AE85) between rounds.
Thread t's k-th call uses c = (k, 0, t, 0) and key (seed_lo, seed_hi); randint keeps (value % 256).
All arithmetic runs in int64 with 16-bit splits of the multiplier, so nothing overflows.
"""

import torch

M0, M1 = 0xD2511F53, 0xCD9E8D57
W0, W1 = 0x9E3779B9, 0xBB67AE85
MASK = 0xFFFFFFFF


def _mulhilo(a, m):
    """(hi, lo) 32-bit words of a * m for int64 tensors a in [0, 2^32) and a 32-bit constant m, without overflow."""
    m_lo, m_hi = m & 0xFFFF, m >> 16
    p1 = a * m_lo                      # < 2^48
    p2 = a * m_hi                      # < 2^48
    mid = p1 + ((p2 & 0xFFFF) << 16)   # < 2^49
    lo = mid & MASK
    hi = ((p2 >> 16) + (mid >> 32)) & MASK
    return hi, lo


def philox4x32_10(c0, c1, c2, c3, k0, k1):
    """Ten Philox4x32 rounds on int64 tensors holding uint32 values; returns the four output words."""
    for r in range(10):
        if r:
            k0 = (k0 + W0) & MASK
            k1 = (k1 + W1) & MASK
        hi0, lo0 = _mulhilo(c0, M0)
        hi1, lo1 = _mulhilo(c2, M1)
        c0, c1, c2, c3 = (hi1 ^ c1 ^ k0) & MASK, lo1, (hi0 ^ c3 ^ k1) & MASK, lo0
    return c0, c1, c2, c3


def randint_u8(shape, seed, sm_count, device, threads_per_sm=2048, max_pairs=1 << 24):
    """torch.randint(0, 256, shape, dtype=torch.uint8) from a fresh generator seeded with `seed`, as drawn on a CUDA
    device with `sm_count` SMs and `threads_per_sm` resident threads per SM (torch 2.9.1 kernel geometry).
    Draw d of thread t fills elements t + T*(4d + i), i = 0..3, so the flat output in element order is the
    (draw, word, thread) grid flattened; draws are processed in chunks of at most `max_pairs` (draw, thread)
    pairs to bound memory."""
    n = 1
    for s in shape:
        n *= int(s)
    out = torch.empty(n, dtype=torch.uint8, device=device)
    if n == 0:
        return out.reshape(shape)
    block = 256
    grid = min(sm_count * (threads_per_sm // block), (n + block - 1) // block)
    T = block * grid
    draws = (n - 1) // (T * 4) + 1
    k0 = torch.tensor(seed & MASK, dtype=torch.int64, device=device)
    k1 = torch.tensor((seed >> 32) & MASK, dtype=torch.int64, device=device)
    tids = torch.arange(T, dtype=torch.int64, device=device).unsqueeze(0)      # subsequence = thread index
    step = max(1, max_pairs // T)
    for d0 in range(0, draws, step):
        d1 = min(draws, d0 + step)
        c0 = torch.arange(d0, d1, dtype=torch.int64, device=device).unsqueeze(1).expand(d1 - d0, T)
        c2 = tids.expand(d1 - d0, T)
        z = torch.zeros_like(c0)
        words = philox4x32_10(c0, z, c2, z, k0, k1)
        flat = torch.stack(words, dim=1).reshape(-1)                            # (draw, word, thread) order
        start = d0 * 4 * T
        m = min(n - start, flat.numel())
        out[start:start + m] = (flat[:m] & 0xFF).to(torch.uint8)
    return out.reshape(shape)


def install_evaluator_gpu(sm_count, threads_per_sm=2048, fp8_compute=False):
    """Make this process's Comfy compute what the validator's evaluator GPU computes (C.EVAL_GPU, an A100):
      - comfy.float.stochastic_rounding draws its fp8 rounding noise with that GPU's kernel geometry
        (randint_u8) instead of this device's - the LoRA merge re-rounding (plain fp8: ModelPatcher) and the
        re-quantisation (comfy_quant: quant_ops) both call it by attribute, so both follow;
      - comfy.model_management.supports_fp8_compute reports that GPU's answer (sm80: False), so comfy_quant
        layers take the emulated path (dequantised weights, bf16 activations) as they do on the evaluator.
    The noise patch is installed only after self_check() reproduces THIS device's own torch.randint bit for bit
    (fail closed: a PyTorch build with another kernel geometry keeps Comfy's own draws, and
    comfy.float._crown_eval_sm_count stays unset so --exact-merge does not round against a pattern it cannot
    reproduce). Idempotent. Returns a one-line description for the log."""
    import comfy.float as cf
    import comfy.model_management as mm

    mm.supports_fp8_compute = lambda device=None: fp8_compute
    if getattr(cf, "_crown_eval_sm_count", None) == sm_count:
        return f"evaluator GPU emulated (already): {sm_count} SMs, fp8 compute {fp8_compute}"
    if torch.cuda.is_available():
        ok, msg = self_check(torch.device("cuda"), shapes=((1000,), (300_001,), (3072, 3072)), seeds=(0, 3287197034))
        if not ok:
            return f"evaluator GPU NOT emulated: philox self-check failed ({msg}); fp8 compute {fp8_compute}"
    orig = getattr(cf, "_crown_orig_stochastic_rounding", None) or cf.stochastic_rounding
    cf._crown_orig_stochastic_rounding = orig

    def stochastic_rounding(value, dtype, seed=0):
        if (dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and value.is_cuda
                and getattr(cf, "_CK_STOCHASTIC_ROUNDING_AVAILABLE", False)):
            rng = randint_u8(tuple(value.size()), int(seed), sm_count, value.device, threads_per_sm)
            return cf._ck_stochastic_rounding_fp8(value, rng, dtype)
        return orig(value, dtype, seed=seed)

    cf.stochastic_rounding = stochastic_rounding
    cf._crown_eval_sm_count = sm_count
    cf._crown_eval_threads_per_sm = threads_per_sm
    return f"evaluator GPU emulated: stochastic-rounding noise with {sm_count} SMs x {threads_per_sm} threads, fp8 compute {fp8_compute}"


def self_check(device, shapes=((7,), (1000,), (221_185,), (300_001,), (3072, 3072), (12289, 3071)), seeds=(0, 1, 3287197034, 0xFFFFFFFF)):
    """Reproduce this device's own torch.randint bit for bit with its own SM count. True only if every case matches."""
    props = torch.cuda.get_device_properties(device)
    for shape in shapes:
        for seed in seeds:
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            ref = torch.randint(0, 256, shape, dtype=torch.uint8, device=device, generator=g)
            ours = randint_u8(shape, seed, props.multi_processor_count, device, props.max_threads_per_multi_processor)
            if not torch.equal(ref, ours):
                return False, f"shape {shape} seed {seed}: {int((ref != ours).sum())}/{ref.numel()} differ"
    return True, f"{len(shapes) * len(seeds)} cases bitwise equal on {props.name} ({props.multi_processor_count} SMs)"
