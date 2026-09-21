"""Runtime patches ESMFold2 needs on consumer NVIDIA cards.

Call apply_patches() BEFORE importing esm.models.esmfold2 or the transformers
esmfold2 modules, in case a future revision binds torch.linalg.svd at import
time rather than calling it through the module attribute.

    import esmfold2_patches; esmfold2_patches.apply_patches()
    from esm.models.esmfold2 import ESMFold2InputBuilder
"""
import torch

_orig_svd = torch.linalg.svd
_applied = False


def _safe_svd(A, full_matrices=True, driver=None):
    """CPU fp32 fallback for the 3x3 frame SVDs in the structure module.

    Under bf16 autocast those matrices occasionally arrive with non-finite
    entries, and cuSOLVER raises rather than returning a usable factorisation.
    At 3x3 the fallback costs microseconds.
    """
    if A.is_cuda and A.shape[-1] <= 4 and A.shape[-2] <= 4:
        Acpu = A.detach().float().cpu()
        if not torch.isfinite(Acpu).all():
            Acpu = torch.nan_to_num(Acpu, nan=0.0, posinf=1e6, neginf=-1e6)
        out = _orig_svd(Acpu, full_matrices=full_matrices)
        return type(out)(tuple(t.to(A.device, A.dtype) for t in out))
    return _orig_svd(A, full_matrices=full_matrices, driver=driver)


def apply_patches():
    """Idempotent. Returns the list of patch names applied."""
    global _applied
    if not _applied:
        torch.linalg.svd = _safe_svd
        _applied = True
    return ["small_matrix_svd_cpu_fallback"]
