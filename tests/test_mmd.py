import importlib.util
from pathlib import Path

import torch

try:
    from pygda.utils.mmd import MMD, get_MMD, guassian_kernel
except ModuleNotFoundError:
    _mmd_path = Path(__file__).resolve().parents[1] / "pygda" / "utils" / "mmd.py"
    _spec = importlib.util.spec_from_file_location("pygda_utils_mmd", _mmd_path)
    _mmd = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mmd)
    MMD, get_MMD, guassian_kernel = _mmd.MMD, _mmd.get_MMD, _mmd.guassian_kernel


def test_get_mmd_equal_sizes_matches_v_statistic():
    torch.manual_seed(0)
    source = torch.randn(8, 4)
    target = torch.randn(8, 4) + 1.0

    loss = get_MMD(source, target, fix_sigma=1.0)
    kernels = guassian_kernel(source, target, fix_sigma=1.0)
    expected = torch.mean(
        kernels[:8, :8] + kernels[8:, 8:] - kernels[:8, 8:] - kernels[8:, :8]
    )
    torch.testing.assert_close(loss, expected)


def test_get_mmd_unequal_sizes_matches_v_statistic_and_is_symmetric():
    torch.manual_seed(1)
    source = torch.randn(10, 4)
    target = torch.randn(6, 4) + 0.5

    loss_st = get_MMD(source, target, fix_sigma=1.0)
    loss_ts = get_MMD(target, source, fix_sigma=1.0)

    kernels = guassian_kernel(source, target, fix_sigma=1.0)
    expected = (
        kernels[:10, :10].mean()
        + kernels[10:, 10:].mean()
        - kernels[:10, 10:].mean()
        - kernels[10:, :10].mean()
    )

    torch.testing.assert_close(loss_st, expected)
    torch.testing.assert_close(loss_st, loss_ts)
    assert loss_st.item() >= 0.0


def test_get_mmd_vanishes_for_identical_empirical_distributions_of_different_sizes():
    torch.manual_seed(2)
    source = torch.randn(5, 3)
    target = source.repeat(2, 1)

    loss = get_MMD(source, target, fix_sigma=1.0)
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_get_mmd_unequal_sizes_propagates_gradients():
    torch.manual_seed(3)
    source = torch.randn(7, 4, requires_grad=True)
    target = torch.randn(4, 4, requires_grad=True)

    loss = get_MMD(source, target)
    loss.backward()

    assert source.grad is not None and torch.isfinite(source.grad).all()
    assert target.grad is not None and torch.isfinite(target.grad).all()


def test_sampled_mmd_runs_with_unequal_inputs():
    torch.manual_seed(4)
    source = torch.randn(12, 4)
    target = torch.randn(7, 4)

    loss = MMD(source, target, sampling_num=5, times=3)
    assert torch.isfinite(loss)
