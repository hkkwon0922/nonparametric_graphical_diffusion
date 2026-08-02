"""Exact algebra checks for :class:`DDPMScoreAdapter`.

Uses a mock network with a known, deterministic output so the conversion
``score = -eps / sqrt(1 - alpha_bar_t)`` can be verified to floating-point
precision without training anything.
"""
import pytest
import torch

from models.dag_diffusion.score_adapter import DDPMScoreAdapter
from models.ddpm.core.ddpm_torch.toy import GaussianDiffusion, get_beta_schedule

TIMESTEPS = 40


class ConstantEpsNet(torch.nn.Module):
    """Returns ``x_t * a + b * (t + 1)`` — deterministic and t-dependent."""

    def __init__(self, dim, a=0.5, b=0.1):
        super().__init__()
        self.input_dim = dim
        self.a, self.b = a, b
        # a parameter so `next(model.parameters()).device` works
        self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x, t):
        return self.a * x + self.b * (t.to(x.dtype).unsqueeze(1) + 1.0)


class X0Net(ConstantEpsNet):
    pass


def make_diffusion(mean_type="eps"):
    betas = get_beta_schedule("linear", beta_start=0.001, beta_end=0.2, timesteps=TIMESTEPS)
    return GaussianDiffusion(betas=betas, model_mean_type=mean_type,
                             model_var_type="fixed-large", loss_type="mse")


def test_eps_score_matches_closed_form_scalar_t():
    dim, n, t = 4, 6, 7
    diffusion = make_diffusion("eps")
    net = ConstantEpsNet(dim)
    adapter = DDPMScoreAdapter(net, diffusion, device="cpu")

    x = torch.randn(n, dim, generator=torch.Generator().manual_seed(0))
    got = adapter.score(x, t)

    t_vec = torch.full((n,), t, dtype=torch.int64)
    eps = net(x, t_vec)
    sigma = float((1.0 - diffusion.alphas_bar.to(torch.float32)[t]).sqrt())
    expected = -eps / sigma

    assert got.shape == x.shape
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_eps_score_batched_timesteps():
    dim, n = 3, 5
    diffusion = make_diffusion("eps")
    net = ConstantEpsNet(dim)
    adapter = DDPMScoreAdapter(net, diffusion, device="cpu")

    x = torch.randn(n, dim, generator=torch.Generator().manual_seed(1))
    t_vec = torch.tensor([0, 1, 5, 11, 39], dtype=torch.int64)
    got = adapter.score(x, t_vec)

    # Match the adapter's float32 computation order: cast alphas_bar first,
    # then take the square root (at t=0 sigma is tiny, so the order matters).
    alphas_bar32 = diffusion.alphas_bar.to(torch.float32)
    sigma = (1.0 - alphas_bar32[t_vec]).sqrt().unsqueeze(1)
    expected = -net(x, t_vec) / sigma
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_scalar_and_zero_dim_tensor_agree():
    dim, n, t = 3, 4, 9
    diffusion = make_diffusion("eps")
    adapter = DDPMScoreAdapter(ConstantEpsNet(dim), diffusion, device="cpu")
    x = torch.randn(n, dim, generator=torch.Generator().manual_seed(2))
    torch.testing.assert_close(adapter.score(x, t), adapter.score(x, torch.tensor(t)))


def test_x0_parameterisation_score():
    dim, n, t = 3, 5, 6
    diffusion = make_diffusion("x_0")
    net = X0Net(dim)
    adapter = DDPMScoreAdapter(net, diffusion, device="cpu")

    x = torch.randn(n, dim, generator=torch.Generator().manual_seed(3))
    got = adapter.score(x, t)

    t_vec = torch.full((n,), t, dtype=torch.int64)
    x0_hat = net(x, t_vec)
    alpha_bar = float(diffusion.alphas_bar[t])
    expected = (alpha_bar ** 0.5 * x0_hat - x) / (1.0 - alpha_bar)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_unsupported_mean_type_raises():
    diffusion = make_diffusion("eps")
    diffusion.model_mean_type = "mean"
    with pytest.raises(ValueError, match="model_mean_type"):
        DDPMScoreAdapter(ConstantEpsNet(3), diffusion, device="cpu")


@pytest.mark.parametrize("bad_t", [-1, TIMESTEPS, TIMESTEPS + 5])
def test_out_of_range_timestep_raises(bad_t):
    adapter = DDPMScoreAdapter(ConstantEpsNet(3), make_diffusion("eps"), device="cpu")
    with pytest.raises(ValueError, match="out of range"):
        adapter.score(torch.zeros(2, 3), bad_t)


def test_bad_shape_raises():
    adapter = DDPMScoreAdapter(ConstantEpsNet(3), make_diffusion("eps"), device="cpu")
    with pytest.raises(ValueError, match=r"shape \(N, D\)"):
        adapter.score(torch.zeros(3), 1)


def test_nonfinite_input_raises():
    adapter = DDPMScoreAdapter(ConstantEpsNet(3), make_diffusion("eps"), device="cpu")
    x = torch.zeros(2, 3)
    x[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        adapter.score(x, 1)


def test_timestep_tensor_length_mismatch_raises():
    adapter = DDPMScoreAdapter(ConstantEpsNet(3), make_diffusion("eps"), device="cpu")
    with pytest.raises(ValueError, match="batch size"):
        adapter.score(torch.zeros(4, 3), torch.tensor([0, 1, 2], dtype=torch.int64))


def test_output_is_finite():
    adapter = DDPMScoreAdapter(ConstantEpsNet(5), make_diffusion("eps"), device="cpu")
    x = torch.randn(8, 5, generator=torch.Generator().manual_seed(4))
    assert torch.isfinite(adapter.score(x, 3)).all()
