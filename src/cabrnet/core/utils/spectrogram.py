import torch
from einops import repeat
from loguru import logger
from torch import Tensor, nn


def _create_triangular_filterbank(
    all_freqs: Tensor,
    f_pts: Tensor,
) -> Tensor:
    """Create a triangular filter bank.

    Args:
        all_freqs (Tensor): STFT freq points of size (`n_freqs`).
        f_pts (Tensor): Filter mid points of size (`n_filter`).

    Returns:
        fb (Tensor): The filter bank of size (`n_freqs`, `n_filter`).
    """
    # Adopted from Librosa
    # calculate the difference between each filter mid point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_filter + 2)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))

    return fb


def logscale_fbanks(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_bands: int,
    sample_rate: int,
) -> Tensor:
    r"""Create a frequency bin conversion matrix with logarithmic spacing.

    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        f_min (float): Minimum frequency (Hz)
        f_max (float): Maximum frequency (Hz)
        n_bands (int): Number of log filterbanks
        sample_rate (int): Sample rate of the audio waveform
            (area normalization). (Default: ``None``)

    Returns:
        Tensor: Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_bands``)
    """

    # freq bins (linear spacing from 0 to Nyquist)
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate log-spaced freq bins (simple logarithmic spacing)
    f_pts = torch.logspace(
        torch.log10(torch.tensor(f_min)), torch.log10(torch.tensor(f_max)), n_bands + 2
    )

    # create filterbank
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    zero_bands = (fb.max(dim=0).values == 0.0).nonzero(as_tuple=True)[0].tolist()
    if zero_bands:
        logger.warning(
            f"Filterbank bands {zero_bands} have all zero values. \n"
            + "This means those frequency filters contain no STFT bin. \n"
            + f"To fix: increase `n_freqs`, decrease `n_bands`, or raise `f_min` (currently {f_min} Hz)\n"
            + f"so the lowest filter spans at least one bin (bin resolution is {(sample_rate / 2) / (n_freqs - 1):.2f} Hz). \n"
            + f"Also verify `f_max` ({f_max} Hz) does not exceed Nyquist ({sample_rate / 2} Hz)."
        )

    return fb


class SpectroToImg(nn.Module):
    """Module that converts FFT complex tensors to spectrograms.

    Can be batched and moved to GPU for efficient processing.

    Args:
        spectro_config: spectrogram configuration for filterbank and post-processing
        n_fft: FFT size (loaded from params.json if not provided)
        sample_rate: sample rate (loaded from params.json if not provided)
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        n_bands: int,
        f_min: float,
        f_max: float,
        alpha: float,
        flip: bool = False,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max
        self.alpha = alpha
        self.flip = flip

    def forward(self, spec: Tensor) -> Tensor:
        """Convert FFT complex tensor to spectrogram.

        Args:
            spec: complex STFT tensor of shape (batch, freq, time)

        Returns:
            spectrogram tensor of shape (batch, channels, freq, time)
        """
        b, h_fft, t = spec.shape
        spec_norm = torch.abs(spec) ** 2

        filterbank = logscale_fbanks(
            n_freqs=h_fft,
            n_bands=self.n_bands,
            sample_rate=self.sample_rate,
            f_min=self.f_min,
            f_max=self.f_max,
        ).to(spec.device)
        logspec = filterbank.T @ spec_norm
        logspec = torch.pow(logspec, self.alpha)
        logspec = logspec / (logspec.amax(dim=(1, 2), keepdim=True).abs() + 1e-6)

        if self.flip:
            return logspec.flip(-2)
        return logspec


class Repeat3Channel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return repeat(
            input, "b f t -> b c f t", c=3
        )  # 3 channels for RGB compatibility
        return
