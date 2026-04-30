from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch import Tensor
from torch.utils.data.dataset import Dataset
from torchaudio.datasets import SPEECHCOMMANDS
from tqdm import trange


@dataclass
class SpectroConfig:
    """FFT parameters for STFT computation. Serializable to/from YAML."""

    sample_rate: int
    # size of the buffer used to compute each spectrum. Note that we use the same value for `n_fft` and `win_length`
    n_fft: int
    #
    hop_time: float
    # the length of the audio signal. If the signal is shorter, it will be padded with zeros
    total_length: int
    # the name of the hindow to apply in torchaudio
    window: str = "hann_window"

    @property
    def audio_time(self):
        return self.total_length / self.sample_rate

    @property
    def hop_length(self):
        return int(self.hop_time * self.sample_rate)

    @property
    def width(self):
        return int(self.total_length / self.hop_length) + 1

    def stft(self, input_audio):
        window_fn = getattr(torch, self.window, torch.hann_window)
        assert isinstance(window_fn, Tensor)

        return torch.stft(
            input_audio,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            normalized=False,
            window=window_fn,
            center=True,
            return_complex=True,
        )


class LabeledSpectroDataset(ABC, Dataset[tuple[Tensor, int]]):
    @abstractmethod
    def __init__(
        self,
        root: Path,
        subset: str,
        *,
        n_fft: int,
        hop_time: float,
        width: int,
        window_fn: "str" = "hann_window",
        transform: torch.nn.Module | None = None,
    ): ...

    @abstractmethod
    def __len__(self) -> int: ...

    @property
    @abstractmethod
    def root_path(self) -> Path:
        """
        The path where the dataset will be stored (both audio and spectrograms)
        """

    @property
    @abstractmethod
    def classes(self) -> list[int]: ...

    @abstractmethod
    def get_class(self, idx: int) -> int: ...

    @property
    @abstractmethod
    def spectro_config(self) -> SpectroConfig: ...

    @property
    @abstractmethod
    def transform(self) -> torch.nn.Module:
        """
        A transform applied to every input
        """

    @abstractmethod
    def get_audio(self, idx: int) -> Tensor:
        """
        Shape (1, L) where L is equal to `self.audio_length()`
        """

    @abstractmethod
    def get_identifier(self, idx: int) -> str:
        """
        Return a unique ID for this element, as a string.
        This is usually the file name
        """

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        """
        Returns:
            Return the complex spectrogram tensor with shape (B, F, T), with a label.
        """
        path = self.spectro_path
        if not path.exists():
            raise ValueError("The spectrograms have not been generated. Call `generate_spectrograms` before using")
        file_id = self.get_identifier(idx)
        data = np.load(path / f"{file_id}.npy").astype(np.float32)
        spec = torch.view_as_complex(torch.from_numpy(data))
        return self.transform(spec.unsqueeze(0)).squeeze(0), int(self.get_class(idx))

    @property
    def spectro_path(self) -> Path:
        n_fft = self.spectro_config.n_fft
        hop_length = self.spectro_config.hop_length
        return self.root_path / f"spectro_{n_fft}_{hop_length}"

    def generate_spectrograms(self, force_generate):
        self.spectro_path.mkdir(parents=True, exist_ok=True)

        for i in trange(
            len(self),
            desc=f"Creating STFTs in {self.spectro_path}",
        ):
            sound = self.get_audio(i)
            sample_length = sound.shape[1]
            config = self.spectro_config
            if sample_length != config.total_length:
                raise ValueError(
                    f"sound {i} of has incorrect length: expected {config.total_length} from config, but sample has length {sample_length}"
                )
            spec = config.stft(sound)[0]
            spec_stored = torch.stack([spec.real, spec.imag], dim=-1).half()
            file_id = self.get_identifier(i)
            save_path = self.spectro_path / f"{file_id}.npy"
            if save_path.exists() and not force_generate:
                logger.info(f"spectrogram {save_path} already generated, skipping dataset. Use `force` to override")
                return
            np.save(save_path, spec_stored.cpu().numpy())


class SpeechCommandsSpectroDataset(LabeledSpectroDataset, SPEECHCOMMANDS):
    def __init__(
        self,
        root: Path | str,
        subset: str,
        *,
        n_fft: int,
        hop_time: float,
        width: int,
        window: "str" = "hann_window",
        transform: torch.nn.Module | None = None,
    ):
        root = Path(root)
        root_audio = root / "audio"
        root_audio.mkdir(exist_ok=True)
        SPEECHCOMMANDS.__init__(self, root=root_audio, subset=subset, download=True)
        self.classname_to_label = dict()
        self.max_length = 16_000
        all_classes = set([self.get_metadata(i)[2] for i in range(len(self))])
        all_classes = sorted(list(all_classes))
        self.classname_to_label = {classname: i for i, classname in enumerate(all_classes)}
        self._classes = list(self.classname_to_label)

        self._root = Path(root)
        self._transform = transform or torch.nn.Identity()

        # same length for all audio samples
        self._spectro_config = SpectroConfig(
            sample_rate=16_000,
            n_fft=n_fft,
            total_length=16_000,
            hop_time=hop_time,
        )
        assert self._spectro_config.width == width

    @property
    def classes(self):
        return self._classes

    def __len__(self):
        return SPEECHCOMMANDS.__len__(self)

    def get_class(self, idx: int):
        _, _, classname, _, _ = SPEECHCOMMANDS.get_metadata(self, idx)
        return self.classname_to_label[classname]

    def get_identifier(self, idx: int):
        audio_path, _, _, _, _ = SPEECHCOMMANDS.get_metadata(self, idx)
        return Path(audio_path).parts[-2] + "_" + Path(audio_path).stem

    def get_audio(self, idx: int):
        sound, _sample_rate, _, _, _ = SPEECHCOMMANDS.__getitem__(self, idx)
        assert sound.ndim == 2
        assert sound.shape[1] <= self.max_length
        padded_sound = torch.zeros(1, self.max_length)
        padded_sound[:, : len(sound[0])] = sound[0]
        return padded_sound

    @property
    def spectro_config(self) -> SpectroConfig:
        return self._spectro_config

    @property
    def sample_rate(self) -> int:
        # constant sample rate, 1s audio samples
        return 16000

    @property
    def root_path(self):
        return self._root

    @property
    def hop_length(self) -> int:
        return self._hop_length

    @property
    def window(self):
        return self._window

    @property
    def transform(self) -> torch.nn.Module:
        return self._transform


if __name__ == "__main__":
    speechcommand = SpeechCommandsSpectroDataset(
        "data/Speechcommands/dataset/validation",
        "validation",
        n_fft=1024,
        hop_time=0.01,
        window="hann_window",
        width=101,
    )
