import pytest
import torch

from torch_istft_onnx.istft import ISTFT


class TestIstft:
    @pytest.fixture
    def n_fft(self) -> int:
        return 256

    @pytest.fixture
    def hop_length(self) -> int:
        return 64

    @pytest.fixture
    def istft(self, n_fft: int, hop_length: int) -> torch.nn.Module:
        istft = ISTFT(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=torch.hann_window(n_fft), normalized=True
        )
        return istft

    def test_forward_pass(self, n_fft: int, istft: ISTFT):
        # freqs x frames x m
        input = torch.randn(n_fft + 2, 20, 2)  # last dim is 2 for real and imaginary parts
        output = istft(input)
        assert len(output.shape) == 2

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_torch_stft_inversion(self, batch_size: int, n_fft: int, hop_length: int, istft: ISTFT):
        input = torch.randn((batch_size, 4096))

        spectro = torch.stft(
            input,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=torch.hann_window(n_fft),
            normalized=True,
            return_complex=False,  # not return complex tensor
        )

        output = istft(spectro)

        assert len(output.shape) == 2
        assert output.shape[1] == (input.shape[1] // hop_length) * hop_length

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_consistent_values_with_torch_istft(self, batch_size: int, n_fft: int, hop_length: int, istft: ISTFT):
        input = torch.randn((batch_size, 4096))
        spectro = torch.stft(
            input,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=torch.hann_window(n_fft),
            normalized=True,
            center=True,
            return_complex=False,  # not return complex tensor
        )

        # convert the spectro to complex tensors
        spectro_complex = torch.complex(spectro[..., 0], spectro[..., 1])
        torch_output = torch.istft(
            spectro_complex,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=torch.hann_window(n_fft),
            normalized=True,
        )

        output = istft(spectro)

        assert torch_output.shape == output.shape
        # issue with high error in the last `hop_length` samples
        assert (torch_output[..., :-hop_length] - output[..., :-hop_length]).abs().mean() < 1e-5
