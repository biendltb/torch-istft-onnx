"""
Example of ONNX export with torch.stft (supported by ONNX) and this implementation of torch.istft
"""
import torch
from torch import nn
from torch_istft_onnx.istft import ISTFT


N_FFT = 512
HOP_LENGTH = 128


class ExampleSignalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.istft = ISTFT(
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            window=torch.hann_window(N_FFT),
            normalized=True,
        )

    def forward(self, x):
        spec = torch.stft(
            x,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            window=torch.hann_window(N_FFT),
            normalized=True,
            return_complex=False,  # not return complex tensor
        )
        reconstructed = self.istft(spec)
        return reconstructed


if __name__ == "__main__":
    model = ExampleSignalModel()
    x = torch.randn(1, 65535)
    # test forward pass
    with torch.no_grad():
        _ = model(x)

    print("Export to ONNX")
    torch.onnx.export(
        model,
        x,
        "/tmp/model.onnx",
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={"mix": {0: "batch_size", 2: "sample_length"}},
        export_params=True,
        opset_version=17,
        verbose=False,
    )
    print("Done")
