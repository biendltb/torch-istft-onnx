# Torch iSTFT ONNX

An ONNX-exportable implementation of Inverse Short-Time Fourier Transform (iSTFT).


## Why this implementation?

There are many conflicts in the development roadmaps of torch.onnx and the official ONNX regarding STFT.

Context:
* PyTorch is about to force using complex numbers in [torch.stft](https://pytorch.org/docs/stable/generated/torch.stft.html#torch.stft) and
[torch.istft](https://pytorch.org/docs/stable/generated/torch.istft.html#torch.istft) for their inputs and output respectively.
* ONNX does not yet support complex tensors.
* Upon my current version of torch (2.4.1), `return_complex=False` is still allowed in `torch.stft()` but it will soon be
removed. However, we still can export torch models that using `torch.stft` in the mean time.

So what's about `torch.istft`? PyTorch already enforce the complex tensor input for it. Sure, you can use an older version
of PyTorch but ONNX has not yet support `torch.istft` in its latest version (opset 17) (even there is an [unpublised/unfinished work](https://github.com/onnx/onnx/blob/b8baa8446686496da4cc8fda09f2b6fe65c2a02c/onnx/reference/ops/op_stft.py#L77)
for supporting it there). The issue has been [raised multiple times](https://github.com/pytorch/pytorch/issues/81075) but
it's not clear who should take responsible for it, ONNX or PyTorch.

While waiting for the official support, this implementation comes as an alternative. It uses torch operation which is supported
in ONNX to make the export possible.

## Technical details

The implementation is based on [Pseeth's implementation](https://github.com/pseeth/torch-stft) with following improvements:
* Remove dependencies on `numpy`, `scipy` and `librosa`.
* Implement a workaround to avoid OOM errors on GPU with large tensor inputs.
* Support multiple batch sizes.

## Contribution
To get started; install poetry:

  ```bash
  curl -sSL https://install.python-poetry.org | python3 -
  ```

* Install the dependencies and run the unit tests,
    ```bash
    poetry install && poetry shell
    pre-commit install
    python -m pytest
    ```

## Known issues

* Mismatch with `torch.istft` output in the last `hop_length` samples.


## License

MIT License
