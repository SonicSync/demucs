# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Code to apply a model to a mix. It will handle chunking with overlaps and
inteprolation between chunks, as well as the "shift trick".
"""
from concurrent.futures import ThreadPoolExecutor
import copy
import random
from threading import Lock
import typing as tp

import torch as th
from torch import nn
from torch.nn import functional as F
import tqdm

from .demucs import Demucs
from .hdemucs import HDemucs
from .htdemucs import HTDemucs
from .utils import center_trim, DummyPoolExecutor

Model = tp.Union[Demucs, HDemucs, HTDemucs]


class BagOfModels(nn.Module):
    def __init__(self, models: tp.List[Model],
                 weights: tp.Optional[tp.List[tp.List[float]]] = None,
                 segment: tp.Optional[float] = None):
        """
        Represents a bag of models with specific weights.
        You should call `apply_model` rather than calling directly the forward here for
        optimal performance.

        Args:
            models (list[nn.Module]): list of Demucs/HDemucs models.
            weights (list[list[float]]): list of weights. If None, assumed to
                be all ones, otherwise it should be a list of N list (N number of models),
                each containing S floats (S number of sources).
            segment (None or float): overrides the `segment` attribute of each model
                (this is performed inplace, be careful is you reuse the models passed).
        """
        super().__init__()
        assert len(models) > 0
        first = models[0]
        for other in models:
            assert other.sources == first.sources
            assert other.samplerate == first.samplerate
            assert other.audio_channels == first.audio_channels
            if segment is not None:
                if not isinstance(other, HTDemucs) and segment > other.segment:
                    other.segment = segment

        self.audio_channels = first.audio_channels
        self.samplerate = first.samplerate
        self.sources = first.sources
        self.models = nn.ModuleList(models)

        if weights is None:
            weights = [[1. for _ in first.sources] for _ in models]
        else:
            assert len(weights) == len(models)
            for weight in weights:
                assert len(weight) == len(first.sources)
        self.weights = weights

    @property
    def max_allowed_segment(self) -> float:
        max_allowed_segment = float('inf')
        for model in self.models:
            if isinstance(model, HTDemucs):
                max_allowed_segment = min(max_allowed_segment, float(model.segment))
        return max_allowed_segment

    def forward(self, x):
        raise NotImplementedError("Call `apply_model` on this.")


class TensorChunk:
    def __init__(self, tensor, offset=0, length=None):
        total_length = tensor.shape[-1]
        assert offset >= 0
        assert offset < total_length

        if length is None:
            length = total_length - offset
        else:
            length = min(total_length - offset, length)

        if isinstance(tensor, TensorChunk):
            self.tensor = tensor.tensor
            self.offset = offset + tensor.offset
        else:
            self.tensor = tensor
            self.offset = offset
        self.length = length
        self.device = tensor.device

    @property
    def shape(self):
        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape

    def padded(self, target_length):
        delta = target_length - self.length
        total_length = self.tensor.shape[-1]
        assert delta >= 0

        start = self.offset - delta // 2
        end = start + target_length

        correct_start = max(0, start)
        correct_end = min(total_length, end)

        pad_left = correct_start - start
        pad_right = end - correct_end

        out = F.pad(self.tensor[..., correct_start:correct_end], (pad_left, pad_right))
        assert out.shape[-1] == target_length
        return out


def tensor_chunk(tensor_or_chunk):
    if isinstance(tensor_or_chunk, TensorChunk):
        return tensor_or_chunk
    else:
        assert isinstance(tensor_or_chunk, th.Tensor)
        return TensorChunk(tensor_or_chunk)


def _replace_dict(_dict: tp.Optional[dict], *subs: tp.Tuple[tp.Hashable, tp.Any]) -> dict:
    if _dict is None:
        _dict = {}
    else:
        _dict = copy.copy(_dict)
    for key, value in subs:
        _dict[key] = value
    return _dict


def apply_model(model: tp.Union[BagOfModels, Model],
                mix: tp.Union[th.Tensor, TensorChunk],
                shifts: int = 1, split: bool = True,
                overlap: float = 0.25, transition_power: float = 1.,
                progress: bool = False, device=None,
                num_workers: int = 0, segment: tp.Optional[float] = None,
                pool=None, lock=None,
                callback: tp.Optional[tp.Callable[[dict], None]] = None,
                callback_arg: tp.Optional[dict] = None) -> tp.Optional[th.Tensor]:

    # If device is not specified, use the device of the 'mix' tensor.
    # If 'device' is specified, convert it to a PyTorch device.
    if device is None:
        device = mix.device
    else:
        device = th.device(device)

    # If 'pool' is not specified, create a ThreadPoolExecutor for parallel execution
    # if 'num_workers' is greater than 0 and the device is 'cpu'.
    # Otherwise, use a DummyPoolExecutor.
    if pool is None:
        if num_workers > 0 and device.type == 'cpu':
            pool = ThreadPoolExecutor(num_workers)
        else:
            pool = DummyPoolExecutor()

    # If 'lock' is not specified, create a Lock for synchronization.
    if lock is None:
        lock = Lock()

    # Set default values for callback_arg.
    callback_arg = _replace_dict(
        callback_arg, *{"model_idx_in_bag": 0, "shift_idx": 0, "segment_offset": 0}.items()
    )

    # Define a dictionary 'kwargs' containing various keyword arguments.
    kwargs: tp.Dict[str, tp.Any] = {
        'shifts': shifts,
        'split': split,
        'overlap': overlap,
        'transition_power': transition_power,
        'progress': progress,
        'device': device,
        'pool': pool,
        'segment': segment,
        'lock': lock,
    }

    # Initialize 'out' and 'res' as placeholders for output tensors.
    out: tp.Union[float, th.Tensor]
    res: tp.Union[float, th.Tensor, None]

    # Check if the 'model' parameter is a BagOfModels.
    if isinstance(model, BagOfModels):
        # Special treatment for BagOfModels.
        # Iterate through each sub-model in the bag and apply it to the 'mix' input.
        estimates: tp.Union[float, th.Tensor] = 0.
        totals = [0.] * len(model.sources)
        callback_arg["models"] = len(model.models)

        # Define a callback function for progress tracking.
        kwargs["callback"] = (
            (lambda d, i=callback_arg["model_idx_in_bag"]: callback(
                _replace_dict(d, ("model_idx_in_bag", i))
            ))
            if callable(callback)
            else None
        )

        for sub_model, model_weights in zip(model.models, model.weights):
            original_model_device = next(iter(sub_model.parameters())).device
            sub_model.to(device)
            # Apply the sub-model to the 'mix' input.
            res = apply_model(sub_model, mix, **kwargs, callback_arg=callback_arg)

            if res is None:
                return res

            out = res
            sub_model.to(original_model_device)

            # Apply model-specific weights to the output.
            for k, inst_weight in enumerate(model_weights):
                out[:, k, :, :] *= inst_weight
                totals[k] += inst_weight
            estimates += out
            del out
            callback_arg["model_idx_in_bag"] += 1

        # Normalize the estimates and return the result.
        assert isinstance(estimates, th.Tensor)
        for k in range(estimates.shape[1]):
            estimates[:, k, :, :] /= totals[k]
        return estimates

    # If 'models' is not in callback_arg, set it to 1.
    if "models" not in callback_arg:
        callback_arg["models"] = 1

    # Set the model's device, put it in evaluation mode, and define the minimum transition power.
    model.to(device)
    model.eval()
    assert transition_power >= 1, "transition_power < 1 leads to weird behavior."

    # Extract the dimensions of the 'mix' tensor.
    batch, channels, length = mix.shape

    if shifts:
        # If shifts are enabled, apply the model with random shifts.
        kwargs['shifts'] = 0
        max_shift = int(0.5 * model.samplerate)
        mix = tensor_chunk(mix)
        assert isinstance(mix, TensorChunk)

        # Pad the 'mix' tensor to handle shifts efficiently.
        padded_mix = mix.padded(length + 2 * max_shift)
        out = 0.

        for shift_idx in range(shifts):
            offset = random.randint(0, max_shift)
            shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)
            kwargs["callback"] = (
                (lambda d, i=shift_idx: callback(_replace_dict(d, ("shift_idx", i)))
                 )
                if callable(callback)
                else None
            )
            # Apply the model to the shifted input.
            res = apply_model(model, shifted, **kwargs, callback_arg=callback_arg)

            if res is None:
                return res

            shifted_out = res
            out += shifted_out[..., max_shift - offset:]

        out /= shifts
        assert isinstance(out, th.Tensor)
        return out

    elif split:
        # If splitting is enabled, apply the model in overlapping segments.
        kwargs['split'] = False
        out = th.zeros(batch, len(model.sources), channels, length, device=mix.device)
        sum_weight = th.zeros(length, device=mix.device)

        if segment is None:
            segment = model.segment

        assert segment is not None and segment > 0.
        segment_length: int = int(model.samplerate * segment)
        stride = int((1 - overlap) * segment_length)
        offsets = range(0, length, stride)
        scale = float(format(stride / model.samplerate, ".2f"))

        # Define a weight function for segment transitions.
        weight = th.cat([th.arange(1, segment_length // 2 + 1, device=device),
                         th.arange(segment_length - segment_length // 2, 0, -1, device=device)])
        assert len(weight) == segment_length
        weight = (weight / weight.max())**transition_power

        # Create futures for parallel execution of model on segments.
        futures = []

        for offset in offsets:
            chunk = TensorChunk(mix, offset, segment_length)
            future = pool.submit(apply_model, model, chunk, **kwargs, callback_arg=callback_arg,
                                 callback=(lambda d, i=offset:
                                           callback(_replace_dict(d, ("segment_offset", i))))
                                 if callable(callback) else None)
            futures.append((future, offset))
            offset += segment_length

        if progress:
            # Display a progress bar if 'progress' is set to True.
            futures = tqdm.tqdm(futures, unit_scale=scale, ncols=120, unit='seconds')

        for future, offset in futures:
            chunk_out = future.result()  # Get the result from the future.
            if chunk_out is None:
                pool.shutdown(wait=False, cancel_futures=True)
                return chunk_out

            chunk_length = chunk_out.shape[-1]
            out[..., offset:offset + segment_length] += (
                weight[:chunk_length] * chunk_out).to(mix.device)
            sum_weight[offset:offset + segment_length] += weight[:chunk_length].to(mix.device)

        assert sum_weight.min() > 0
        out /= sum_weight
        assert isinstance(out, th.Tensor)
        return out

    else:
        # If neither shifts nor splitting is enabled, apply the model to the entire input.
        valid_length: int

        if isinstance(model, HTDemucs) and segment is not None:
            valid_length = int(segment * model.samplerate)
        elif hasattr(model, 'valid_length'):
            valid_length = model.valid_length(length)
        else:
            valid_length = length

        mix = tensor_chunk(mix)
        assert isinstance(mix, TensorChunk)

        # Pad the 'mix' tensor to match the valid length.
        padded_mix = mix.padded(valid_length).to(device)

        with lock:
            try:
                callback(_replace_dict(callback_arg, ("state", "start")))  # Execute the start callback.
            except KeyboardInterrupt:
                raise
            except Exception:
                pass

        with th.no_grad():
            # Apply the model to the padded mix.
            assert padded_mix.dtype == th.float32
            out = model(padded_mix)

        with lock:
            try:
                callback(_replace_dict(callback_arg, ("state", "end")))  # Execute the end callback.
            except KeyboardInterrupt:
                raise
            except Exception:
                pass

        assert isinstance(out, th.Tensor)

        # Trim the output to the original length.
        return center_trim(out, length)

