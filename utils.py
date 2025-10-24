import re
import os
import importlib.util
import torch
import lightning as pl
from pathlib import Path
import glob
import shutil
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math

def batch_pad_right(tensors: list, mode="constant", value=0):
    """
    COPY FROM SPEECHBRAIN
    Given a list of torch tensors it batches them together by padding to the right
    on each dimension in order to get same length for all.

    Parameters
    ----------
    tensors : list
        List of tensor we wish to pad together.
    mode : str
        Padding mode see torch.nn.functional.pad documentation.
    value : float
        Padding value see torch.nn.functional.pad documentation.

    Returns
    -------
    tensor : torch.Tensor
        Padded tensor.
    valid_vals : list
        List containing proportion for each dimension of original, non-padded values.

    """

    if not len(tensors):
        raise IndexError("Tensors list must not be empty")

    if len(tensors) == 1:
        # if there is only one tensor in the batch we simply unsqueeze it.
        return tensors[0].unsqueeze(0), torch.tensor([1.0])

    if not (
        all(
            [tensors[i].ndim == tensors[0].ndim for i in range(1, len(tensors))]
        )
    ):
        raise IndexError("All tensors must have same number of dimensions")

    # FIXME we limit the support here: we allow padding of only the first dimension
    # need to remove this when feat extraction is updated to handle multichannel.
    max_shape = []
    for dim in range(tensors[0].ndim):
        if dim != 0:
            if not all(
                [x.shape[dim] == tensors[0].shape[dim] for x in tensors[1:]]
            ):
                raise EnvironmentError(
                    "Tensors should have same dimensions except for the first one"
                )
        max_shape.append(max([x.shape[dim] for x in tensors]))

    batched = []
    valid = []
    for t in tensors:
        # for each tensor we apply pad_right_to
        padded, valid_percent = pad_right_to(
            t, max_shape, mode=mode, value=value
        )
        batched.append(padded)
        valid.append(valid_percent[0])

    batched = torch.stack(batched)

    return batched, torch.tensor(valid)


def pad_right_to(
    tensor: torch.Tensor, target_shape: (list, tuple), mode="constant", value=0,
):
    """
    COPY FROM SPEECHBRAIN
    This function takes a torch tensor of arbitrary shape and pads it to target
    shape by appending values on the right.

    Parameters
    ----------
    tensor : input torch tensor
        Input tensor whose dimension we need to pad.
    target_shape : (list, tuple)
        Target shape we want for the target tensor its len must be equal to tensor.ndim
    mode : str
        Pad mode, please refer to torch.nn.functional.pad documentation.
    value : float
        Pad value, please refer to torch.nn.functional.pad documentation.

    Returns
    -------
    tensor : torch.Tensor
        Padded tensor.
    valid_vals : list
        List containing proportion for each dimension of original, non-padded values.
    """
    assert len(target_shape) == tensor.ndim
    pads = []  # this contains the abs length of the padding for each dimension.
    valid_vals = []  # this contains the relative lengths for each dimension.
    i = len(target_shape) - 1  # iterating over target_shape ndims
    j = 0
    while i >= 0:
        assert (
            target_shape[i] >= tensor.shape[i]
        ), "Target shape must be >= original shape for every dim"
        pads.extend([0, target_shape[i] - tensor.shape[i]])
        valid_vals.append(tensor.shape[j] / target_shape[j])
        i -= 1
        j += 1

    tensor = torch.nn.functional.pad(tensor, pads, mode=mode, value=value)

    return tensor, valid_vals


def length_to_mask(length, max_len=None, dtype=None, device=None):
    """
    COPY FROM SPEECHBRAIN
    Creates a binary mask for each sequence.

    Reference: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3

    Arguments
    ---------
    length : torch.LongTensor
        Containing the length of each sequence in the batch. Must be 1D.
    max_len : int
        Max length for the mask, also the size of the second dimension.
    dtype : torch.dtype, default: None
        The dtype of the generated mask.
    device: torch.device, default: None
        The device to put the mask variable.

    Returns
    -------
    mask : tensor
        The binary mask.

    Example
    -------
    >>> length=torch.Tensor([1,2,3])
    >>> mask=length_to_mask(length)
    >>> mask
    tensor([[1., 0., 0.],
            [1., 1., 0.],
            [1., 1., 1.]])
    """
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()  # using arange to generate mask
    mask = torch.arange(
        max_len, device=length.device, dtype=length.dtype
    ).expand(len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask

def select_latest_ckpt(ckpt_dir: str):
    ckpt_dir = Path(ckpt_dir)
    ckpts = sorted(glob.glob(str(ckpt_dir / "*.ckpt")), key=extract_number)
    hpc_ckpts = sorted(glob.glob(str(ckpt_dir / "hpc_ckpt_*.ckpt")), key=extract_number)
    # cleanup older hpc checkpoints (keep last)
    for p in hpc_ckpts[:-1]:
        try:
            if os.path.isfile(p) or os.path.islink(p):
                os.remove(p)
            elif os.path.isdir(p):
                shutil.rmtree(p)
        except FileNotFoundError:
            pass
    # choose most recent between last hpc and last ckpt
    candidate = None
    if hpc_ckpts and ckpts:
        try:
            if os.path.getmtime(hpc_ckpts[-1]) > os.path.getmtime(ckpts[-1]) and os.listdir(hpc_ckpts[-1]):
                candidate = "hpc"
            else:
                candidate = ckpts[-1]
        except OSError:
            candidate = ckpts[-1]
    elif hpc_ckpts:
        candidate = "hpc"
    elif ckpts:
        candidate = ckpts[-1]
    return candidate

class SaveAtSpecificStep(pl.Callback):
    def __init__(self, save_steps=100000, ckpt_dir=None):
        self.save_steps = save_steps
        self.ckpt_dir = ckpt_dir

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.save_steps == 0:
            print(f"Save checkpoint at step {trainer.global_step}")
            checkpoint_path = f"{self.ckpt_dir}/checkpoint_at_step_{trainer.global_step}.ckpt"
            trainer.save_checkpoint(checkpoint_path)

            
def writing_output_to_file(output, output_dir, token=False):
    if token:
        f_token_loss = open(os.path.join(output_dir, f"token_loss.txt"), "w")

    with open(os.path.join(output_dir, f"loss.txt"), "w") as f_loss:
        for batch in output:
            if token:
                for id, loss, token_loss in zip(*batch):
                    if type(loss) == torch.Tensor:
                        loss = loss.cpu().numpy()
                        # turn into string
                        loss = " ".join([str(l) for l in loss])
                    if type(token_loss) == torch.Tensor:
                        token_loss = token_loss.cpu().numpy()
                        token_loss = " ".join([str(l) for l in token_loss])
                    f_loss.write(f"{id} {loss}\n")
                    f_token_loss.write(f"{id} {token_loss}\n")
            else:
                for id, loss in zip(*batch):
                    if type(loss) == torch.Tensor:
                        loss = loss.cpu().numpy()
                        loss = " ".join([str(l) for l in loss])
                    f_loss.write(f"{id} {loss}\n")
    return

def import_module_from_path(module_name, module_path):
    try:
        # Create a spec for the module
        module_spec = importlib.util.spec_from_file_location(module_name, module_path)
        # Load the module based on the spec
        custom_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(custom_module)
        # Set the name attribute of the module
        custom_module.__name__ = module_name
        #print(f"Successfully imported module '{module_name}' from path '{module_path}'")
        return custom_module
    except Exception as e:
        print(f"Failed to import module from path '{module_path}': {e}")
        return None

def extract_number(file_path):
    # Extract the file name from the path
    file_name = file_path.split('/')[-1]
    
    # Extract the numeric part using regular expression
    match = re.search(r'\d+', file_name)
    if match:
        return int(match.group())
    else:
        return 0

def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1, min_lr_ratio: float = 0.1):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_periods (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0, min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def replace_values(original_dict, replacement_dict):
    for key, value in replacement_dict.items():
        if isinstance(value, dict):
            if key in original_dict and isinstance(original_dict[key], dict):
                replace_values(original_dict[key], value)
        else:
            if key in original_dict:
                original_dict[key] = value