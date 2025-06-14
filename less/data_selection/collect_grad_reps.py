import json
import os
from hashlib import md5
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F
from functorch import grad, make_functional_with_buffers, vmap
from peft import PeftModel
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.functional import normalize
from tqdm import tqdm
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
from transformers import RobertaModel


def prepare_batch(batch, device=torch.device("cuda:0")):
    """ Move the batch to the device. """
    for key in batch:
        batch[key] = batch[key].to(device)


def get_max_saved_index(output_dir: str, prefix="reps") -> int:
    """ 
    Retrieve the highest index for which the data (representation, gradients or losses) has been stored. 

    Args:
        output_dir (str): The output directory.
        prefix (str, optional): The prefix of the files, [reps | grads | losses]. Defaults to "reps".

    Returns:
        int: The maximum representation index, or -1 if no index is found.
    """

    files = [file for file in os.listdir(
        output_dir) if file.startswith(prefix)]
    index = [int(file.split(".")[0].split("-")[1])
             for file in files]  # e.g., output_dir/reps-100.pt
    return max(index) if len(index) > 0 else -1


def get_output(model,
               weights: Iterable[Tensor],
               buffers: Iterable[Tensor],
               input_ids=None,
               attention_mask=None,
               labels=None,
               ) -> Tensor:
    logits = model(weights, buffers, *(input_ids.unsqueeze(0),
                   attention_mask.unsqueeze(0))).logits
    labels = labels.unsqueeze(0)
    loss_fct = F.cross_entropy
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
    return loss


def get_trak_projector(device: torch.device):
    """ Get trak projectors (see https://github.com/MadryLab/trak for details) """
    try:
        num_sms = torch.cuda.get_device_properties(
            device.index).multi_processor_count
        import fast_jl

        # test run to catch at init time if projection goes through
        fast_jl.project_rademacher_8(torch.zeros(
            8, 1_000, device=device), 512, 0, num_sms)
        projector = CudaProjector
        print("Using CudaProjector")
    except Exception as e:
        projector = BasicProjector
        print("Using BasicProjector")
        print(f"Caught an exception:\n{e}")
    return projector


def get_number_of_params(model):
    """ Make sure that only lora parameters require gradients in peft models. """
    if isinstance(model, PeftModel):
        names = [n for n, p in model.named_parameters(
        ) if p.requires_grad and "lora" not in n]
        assert len(names) == 0
    num_params = sum([p.numel()
                     for p in model.parameters() if p.requires_grad])
    print(f"Total number of parameters that require gradients: {num_params}")
    return num_params


def obtain_gradients(model, batch):
    """ obtain gradients. """
    loss = model(**batch).loss  # since batch_size default to 1, these are thus loss and loss_grad on a SINGLE training/eval point
    loss.backward()
    vectorized_grads = torch.cat(
        [p.grad.view(-1) for p in model.parameters() if p.grad is not None])
    return vectorized_grads


def obtain_sign_gradients(model, batch):
    """ obtain gradients with sign. """
    loss = model(**batch).loss
    loss.backward()

    # Instead of concatenating the gradients, concatenate their signs
    vectorized_grad_signs = torch.cat(
        [torch.sign(p.grad).view(-1) for p in model.parameters() if p.grad is not None])

    return vectorized_grad_signs


def obtain_gradients_with_adam(model, batch, avg, avg_sq):
    """ obtain gradients with adam optimizer states. """
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-08

    loss = model(**batch).loss
    loss.backward() # must forward and backprop to get each batch of loss grad 

    vectorized_grads = torch.cat(
        [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None])   # since already backward above, all the params of lora layers should have p.grad != None
    # after using torch.cat([...], dim=0 by default), `vectorized_grads` is already flattened

    updated_avg = beta1 * avg + (1 - beta1) * vectorized_grads  # implicitly validates that len(avg) == len(vectorized_grads)
    updated_avg_sq = beta2 * avg_sq + (1 - beta2) * vectorized_grads ** 2
    vectorized_grads = updated_avg / torch.sqrt(updated_avg_sq + eps)

    return vectorized_grads


def prepare_optimizer_state(model, optimizer_state, device):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in names])    # optimizer_state should be a key-value dict instead of index-value
    avg_sq = torch.cat([optimizer_state[n]["exp_avg_sq"].view(-1)
                       for n in names])
    avg = avg.to(device)
    avg_sq = avg_sq.to(device)
    return avg, avg_sq


def collect_grads(dataloader,
                  model,
                  output_dir,
                  proj_dim: List[int] = [8192],
                  adam_optimizer_state: Optional[dict] = None,
                  gradient_type: str = "adam",
                  max_samples: Optional[int] = None):
    """
    Collects gradients from the model during evaluation and saves them to disk.

    Args:
        dataloader (torch.utils.data.DataLoader): The data loader for evaluation dataset.
        model (torch.nn.Module): The model from which gradients will be collected.
        output_dir (str): The directory where the gradients will be saved.
        proj_dim List[int]: The dimensions of the target projectors. Each dimension will be saved in a separate folder.
        gradient_type (str): The type of gradients to collect. [adam | sign | sgd]
        adam_optimizer_state (dict): The optimizer state of adam optimizers. If None, the gradients will be collected without considering Adam optimization states. 
        max_samples (int, optional): The maximum number of samples to collect. Defaults to None.
    """

    model_id = 0  # model_id is used to draft the random seed for the projectors
    block_size = 128  # fixed block size for the projectors (block-wise matmul for random projection)
    projector_batch_size = 32  # batch size for the projectors; Must be either 8, 16, or 32 (https://trak.readthedocs.io/en/latest/trak.html#trak.projectors.CudaProjector); default by mengzhou: 16
    torch.random.manual_seed(0)  # set the random seed for torch

    project_interval = 32  # project every [] examples (as bsz default to 1 for `dataloader`); default by mengzhou: 16 (i.e., the same as `projector_batch_size`)
    save_interval = 12000  # save every [] examples (as bsz default to 1 for `dataloader`); default by mengzhou: 160

    def _project(current_full_grads, projected_grads):
        current_full_grads = torch.stack(current_full_grads).to(torch.float16)  # torch.stack will create a new dimension
        # TODO: why converting `full_grads` to float16 instead of bfloat16 or float32?
        # if the size of `full_grads` is not too large and within the scope of float16, then float16 is better than bfloat16;
        # but float32 is always better than float16: should change to float32 if the dtype in `get_info.py` is also upgraded
        for i, projector in enumerate(projectors):
            current_projected_grads = projector.project(
                current_full_grads, model_id=model_id)
            projected_grads[proj_dim[i]].append(current_projected_grads.cpu())

    def _save(projected_grads, output_dirs):
        for dim in proj_dim:
            if len(projected_grads[dim]) == 0:
                continue
            projected_grads[dim] = torch.cat(projected_grads[dim])

            output_dir = output_dirs[dim]
            outfile = os.path.join(output_dir, f"grads-{count}.pt") # grads-{count}.pt is saved in the same order as original `dataset` (dataloader)
            torch.save(projected_grads[dim], outfile)
            print(
                f"Saving {outfile}, {projected_grads[dim].shape}", flush=True)
            projected_grads[dim] = []   # `full_grads` is cleared in the main loop, while `projected_grads` is cleared inside `_save`

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # prepare optimization states
    if gradient_type == "adam":
        assert adam_optimizer_state is not None
        # first and second moment estimates
        m, v = prepare_optimizer_state(model, adam_optimizer_state, device)

    projector = get_trak_projector(device)
    number_of_params = get_number_of_params(model)

    # never made it work sadly
    # fmodel, params, buffers = make_functional_with_buffers(model)
    # grads_loss = torch.func.grad(get_output, has_aux=False, argnums=1)

    # initialize a projector for each target project dimension
    projectors = []
    for dim in proj_dim:
        proj = projector(grad_dim=number_of_params,
                         proj_dim=dim,
                         seed=0,
                         proj_type=ProjectionType.rademacher,
                         device=device,
                         dtype=dtype,   # NOTE: the dtype for projector is the same as the dtype for model specified in `get_info.py`
                         block_size=block_size,
                         max_batch_size=projector_batch_size)
        projectors.append(proj)

    count = 0

    # set up an output directory for each dimension
    output_dirs = {}
    for dim in proj_dim:
        output_dir_per_dim = os.path.join(output_dir, f"dim{dim}")
        output_dirs[dim] = output_dir_per_dim
        os.makedirs(output_dir_per_dim, exist_ok=True)

    # max index for each dimension -> to resume calculation from where you stop last time
    max_index = min(get_max_saved_index(
        output_dirs[dim], "grads") for dim in proj_dim)
    # max_index = 31520   # to start from an arbitary resume, instead of always from the max index
    # # DEBUG 1
    # print(f"MAX_INDEX: {max_index}")
    # return

    # projected_gradients
    full_grads = []  # full gradients
    projected_grads = {dim: [] for dim in proj_dim}  # projected gradients

    for batch in tqdm(dataloader, total=len(dataloader)):
        prepare_batch(batch)
        count += 1

        if count <= max_index:  # if count != max_index
            print("skipping count", count)
            continue

        if gradient_type == "adam":
            if count == 1:
                print("Using Adam gradients")
            vectorized_grads = obtain_gradients_with_adam(model, batch, m, v)
        elif gradient_type == "sign":
            if count == 1:
                print("Using Sign gradients")
            vectorized_grads = obtain_sign_gradients(model, batch)
        else:
            if count == 1:
                print("Using SGD gradients")
            vectorized_grads = obtain_gradients(model, batch)

        # add the gradients to the full_grads
        full_grads.append(vectorized_grads)
        model.zero_grad()   # clean the forward and backprop trace in `obtain_gradients_with_adam`

        if count % project_interval == 0:
            _project(full_grads, projected_grads)
            full_grads = [] # clean full_grads for storing next interval

        if count % save_interval == 0:
            _save(projected_grads, output_dirs)

        if max_samples is not None and count == max_samples:
            break

    # project and save the final checkpoint
    if len(full_grads) > 0:
        _project(full_grads, projected_grads)
        full_grads = []

    for dim in proj_dim:
        _save(projected_grads, output_dirs)

    torch.cuda.empty_cache()
    for dim in proj_dim:
        output_dir = output_dirs[dim]
        merge_and_normalize_info(output_dir, prefix="grads")
        merge_info(output_dir, prefix="grads")

    print("Finished")


def merge_and_normalize_info(output_dir: str, prefix="reps"):
    """ Merge and normalize the representations and gradients into a single file. """
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order, in order to make sure that in matching.py `influence_score.reshape(influence_score.shape[0], N_SUBTASKS[target_task_name], -1)` correctly expands the third dimension of samples from the same subtask
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = []
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        normalized_data = normalize(data, dim=1)
        merged_data.append(normalized_data)
    merged_data = torch.cat(merged_data, dim=0)

    output_file = os.path.join(output_dir, f"all_orig.pt")
    torch.save(merged_data, output_file)
    print(
        f"Saving the normalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")


def merge_info(output_dir: str, prefix="reps"):
    """ Merge the representations and gradients into a single file without normalization. """
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = []
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        merged_data.append(data)
    merged_data = torch.cat(merged_data, dim=0)

    output_file = os.path.join(output_dir, f"all_unormalized.pt")
    torch.save(merged_data, output_file)
    print(
        f"Saving the unnormalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")


def collect_reps(dataloader: torch.utils.data.DataLoader,
                 model: torch.nn.Module,
                 output_dir: str,
                 max_samples: Optional[int] = None):
    """
    Collects representations from a dataloader using a given model and saves them to the output directory.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader containing the input data.
        model (torch.nn.Module): The model used to compute the representations.
        output_dir (str): The directory where the representations will be saved.
        max_samples (int, optional): The maximum number of samples to collect. Defaults to None.
    """

    all_reps = []
    count = 0
    save_interval = 160  # save every 160 batches

    device = next(model.parameters()).device  # only works for single gpu
    max_index = get_max_saved_index(output_dir, prefix="reps")

    for batch in tqdm(dataloader):
        # batch_size is default to 1, and this for loop preserves the original order of dataloader (dataset)
        count += 1
        if count <= max_index:
            print("skipping count", count)
            continue

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.inference_mode():
            if isinstance(model, RobertaModel):
                reps = model(input_ids=input_ids,
                             attention_mask=attention_mask, output_hidden_states=True, return_dict=True).pooler_output
            else:
                hidden_states = model(input_ids,
                                      labels=input_ids,
                                      attention_mask=attention_mask,
                                      output_hidden_states=True).hidden_states
                ids = torch.arange(len(input_ids), device=input_ids.device)
                pos = attention_mask.sum(dim=1) - 1
                reps = hidden_states[-1][ids, pos]  # last example in a batch -> last token in an example -> last hidden layer for a token

            all_reps.append(reps.cpu())
            if count % save_interval == 0:
                all_reps = torch.cat(all_reps)
                outfile = os.path.join(output_dir, f"reps-{count}.pt")
                torch.save(all_reps, outfile)
                all_reps = []
                print(f"Saving {outfile}")

            if max_samples is not None and count >= max_samples:
                break

    if len(all_reps) > 0:
        all_reps = torch.cat(all_reps)
        outfile = os.path.join(output_dir, f"reps-{count}.pt")
        torch.save(all_reps, outfile)
        print(f"Saving {outfile}")

    torch.cuda.empty_cache()
    merge_and_normalize_info(output_dir, prefix="reps")

    print("Finished")


def get_loss(dataloader: torch.utils.data.DataLoader,
             model: torch.nn.Module,
             output_dir: str,):
    """ Get the loss of the model on the given dataset. """
    total_loss = 0
    total_tokens = 0
    for batch in tqdm(dataloader):
        prepare_batch(batch)
        num_token = (batch["labels"] != -100).sum()
        with torch.inference_mode():
            loss = model(**batch).loss * num_token
        total_loss += loss.item()
        total_tokens += num_token.item()

    print(f"Loss: {total_loss / total_tokens}")
    result = {"num_tokens": total_tokens, "loss": (
        total_loss / total_tokens)}
    with open(os.path.join(output_dir, "loss.txt"), "w") as f:
        f.write(json.dumps(result, indent=4))


def compute_per_sample_loss(logits, labels):
    """
    For logits with shape [B, L, D], return per-sample loss with shape [B], by using `reduction='none'`
    Refer to https://huggingface.co/learn/nlp-course/chapter7/6 for more details

    Args:
        logits (_type_): [B, L, D]
        labels (_type_): [B, L]
    """
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Calculate per-token loss
    loss_fct = CrossEntropyLoss(reduction='none')   # i.e., reduce=False
    loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Resize and average to get per-sample loss
    loss_per_sample = loss_per_token.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
    
    return loss_per_sample


def collect_losses(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    output_dir: str,
    max_samples: Optional[int] = None,
):
    """
    Collect model loss for each training example at a specific model checkpoint;
    Save as a 1-D tensor of shape: [len(dataset)]

    Args:
        dataloader (torch.utils.data.DataLoader): _description_
        model (torch.nn.Module): _description_
        output_dir (str): _description_
        max_samples (Optional[int], optional): _description_. Defaults to None.
    """    
    
    all_losses = [] # loss vector for multiple training points, computed at the same model checkpoint
    count = 0
    batch_size = dataloader.batch_size
    save_interval = 12000    # save every `{save_interval}` examples, i.e., `{save_interval/batch_size}` batches
    
    # device = next(model.parameters()).device  # only works for single gpu
    max_index = get_max_saved_index(output_dir, prefix="losses")
    
    for batch in tqdm(dataloader, total=len(dataloader)):
        count += batch_size
        if count <= max_index:
            print("skipping count", count)
            continue
        
        prepare_batch(batch)
        with torch.inference_mode():
            outputs = model(**batch)
            loss_per_sample = compute_per_sample_loss(outputs.logits, batch["labels"])  # loss_per_sample.shape: [batch_size]
            
            all_losses.append(loss_per_sample.cpu())
            if count % save_interval == 0:
                all_losses = torch.cat(all_losses, dim=0)
                assert all_losses.shape == torch.Size([save_interval])
                outfile = os.path.join(output_dir, f"losses-{count}.pt")
                torch.save(all_losses, outfile)
                all_losses = []
                print(f"Saving {outfile}", flush=True)
            
            if max_samples is not None and count >= max_samples:
                break
        
    if len(all_losses) > 0:
        all_losses = torch.cat(all_losses, dim=0)
        outfile = os.path.join(output_dir, f"losses-{max_samples}.pt")
        torch.save(all_losses, outfile)
        print(f"Saving {outfile}", flush=True)
    
    torch.cuda.empty_cache()
    merge_info(output_dir, prefix="losses")
    
    print("Finished")
