import argparse
import os

import torch

argparser = argparse.ArgumentParser(
    description='Script for calculating & saving Group-level Attribution Matrix (Group AM)')
argparser.add_argument('--gradient_path', type=str, default="{}-ckpt{}",
                       help='The path to the gradient file')
argparser.add_argument('--train_file_names', type=str, nargs='+',
                       help='The list of training dataset names')
argparser.add_argument('--ckpts', type=int, nargs='+',
                       help="The list of checkpoint numbers.")
argparser.add_argument('--checkpoint_weights', type=float, nargs='+',
                       help="The list of checkpoint weights")
argparser.add_argument('--target_task_names', type=str,
                       nargs='+', help="The list of target task names")
argparser.add_argument('--validation_gradient_path', type=str,
                       default="{}-ckpt{}", help='The path to the validation gradient file')
argparser.add_argument('--output_path', type=str, default="../attribution_matrix/ultra_interact",
                       help='The path to the output')
argparser.add_argument("--use_unnormed_grad", default=False, action="store_true", help="Whether to use normed gradients (i.e., cosine similarity) or unnormalized inner product")

args = argparser.parse_args()
print(args, '\n')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_AM_block_mean_and_shape(training_info: torch.Tensor, validation_info: torch.Tensor):
    """Calculate one of the blocks in the final AM.

    Args:
        training_info (torch.Tensor): training info (gradients/representations) stored in a tensor of shape N_TRAIN x N_DIM
        validation_info (torch.Tensor): validation info (gradients/representations) stored in a tensor of shape N_VALIDATION x N_DIM
    """
    AM_block = torch.matmul(
        training_info, validation_info.transpose(0, 1)) # return an N_TRAIN x N_VALIDATION matrix (2D tensor)

    AM_block_mean = torch.mean(AM_block).item()
    AM_block_shape = AM_block.shape

    return AM_block_mean, AM_block_shape


# renormalize the checkpoint weights
if sum(args.checkpoint_weights) != 1:
    s = sum(args.checkpoint_weights)
    args.checkpoint_weights = [i/s for i in args.checkpoint_weights]


# calculate each element of the attribution matrix
attribution_matrix = torch.zeros(len(args.train_file_names), len(args.target_task_names))
print(f"Shape of the final AM: {attribution_matrix.shape}\n")
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

for j, target_task_name in enumerate(args.target_task_names):
    for i, train_file_name in enumerate(args.train_file_names):
        avg_info_score = 0.0

        for k, ckpt in enumerate(args.ckpts):
            validation_path = args.validation_gradient_path.format(target_task_name, ckpt)
            if os.path.isdir(validation_path):
                if not args.use_unnormed_grad:
                    validation_path = os.path.join(validation_path, "all_orig.pt")
                    print(f"Using normed grads at: {validation_path}")
                else:
                    validation_path = os.path.join(validation_path, "all_unormalized.pt")
                    print(f"Using un-normed grads at: {validation_path}")
            validation_info = torch.load(validation_path)
            if not torch.is_tensor(validation_info):
                validation_info = torch.tensor(validation_info)
            validation_info = validation_info.to(device).float()

            gradient_path = args.gradient_path.format(train_file_name, ckpt)
            if os.path.isdir(gradient_path):
                if not args.use_unnormed_grad:
                    gradient_path = os.path.join(gradient_path, "all_orig.pt")
                    print(f"Using normed grads at: {gradient_path}")
                else:
                    gradient_path = os.path.join(gradient_path, "all_unormalized.pt")
                    print(f"Using un-normed grads at: {gradient_path}")
            training_info = torch.load(gradient_path)
            if not torch.is_tensor(training_info):
                training_info = torch.tensor(training_info)
            training_info = training_info.to(device).float()

            # sum influence scores across different epochs (i.e., ckpts), by different ckpt weights
            # N_TRAIN * N_VALIDATION; This is exactly one block/cell of the instance-level AM
            AM_block_mean, AM_block_shape = calculate_AM_block_mean_and_shape(training_info=training_info, validation_info=validation_info)
            group_avg_info = args.checkpoint_weights[k] * AM_block_mean

            print(f"TRAIN: {train_file_name}; EVAL: {target_task_name}; CKPT: {ckpt}")
            print(f"AM block shape: {AM_block_shape}")
            print(f"Group Avg Info: {group_avg_info}")
            print('\n')

            avg_info_score += group_avg_info

        attribution_matrix[i, j] = avg_info_score

filename = "group_AM.pt" if not args.use_unnormed_grad else "group_AM_unnormed_grad.pt"
output_file = os.path.join(args.output_path, filename)
torch.save(attribution_matrix, output_file)
print("Saved group AM to {}".format(output_file))
print(f"Saved AM values:\n{attribution_matrix}")