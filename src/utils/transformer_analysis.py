from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import math
from torch.nn import functional as F
from ..data.dataset import decode

def analyze_cross_attention(model, examples, start_idx=0, step=3, att_head=2, figsize=(15, 12), cmap='Blues', show_diagonal=False, num_examples=9):
    """
    Analyze and visualize cross-attention for multiple examples in a grid.
    
    Args:
        start_idx (int): Starting index of examples from validation set
        step (int): Generation step to analyze
        att_head (int): Attention head to visualize
        cmap (str): Colormap for the attention heatmap
        show_diagonal (bool): Whether to show the diagonal line
        num_examples (int): Number of examples to show (default 9 for 3x3 grid)
    
    Returns:
        list: List of model outputs for each example
    """
    # Calculate grid dimensions
    grid_size = int(num_examples ** 0.5)
    if grid_size * grid_size < num_examples:
        grid_size += 1
    
    # Create figure with subplots and increased spacing
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    if grid_size == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    all_outputs = []
    
    for i in range(num_examples):
        ex_idx = start_idx + i
        if ex_idx >= len(examples['inputs']):
            break
            
        # Get example data
        inputs = examples['inputs'][ex_idx]
        targets = examples['targets'][ex_idx]
        max_new_tokens = targets.shape[0] - 1
        
        # Generate with attention tracking
        outputs = model.generate_with_attention_tracking(
            src=inputs.unsqueeze(0),
            tgt=targets.unsqueeze(0)[:, :-max_new_tokens],
            max_new_tokens=max_new_tokens,
            temperature=1,
            top_k=1
        )
        all_outputs.append(outputs)
        
        # Check if prediction is correct
        is_correct = (outputs["generated_tokens"][0, -max_new_tokens:] == targets[-max_new_tokens:]).all().item()
        
        # Extract cross-attention for visualization
        crs_att = outputs['attention_history'][step]['decoder'][0]['cross_attention'][0, att_head, :, :].squeeze()
        
        # Plot in subplot
        ax = axes[i]
        im = ax.imshow(crs_att.cpu().numpy(), cmap=cmap, aspect='auto')
        ax.set_ylabel('Generated Position', fontsize=9)
        ax.set_xlabel('Input Position', fontsize=9)
        
        # Add correctness indicator to title
        status_symbol = "✓" if is_correct else "✗"
        ax.set_title(f'Ex {ex_idx} {status_symbol}', fontsize=10)
        
        # Reduce tick label size to prevent overlap
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        # Optional diagonal line
        if show_diagonal:
            height, width = crs_att.shape
            ax.plot([1, width-1], [height-2, 0], 'r--', linewidth=1, alpha=0.8)
    
    # Hide unused subplots
    for i in range(num_examples, len(axes)):
        axes[i].set_visible(False)
    
    # Adjust layout with more spacing to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, right=0.9, hspace=0.4, wspace=0.3)
    
    # Add shared colorbar with proper positioning
    cbar = fig.colorbar(im, ax=axes[:num_examples], shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label('Attention Score', fontsize=10)
    
    # Add main title with proper spacing
    fig.suptitle(f'Cross-Attention Head {att_head} at Step {step} Generation', fontsize=16, y=0.95)
    
    plt.show()
    
    return all_outputs

def analyze_decoder_attention(model, examples, start_idx=0, step=10, att_head=0, figsize=(15, 12), cmap='Blues', show_diagonal=False, num_examples=9):
    """
    Analyze and visualize decoder self-attention for multiple examples in a grid.
    
    Args:
        start_idx (int): Starting index of examples from validation set
        step (int): Generation step to analyze
        att_head (int): Attention head to visualize
        cmap (str): Colormap for the attention heatmap
        show_diagonal (bool): Whether to show the diagonal line
        num_examples (int): Number of examples to show (default 9 for 3x3 grid)
        figsize (tuple): Figure size for the plot
    
    Returns:
        list: List of model outputs for each example
    """
    # Calculate grid dimensions
    grid_size = int(num_examples ** 0.5)
    if grid_size * grid_size < num_examples:
        grid_size += 1
    
    # Create figure with subplots and increased spacing
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    if grid_size == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    all_outputs = []
    
    for i in range(num_examples):
        ex_idx = start_idx + i
        if ex_idx >= len(examples['inputs']):
            break
            
        # Get example data
        inputs = examples['inputs'][ex_idx]
        targets = examples['targets'][ex_idx]
        max_new_tokens = targets.shape[0] - 1
        
        # Generate with attention tracking
        outputs = model.generate_with_attention_tracking(
            src=inputs.unsqueeze(0),
            tgt=targets.unsqueeze(0)[:, :-max_new_tokens],
            max_new_tokens=max_new_tokens,
            temperature=1,
            top_k=1
        )
        all_outputs.append(outputs)
        
        # Check if prediction is correct
        is_correct = (outputs["generated_tokens"][0, -max_new_tokens:] == targets[-max_new_tokens:]).all().item()
        
        # Extract decoder self-attention for visualization
        dec_self_att_raw = outputs['attention_history'][step]['decoder'][0]['self_attention']
        
        # Handle different possible tensor shapes
        if dec_self_att_raw.dim() == 4:
            dec_self_att = dec_self_att_raw[0, att_head, :, :]  # Remove batch dim, select head
        elif dec_self_att_raw.dim() == 3:
            dec_self_att = dec_self_att_raw[att_head, :, :]  # Select head if no batch dim
        else:
            print(f"Warning: Unexpected decoder self-attention tensor shape: {dec_self_att_raw.shape}")
            dec_self_att = dec_self_att_raw.squeeze()
        
        # Ensure we have a 2D tensor
        if dec_self_att.dim() == 1:
            dec_self_att = dec_self_att.unsqueeze(0)
        elif dec_self_att.dim() > 2:
            while dec_self_att.dim() > 2 and dec_self_att.size(0) == 1:
                dec_self_att = dec_self_att.squeeze(0)
        
        # Plot in subplot
        ax = axes[i]
        im = ax.imshow(dec_self_att.cpu().numpy(), cmap=cmap, aspect='auto')
        ax.set_ylabel('Query Position', fontsize=9)
        ax.set_xlabel('Key Position', fontsize=9)
        
        # Add correctness indicator to title
        status_symbol = "✓" if is_correct else "✗"
        ax.set_title(f'Ex {ex_idx} {status_symbol}', fontsize=10)
        
        # Reduce tick label size to prevent overlap
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        # Optional diagonal line
        if show_diagonal:
            height, width = dec_self_att.shape
            if height > 1 and width > 1:
                ax.plot([0, width-1], [0, height-1], 'r--', linewidth=1, alpha=0.8)
    
    # Hide unused subplots
    for i in range(num_examples, len(axes)):
        axes[i].set_visible(False)
    
    # Adjust layout with more spacing to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, right=0.9, hspace=0.4, wspace=0.3)
    
    # Add shared colorbar with proper positioning
    cbar = fig.colorbar(im, ax=axes[:num_examples], shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label('Attention Score', fontsize=10)
    
    # Add main title with proper spacing
    fig.suptitle(f'Decoder Self-Attention Head {att_head} at Step {step} Generation', fontsize=16, y=0.95)
    
    plt.show()
    
    return all_outputs

def analyze_cross_attention_all_steps(model, examples, ex_idx=0, att_head=2, cmap='Blues', figsize=(20, 6), show_diagonal=False):
    """
    Visualize cross-attention for all generation steps for a single example.
    Args:
        model: The transformer model
        examples: Dict with 'inputs' and 'targets'
        ex_idx: Index of the example to analyze
        att_head: Attention head to visualize
        cmap: Colormap for the attention heatmap
        figsize: Figure size for the plot
        show_diagonal: Whether to show the diagonal line
    Returns:
        List of cross-attention matrices for each step
    """
    inputs = examples['inputs'][ex_idx]
    targets = examples['targets'][ex_idx]
    max_new_tokens = targets.shape[0] - 1
    outputs = model.generate_with_attention_tracking(
        src=inputs.unsqueeze(0),
        tgt=targets.unsqueeze(0)[:, :-max_new_tokens],
        max_new_tokens=max_new_tokens,
        temperature=1,
        top_k=1
    )
    attention_history = outputs['attention_history']
    num_steps = len(attention_history)
    cross_att_matrices = []
    generated = outputs["generated_tokens"][0][-max_new_tokens:]
    
    # Calculate grid dimensions: 4 subplots per row (including Dyck path plot)
    cols = 5
    total_plots = num_steps + 1  # +1 for Dyck path plot
    rows = (total_plots + cols - 1) // cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle case when there's only one row
    if rows == 1:
        if cols == 1:
            axes = [axes]
        else:
            axes = axes.reshape(1, -1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten() if rows > 1 or cols > 1 else [axes]
    
    # Plot 1: Dyck paths visualization (first plot)
    ax = axes_flat[0]
    
    # Plot input, target, and generated paths
    input_x, input_y = tokens_to_path(inputs.cpu().numpy())
    target_x, target_y = tokens_to_path(targets.cpu().numpy())
    gen_x, gen_y = tokens_to_path(generated.cpu().numpy())
    
    ax.plot(input_x, input_y, 'b-', label='Input', linewidth=2, alpha=0.7)
    ax.plot(target_x, target_y, 'g-', label='Target', linewidth=2, alpha=0.7)
    ax.plot(gen_x, gen_y, 'r--', label='Generated', linewidth=2, alpha=0.7)
    # draw y = x line
    max_coord = max(max(gen_x) if gen_x else 0, max(gen_y) if gen_y else 0)
    ax.plot([0, max_coord], [0, max_coord], 'k--', linewidth=0.5, alpha=0.5)
    ax.set_title(f'Dyck Paths (Ex {ex_idx})', fontsize=9)
    ax.set_xlabel('Position', fontsize=8)
    ax.set_ylabel('Height', fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=7)
    
    # Cross-attention plots (starting from second plot)
    for step in range(num_steps):
        # Extract cross-attention - be very careful about dimensions
        crs_att_raw = attention_history[step]['decoder'][0]['cross_attention']
        
        # Select the specific head and ensure we have the right dimensions
        # Expected shape: [batch=1, head, seq_len, src_len]
        if crs_att_raw.dim() == 4:
            crs_att = crs_att_raw[0, att_head, :, :]  # Remove batch dim, select head
        elif crs_att_raw.dim() == 3:
            crs_att = crs_att_raw[att_head, :, :]  # Select head if no batch dim
        else:
            print(f"Warning: Unexpected attention tensor shape at step {step}: {crs_att_raw.shape}")
            crs_att = crs_att_raw
        
        # Ensure we have a 2D tensor
        if crs_att.dim() == 1:
            # If 1D, reshape to [1, seq_len] 
            crs_att = crs_att.unsqueeze(0)
        elif crs_att.dim() > 2:
            # If more than 2D, squeeze carefully
            while crs_att.dim() > 2 and crs_att.size(0) == 1:
                crs_att = crs_att.squeeze(0)
        
        # Final safety check
        if crs_att.dim() != 2:
            print(f"Error: Cannot reshape attention tensor to 2D at step {step}. Shape: {crs_att.shape}")
            continue
            
        cross_att_matrices.append(crs_att)
        
        # Plot index is step + 1 (since first plot is Dyck paths)
        plot_idx = step + 1
        ax = axes_flat[plot_idx]
        im = ax.imshow(crs_att.cpu().numpy(), cmap=cmap, aspect='auto')
        ax.set_title(f'Step {step}', fontsize=9)
        ax.set_xlabel('Input Pos', fontsize=8)
        
        # Only show y-label on leftmost column (excluding first plot)
        if plot_idx % cols == 1:  # Second column (first attention plot in each row)
            ax.set_ylabel('Gen Pos', fontsize=8)
        else:
            ax.set_ylabel('')
            
        ax.tick_params(axis='both', which='major', labelsize=7)
        
        if show_diagonal:
            height, width = crs_att.shape
            if height > 1 and width > 1:  # Only draw diagonal if we have a proper 2D matrix
                ax.plot([1, width-1], [height-2, 0], 'r--', linewidth=1, alpha=0.8)
    
    # Hide unused subplots
    for i in range(total_plots, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.4, wspace=0.5)
    
    # Add colorbar for attention plots only
    if num_steps > 0:
        cbar = fig.colorbar(im, ax=axes_flat[1:num_steps+1], shrink=0.7, aspect=30, pad=0.02)
        cbar.set_label('Attention Score', fontsize=9)
    
    # Check if prediction is correct
    is_correct = (generated[-max_new_tokens:] == targets[-max_new_tokens:]).all().item()
    accuracy_text = "✓ Correct" if is_correct else "✗ Incorrect"
    fig.suptitle(f'Cross-Attention Head {att_head} for Example {ex_idx} ({accuracy_text})', fontsize=14, y=0.96)
    plt.show()
    
    return cross_att_matrices

def analyze_encoder_attention(model, examples, start_idx=0, step=0, att_head=0, figsize=(15, 12), cmap='Blues', show_diagonal=False, num_examples=9):
    """
    Analyze and visualize encoder self-attention for multiple examples in a grid.
    
    Args:
        start_idx (int): Starting index of examples from validation set
        step (int): Generation step to analyze (encoder attention is computed at step 0)
        att_head (int): Attention head to visualize
        cmap (str): Colormap for the attention heatmap
        show_diagonal (bool): Whether to show the diagonal line
        num_examples (int): Number of examples to show (default 9 for 3x3 grid)
    
    Returns:
        list: List of model outputs for each example
    """
    # Calculate grid dimensions
    grid_size = int(num_examples ** 0.5)
    if grid_size * grid_size < num_examples:
        grid_size += 1
    
    # Create figure with subplots and increased spacing
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    if grid_size == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    all_outputs = []
    
    for i in range(num_examples):
        ex_idx = start_idx + i
        if ex_idx >= len(examples['inputs']):
            break
            
        # Get example data
        inputs = examples['inputs'][ex_idx]
        targets = examples['targets'][ex_idx]
        max_new_tokens = targets.shape[0] - 1
        
        # Generate with attention tracking
        outputs = model.generate_with_attention_tracking(
            src=inputs.unsqueeze(0),
            tgt=targets.unsqueeze(0)[:, :-max_new_tokens],
            max_new_tokens=max_new_tokens,
            temperature=1,
            top_k=1
        )
        all_outputs.append(outputs)
        
        # Check if prediction is correct
        is_correct = (outputs["generated_tokens"][0, -max_new_tokens:] == targets[-max_new_tokens:]).all().item()
        
        # Extract encoder self-attention for visualization
        # Encoder attention is typically computed once at the beginning
        enc_att_raw = outputs['attention_history'][step]['encoder'][0]['self_attention']
        
        # Handle different possible tensor shapes
        if enc_att_raw.dim() == 4:
            enc_att = enc_att_raw[0, att_head, :, :]  # Remove batch dim, select head
        elif enc_att_raw.dim() == 3:
            enc_att = enc_att_raw[att_head, :, :]  # Select head if no batch dim
        else:
            print(f"Warning: Unexpected encoder attention tensor shape: {enc_att_raw.shape}")
            enc_att = enc_att_raw.squeeze()
        
        # Ensure we have a 2D tensor
        if enc_att.dim() == 1:
            enc_att = enc_att.unsqueeze(0)
        elif enc_att.dim() > 2:
            while enc_att.dim() > 2 and enc_att.size(0) == 1:
                enc_att = enc_att.squeeze(0)
        
        # Plot in subplot
        ax = axes[i]
        im = ax.imshow(enc_att.cpu().numpy(), cmap=cmap, aspect='auto')
        ax.set_ylabel('Query Position', fontsize=9)
        ax.set_xlabel('Key Position', fontsize=9)
        
        # Add correctness indicator to title
        status_symbol = "✓" if is_correct else "✗"
        ax.set_title(f'Ex {ex_idx} {status_symbol}', fontsize=10)
        
        # Reduce tick label size to prevent overlap
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        # Optional diagonal line
        if show_diagonal:
            height, width = enc_att.shape
            if height > 1 and width > 1:
                ax.plot([0, width-1], [0, height-1], 'r--', linewidth=1, alpha=0.8)
    
    # Hide unused subplots
    for i in range(num_examples, len(axes)):
        axes[i].set_visible(False)
    
    # Adjust layout with more spacing to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, right=0.9, hspace=0.4, wspace=0.3)
    
    # Add shared colorbar with proper positioning
    cbar = fig.colorbar(im, ax=axes[:num_examples], shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label('Attention Score', fontsize=10)
    
    # Add main title with proper spacing
    fig.suptitle(f'Encoder Self-Attention Head {att_head} at Step {step}', fontsize=16, y=0.95)
    
    plt.show()
    
    return all_outputs

# Convert tokens to Dyck path coordinates for visualization
def tokens_to_path(tokens):
    """Convert tokens to path coordinates for plotting. North step is represented by a 1 and an East step is represented by a 0."""
    x_coords = [0]
    y_coords = [0]

    for i, token in enumerate(tokens):
        if token == 2:  # 0 in Dyck path (right step)
            y_coords.append(y_coords[-1])
            x_coords.append(x_coords[-1] + 1)
        elif token == 3:  # 1 in Dyck path (up step)
            y_coords.append(y_coords[-1] + 1)
            x_coords.append(x_coords[-1])
        else:  # Other tokens (BOS, EOS, etc.)
            # Skip these tokens entirely - don't add to path
            continue

    return x_coords, y_coords

def attention_example(model, examples, ex_idx=0, step=10, cross_att_head=0, encoder_att_head=0, decoder_att_head=0, figsize=(16, 12), cmap='Blues'):
    """
    Comprehensive attention analysis showing 4 visualizations in a 2x2 grid:
    1. Dyck paths (input, target, generated)
    2. Cross-attention
    3. Encoder self-attention  
    4. Decoder self-attention
    
    Args:
        model: The transformer model
        examples: Dict with 'inputs' and 'targets'
        ex_idx: Index of the example to analyze
        step: Generation step to analyze for decoder attentions
        cross_att_head: Attention head to visualize for cross-attention
        encoder_att_head: Attention head to visualize for encoder self-attention
        decoder_att_head: Attention head to visualize for decoder self-attention
        figsize: Figure size for the entire plot
        cmap: Colormap for attention heatmaps
    """
    # Get example data
    inputs = examples['inputs'][ex_idx]
    targets = examples['targets'][ex_idx]
    max_new_tokens = targets.shape[0] - 1
    
    # Generate with attention tracking
    if model.architecture == "encoder_decoder":
        outputs = model.generate_with_attention_tracking(
            src=inputs.unsqueeze(0),
            tgt=targets.unsqueeze(0)[:, :-max_new_tokens],
            max_new_tokens=max_new_tokens,
            temperature=1,
            top_k=1
        )
    else:
        full_sequence = torch.cat([inputs, targets], dim=0)
        outputs = model.generate_with_attention_tracking(
            src=None,
            tgt=full_sequence.unsqueeze(0)[:, :-max_new_tokens],
            max_new_tokens=max_new_tokens,
            temperature=1,
            top_k=1
        )

    generated = outputs["generated_tokens"][0][-max_new_tokens:]
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Dyck paths visualization (top-left)
    ax1 = axes[0, 0]
        
    # Plot input, target, and generated paths
    input_x, input_y = tokens_to_path(inputs.cpu().numpy())
    target_x, target_y = tokens_to_path(targets.cpu().numpy())
    gen_x, gen_y = tokens_to_path(generated.cpu().numpy())
    
    ax1.plot(input_x, input_y, 'b-', label='Input', linewidth=2, alpha=0.7)
    ax1.plot(target_x, target_y, 'g-', label='Target', linewidth=2, alpha=0.7)
    ax1.plot(gen_x, gen_y, 'r--', label='Generated', linewidth=2, alpha=0.7)
    # draw y = x line
    ax1.plot([0, max(gen_x)], [0, max(gen_y)], 'k--', linewidth=0.5, alpha=0.5)
    ax1.set_title(f'Dyck Paths (Example {ex_idx})', fontsize=12)
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Height')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cross-attention (top-right)
    ax2 = axes[0, 1]
    try:
        crs_att_raw = outputs['attention_history'][step]['decoder'][0]['cross_attention']
        if crs_att_raw.dim() == 4:
            crs_att = crs_att_raw[0, cross_att_head, :, :]
        elif crs_att_raw.dim() == 3:
            crs_att = crs_att_raw[cross_att_head, :, :]
        else:
            crs_att = crs_att_raw.squeeze()
        
        if crs_att.dim() == 1:
            crs_att = crs_att.unsqueeze(0)
        elif crs_att.dim() > 2:
            while crs_att.dim() > 2 and crs_att.size(0) == 1:
                crs_att = crs_att.squeeze(0)
    except KeyError:
        print("Warning: Cross-attention data not found. Using default empty matrix.")
        crs_att = torch.zeros((1, 1))  # Default empty attention matrix

    im2 = ax2.imshow(crs_att.cpu().numpy(), cmap=cmap, aspect='auto')
    ax2.set_title(f'Cross-Attention (Step {step}, Head {cross_att_head})', fontsize=12)
    ax2.set_xlabel('Input Position')
    ax2.set_ylabel('Generated Position')
    
    # 3. Encoder self-attention (bottom-left)
    ax3 = axes[1, 0]
    try:
        enc_att_raw = outputs['attention_history'][0]['encoder'][0]['self_attention']  # Encoder computed at step 0
        if enc_att_raw.dim() == 4:
            enc_att = enc_att_raw[0, encoder_att_head, :, :]
        elif enc_att_raw.dim() == 3:
            enc_att = enc_att_raw[encoder_att_head, :, :]
        else:
            enc_att = enc_att_raw.squeeze()

        if enc_att.dim() == 1:
            enc_att = enc_att.unsqueeze(0)
        elif enc_att.dim() > 2:
            while enc_att.dim() > 2 and enc_att.size(0) == 1:
                enc_att = enc_att.squeeze(0)
    except KeyError:
        print("Warning: Encoder self-attention data not found. Using default empty matrix.")
        enc_att = torch.zeros((1, 1))  # Default empty attention matrix  

    im3 = ax3.imshow(enc_att.cpu().numpy(), cmap=cmap, aspect='auto')
    ax3.set_title(f'Encoder Self-Attention (Head {encoder_att_head})', fontsize=12)
    ax3.set_xlabel('Key Position')
    ax3.set_ylabel('Query Position')
    
    # 4. Decoder self-attention (bottom-right)
    ax4 = axes[1, 1]
    try:
        dec_self_att_raw = outputs['attention_history'][step]['decoder'][0]['self_attention']
        if dec_self_att_raw.dim() == 4:
            dec_self_att = dec_self_att_raw[0, decoder_att_head, :, :]
        elif dec_self_att_raw.dim() == 3:
            dec_self_att = dec_self_att_raw[decoder_att_head, :, :]
        else:
            dec_self_att = dec_self_att_raw.squeeze()
        
        if dec_self_att.dim() == 1:
            dec_self_att = dec_self_att.unsqueeze(0)
        elif dec_self_att.dim() > 2:
            while dec_self_att.dim() > 2 and dec_self_att.size(0) == 1:
                dec_self_att = dec_self_att.squeeze(0)
    except KeyError:
        print("Warning: Decoder self-attention data not found. Using default empty matrix.")
        dec_self_att = torch.zeros((1, 1))  # Default empty attention matrix

    im4 = ax4.imshow(dec_self_att.cpu().numpy(), cmap=cmap, aspect='auto')
    ax4.set_title(f'Decoder Self-Attention (Step {step}, Head {decoder_att_head})', fontsize=12)
    ax4.set_xlabel('Key Position')
    ax4.set_ylabel('Query Position')
    
    # Add colorbars for attention plots
    plt.colorbar(im2, ax=ax2, shrink=0.8, aspect=20)
    plt.colorbar(im3, ax=ax3, shrink=0.8, aspect=20)
    plt.colorbar(im4, ax=ax4, shrink=0.8, aspect=20)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.3)
    
    # Add main title
    is_correct = (generated[-max_new_tokens:] == targets[-max_new_tokens:]).all().item()
    accuracy_text = "✓ Prediction Correct" if is_correct else "✗ Prediction Incorrect"
    fig.suptitle(f'Attention Analysis - Example {ex_idx} ({accuracy_text})', fontsize=16, y=0.97)
    
    plt.show()
    
    # Return attention matrices and metadata
    return {
        'cross_attention': crs_att,
        'encoder_attention': enc_att,
        'decoder_self_attention': dec_self_att,
        'generated_tokens': generated,
        'is_correct': is_correct,
        'input_tokens': inputs,
        'target_tokens': targets
    }

def analyze_embeddings_pca(model, dataset, figsize=(10, 8), alpha=0.5, fontsize=10):
    """
    Analyze and visualize model embeddings using PCA.
    
    Args:
        model: The transformer model
        dataset: Dataset object containing the dictionary
        figsize: Figure size for the plot
        alpha: Transparency for scatter points
        fontsize: Font size for annotations
    
    Returns:
        dict: Dictionary containing PCA results and embeddings
    """
    
    # Get embedding weights from model
    emb = model._embedding_weights()
    
    # Initialize PCA
    pca = PCA(n_components=2)
    
    # Transform embeddings to 2D
    src_emb = pca.fit_transform(emb['src_embedding'].detach().cpu().numpy())
    tgt_emb = pca.fit_transform(emb['tgt_embedding'].detach().cpu().numpy())
    
    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot embeddings
    scatter1 = ax.scatter(src_emb[:, 0], src_emb[:, 1], label='Encoder Embedding', alpha=alpha)
    scatter2 = ax.scatter(tgt_emb[:, 0], tgt_emb[:, 1], label='Decoder Embedding', alpha=alpha)

    ax.set_title('PCA of Encoder and Decoder Embeddings', fontsize=14)
    ax.set_xlabel(f'PC1', fontsize=fontsize)
    ax.set_ylabel(f'PC2', fontsize=fontsize)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add token labels
    dictionary = dataset.dictionary
    
    # Label target embeddings
    for i, (x, y) in enumerate(tgt_emb):
        ax.annotate(f'{dictionary[i]}', (x, y), xytext=(5, 5), textcoords='offset points', 
                   fontsize=fontsize, ha='left', va='bottom', color='orange')
    
    # Label source embeddings
    for i, (x, y) in enumerate(src_emb):
        ax.annotate(f'{dictionary[i]}', (x, y), xytext=(5, 5), textcoords='offset points', 
                   fontsize=fontsize, ha='left', va='bottom', color='blue')
    
    plt.tight_layout()
    plt.show()
    
    # Return results
    return {
        'pca': pca,
        'src_embeddings_2d': src_emb,
        'tgt_embeddings_2d': tgt_emb,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'total_variance_explained': pca.explained_variance_ratio_.sum()
    }

def analyze_positional_embeddings_pca(model, module = "encoder", n_components=28, figsize = (12,10), plot_labels = True):
    """
    Analyze and visualize positional embeddings using PCA.

    Args:
        model: The transformer model
        module: Which module to analyze ("source" or "target")
        figsize: Figure size for the plot

    Returns:
        dict: Dictionary containing PCA results
    """
    # Extract positional encoding weights
    if module == "encoder":
        assert model.src_pos_encoding is not None, "Ensure your model has an encoder."
        pos_encoding_weights = model.src_pos_encoding.pe.weight.data.cpu().numpy()
    if module == "decoder":
        assert model.tgt_pos_encoding is not None, "Ensure your model has a decoder."
        pos_encoding_weights = model.tgt_pos_encoding.pe.weight.data.cpu().numpy()

    # Perform PCA with multiple components
    n_components = min(n_components, pos_encoding_weights.shape[0], pos_encoding_weights.shape[1])
    pca = PCA(n_components=n_components)
    pos_pca_full = pca.fit_transform(pos_encoding_weights)

    # Also do 2D PCA for visualization
    pca_2d = PCA(n_components=2)
    pos_pca_2d = pca_2d.fit_transform(pos_encoding_weights)

    # Create comprehensive PCA visualization
    fig = plt.figure(figsize=figsize)
    fig.suptitle(f"{module} positional embeddings PCA", fontsize=16, y=0.98)

    # Plot 1: 2D PCA scatter plot
    plt.subplot(2, 3, 1)
    scatter = plt.scatter(pos_pca_2d[:, 0], pos_pca_2d[:, 1], c=range(len(pos_pca_2d)), cmap='viridis', alpha=0.8, s=50)
    plt.colorbar(scatter, label='Position Index')
    plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA: Positions in 2D Space', pad=20)
    plt.grid(True, alpha=0.3)

    # Add position labels for small datasets
    if plot_labels:
        for i, (x, y) in enumerate(pos_pca_2d):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

    # Plot 2: Explained variance ratio
    plt.subplot(2, 3, 2)
    plt.plot(range(1, n_components + 1), pca.explained_variance_ratio_, 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Component', pad=20)
    plt.grid(True, alpha=0.3)

    # Plot 3: Cumulative explained variance
    plt.subplot(2, 3, 3)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, n_components + 1), cumulative_variance, 'ro-')
    plt.axhline(y=0.95, color='g', linestyle='--', alpha=0.7, label='95% variance')
    plt.axhline(y=0.99, color='r', linestyle='--', alpha=0.7, label='99% variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance', pad=20)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 4: Position trajectory in PC1-PC2 space
    plt.subplot(2, 3, 4)
    plt.plot(pos_pca_2d[:, 0], pos_pca_2d[:, 1], 'o-', alpha=0.7, markersize=4)
    plt.scatter(pos_pca_2d[0, 0], pos_pca_2d[0, 1], c='red', s=100, marker='s', label='Start (pos 0)')
    plt.scatter(pos_pca_2d[-1, 0], pos_pca_2d[-1, 1], c='blue', s=100, marker='^', label=f'End (pos {len(pos_pca_2d)-1})')
    plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%})')
    plt.title('Position Trajectory in PCA Space', pad=20)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add position labels for small datasets
    if plot_labels:
        for i, (x, y) in enumerate(pos_pca_2d):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

    # Plot 5: PC1 vs position index
    plt.subplot(2, 3, 5)
    plt.plot(range(len(pos_pca_2d)), pos_pca_2d[:, 0], 'o-', alpha=0.7)
    plt.xlabel('Position Index')
    plt.ylabel(f'PC1 Score ({pca_2d.explained_variance_ratio_[0]:.2%} var)')
    plt.title('PC1 Scores vs Position', pad=20)
    plt.grid(True, alpha=0.3)

    # Plot 6: PC2 vs position index
    plt.subplot(2, 3, 6)
    plt.plot(range(len(pos_pca_2d)), pos_pca_2d[:, 1], 'o-', alpha=0.7, color='orange')
    plt.xlabel('Position Index')
    plt.ylabel(f'PC2 Score ({pca_2d.explained_variance_ratio_[1]:.2%} var)')
    plt.title('PC2 Scores vs Position', pad=20)
    plt.grid(True, alpha=0.3)

    # Add proper spacing between subplots and adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.12, hspace=0.45, wspace=0.35)
    plt.show()

    return {
        'pos_pca_full': pos_pca_full,
        'pos_pca_2d': pos_pca_2d,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'total_variance_explained': pca.explained_variance_ratio_.sum()
    }
    
def analyze_positional_embeddings_similarity(model, module = "encoder", figsize = (10,10)):
    # Extract positional encoding weights
    if module == "encoder":
        assert model.src_pos_encoding is not None, "Ensure your model has an encoder."
        pos_encoding_weights = model.src_pos_encoding.pe.weight.data.cpu().numpy()
    if module == "decoder":
        assert model.tgt_pos_encoding is not None, "Ensure your model has a decoder."
        pos_encoding_weights = model.tgt_pos_encoding.pe.weight.data.cpu().numpy()

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(pos_encoding_weights)

    # Create comprehensive similarity analysis
    fig = plt.figure(figsize=figsize)
    fig.suptitle(f"{module} positional embeddings similarity", fontsize=16, y=0.98)

    # Plot 1: Full similarity matrix
    plt.subplot(2, 2, 1)
    im1 = plt.imshow(similarity_matrix, cmap='RdYlBu_r', interpolation='nearest', vmin=-1, vmax=1)
    plt.colorbar(im1, label='Cosine Similarity')
    plt.xlabel('Position')
    plt.ylabel('Position')
    plt.title('Complete Similarity Matrix', pad=20)

    # Plot 2: Adjacent position similarities
    plt.subplot(2, 2, 2)
    adjacent_similarities = [similarity_matrix[i, i+1] for i in range(len(similarity_matrix)-1)]
    plt.plot(adjacent_similarities, 'o-', alpha=0.7)
    plt.xlabel('Position')
    plt.ylabel('Similarity with Next Position')
    plt.title('Adjacent Position Similarities', pad=20)
    plt.grid(True, alpha=0.3)

    # Plot 3: Distance-based similarity analysis
    plt.subplot(2, 2, 3)
    distances = []
    similarities = []
    for i in range(len(similarity_matrix)):
        for j in range(i+1, len(similarity_matrix)):
            distance = abs(i - j)
            similarity = similarity_matrix[i, j]
            distances.append(distance)
            similarities.append(similarity)

    # Group by distance and compute mean similarity
    unique_distances = sorted(set(distances))
    mean_similarities = []
    for d in unique_distances:
        sims_at_distance = [similarities[k] for k, dist in enumerate(distances) if dist == d]
        mean_similarities.append(np.mean(sims_at_distance))

    plt.plot(unique_distances, mean_similarities, 'o-', alpha=0.7)
    plt.xlabel('Position Distance')
    plt.ylabel('Average Similarity')
    plt.title('Similarity vs Position Distance', pad=20)
    plt.grid(True, alpha=0.3)

    # Add proper spacing between subplots and adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(left=0.1, right=0.95, top=0.90, bottom=0.1, hspace=0.4, wspace=0.3)
    plt.show()

    return similarity_matrix


# Create Fourier basis functions
def create_fourier_basis(n_positions, n_components):
    """Create Fourier basis functions for projection"""
    basis = np.zeros((n_positions, n_components))
    
    # DC component
    basis[:, 0] = 1.0 / np.sqrt(n_positions)
    
    # Sine and cosine components
    for k in range(1, n_components // 2 + 1):
        if 2*k-1 < n_components:
            # Cosine component
            basis[:, 2*k-1] = np.sqrt(2/n_positions) * np.cos(2*np.pi*k*np.arange(n_positions)/n_positions)
        if 2*k < n_components:
            # Sine component  
            basis[:, 2*k] = np.sqrt(2/n_positions) * np.sin(2*np.pi*k*np.arange(n_positions)/n_positions)
    
    return basis
        
def analyze_positional_embeddings_fourier(model, module = "encoder", figsize = (20,15)):
    """
    Analyze and visualize positional embeddings using Fourier basis projection.

    Args: 
        model: The transformer model
        module: Which module to analyze ("encoder" or "decoder")
        figsize: Figure size for the entire plot
    """

    # Extract positional encoding weights
    if module == "encoder":
        assert model.src_pos_encoding is not None, "Ensure your model has an encoder."
        pos_encoding_weights = model.src_pos_encoding.pe.weight.data.cpu().numpy()
    if module == "decoder":
        assert model.tgt_pos_encoding is not None, "Ensure your model has a decoder."
        pos_encoding_weights = model.tgt_pos_encoding.pe.weight.data.cpu().numpy()

    n_positions, d_model = pos_encoding_weights.shape

    # Number of Fourier components to use
    n_fourier_components = n_positions
    fourier_basis = create_fourier_basis(n_positions, n_fourier_components)

    # Project each embedding dimension onto Fourier basis
    fourier_projections = np.dot(fourier_basis.T, pos_encoding_weights)  # Shape: (n_fourier_components, d_model)

    # Compute power spectrum for each embedding dimension
    power_spectra = fourier_projections ** 2

    # Create comprehensive visualization
    fig = plt.figure(figsize=figsize)
    fig.suptitle(f"{module} Positional Embeddings Fourier Analysis", fontsize=16, y=0.98)

    # Plot 1: Original positional encoding heatmap
    plt.subplot(2, 3, 1)
    im = plt.imshow(pos_encoding_weights.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    plt.colorbar(im, label='Weight Value')
    plt.xlabel('Position')
    plt.ylabel('Embedding Dimension')
    plt.title('Original Positional Encoding', pad=20)

    # Plot 2: Fourier projections heatmap
    plt.subplot(2, 3, 2)
    im = plt.imshow(fourier_projections, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    plt.colorbar(im, label='Projection Coefficient')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Fourier Component')
    plt.title('Fourier Projections', pad=20)

    # Plot 3: Power spectrum for each embedding dimension
    plt.subplot(2, 3, 3)
    im = plt.imshow(power_spectra, aspect='auto', cmap='hot', interpolation='nearest')
    plt.colorbar(im, label='Power')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Fourier Component')
    plt.title('Power Spectrum (All Dimensions)', pad=20)

    # Plot 4: Reconstruction using top Fourier components
    plt.subplot(2, 3, 4)
    n_recon_components = min(6, n_fourier_components)
    reconstructed = np.dot(fourier_basis[:, :n_recon_components], fourier_projections[:n_recon_components, :])
    im = plt.imshow(reconstructed.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    plt.colorbar(im, label='Weight Value')
    plt.xlabel('Position')
    plt.ylabel('Embedding Dimension')
    plt.title(f'Reconstruction (Top {n_recon_components} Components)', pad=20)

    # Plot 5: Reconstruction error
    plt.subplot(2, 3, 5)
    reconstruction_errors = []
    component_counts = range(1, min(n_fourier_components, n_positions) + 1)
    for n_comp in component_counts:
        recon = np.dot(fourier_basis[:, :n_comp], fourier_projections[:n_comp, :])
        error = np.mean((pos_encoding_weights - recon) ** 2)
        reconstruction_errors.append(error)

    plt.plot(component_counts, reconstruction_errors, 'o-', alpha=0.7)
    plt.xlabel('Number of Fourier Components')
    plt.ylabel('Mean Squared Error')
    plt.title('Reconstruction Error', pad=20)
    plt.grid(True, alpha=0.3)

    # Plot 6: Fourier basis functions (first few)
    plt.subplot(2, 3, 6)
    for i in range(min(6, n_fourier_components)):
        plt.plot(fourier_basis[:, i], label=f'Component {i}', alpha=0.7)
    plt.xlabel('Position')
    plt.ylabel('Basis Function Value')
    plt.title('Fourier Basis Functions (First Few)', pad=20)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add proper spacing between subplots and adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.12, hspace=0.45, wspace=0.35)
    plt.show()

    return {
        "fourier_basis": fourier_basis,
        "fourier_projections": fourier_projections,
        "reconstruction_errors": reconstruction_errors,
}


def zero_cross_attention_to_token(model, dataset, criterion, token_id, device, n_batches = 16, eval_steps = list(range(1,10))):
    '''
    Ablate cross-attention weights for a specific input token by setting them to zero.
    Args:
        model: The transformer model
        dataset: The dataset object with eval_dataloader
        criterion: the loss function
        token_id: The token ID to ablate attention to
        n_batches: Number of batches to evaluate
        eval_steps: List of generation steps to evaluate accuracy on

    Returns:
        accuracies: dict with accuracy at each specified step
        avg_loss: average loss over the examples 
    '''

    assert model.architecture == "encoder_decoder", "Cross-attention ablation only applies to encoder-decoder models."
    token_id = torch.tensor(token_id)
    
    def modified_cross_attention_forward(self, q, k=None, v=None, mask=None):
        """Modified cross attention that zeros out attention to specific token"""
        if k is None and v is None:
            k = v = q
        elif v is None:
            v = k
            
        batch_size = q.size(0)
        seq_len = q.size(1)
        src_len = k.size(1)
        
        # Original attention computation
        q_proj = self.q_linear(q)
        k_proj = self.k_linear(k) 
        v_proj = self.v_linear(v)
        
        # Reshape for multi-head attention
        q_proj = q_proj.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_proj = k_proj.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_proj = v_proj.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q_proj, k_proj.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Apply original mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
            
        # Apply causal mask if this is a causal attention
        if self.is_causal and mask is None:
            causal_mask = torch.ones_like(scores).triu(diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
            
        # Zero out attention to the specific token (token_id)
        # We need to identify positions where the source tokens equal token_id
        # This requires access to the source tokens, which we'll store globally
        if hasattr(self, '_ablation_src_tokens'):
            src_tokens = self._ablation_src_tokens
            # Create mask for positions with token_id: [batch, 1, 1, src_len]
            token_mask = (src_tokens == token_id).unsqueeze(1).unsqueeze(1)
            # Expand to match attention scores shape: [batch, num_heads, seq_len, src_len]
            token_mask = token_mask.expand(-1, self.num_heads, seq_len, -1)
            scores = scores.masked_fill(token_mask, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        self.last_attn_scores = attn_weights.detach()
        
        # Apply attention to values
        attn_output = torch.matmul(self.attn_dropout(attn_weights), v_proj)
        
        # Reshape back and final projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_linear(attn_output)
        
        return output
    
    # Store original forward methods
    original_forwards = {}
    
    # Modify cross attention layers based on actual model structure
    decoder_layers = model.decoder # decoder is a nn.ModuleList of decoder layers
    if decoder_layers is not None:
        for i, layer in enumerate(decoder_layers):
            cross_attn = layer.cross_attn
            original_forwards[f'decoder_layer_{i}_cross_attn'] = cross_attn.forward
            cross_attn.forward = modified_cross_attention_forward.__get__(cross_attn, cross_attn.__class__) # bind ablated forward method
    
    # Evaluation loop
    total_correct_by_steps = {step: 0 for step in eval_steps}
    total_samples = 0
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataset.eval_dataloader):
        if batch_idx >= n_batches:
            break
            
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Store source tokens for the ablation
        if decoder_layers is not None:
            for i, layer in enumerate(decoder_layers):
                cross_attn = layer.cross_attn
                if cross_attn is not None:
                    cross_attn._ablation_src_tokens = inputs
        
        # Forward pass
        outputs = model(src=inputs, tgt=targets[:, :-1])
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets[:, 1:].contiguous().view(-1))
        
        # Generate for accuracy
        generated = model.generate(src=inputs, tgt=torch.full((inputs.size(0), 1), 0).to(device), max_new_tokens=targets.size(1))
        
        for step in eval_steps:
            correct = (generated[:, :targets.size(1)] == targets)[:,:step].all(dim=1).sum()
            total_correct_by_steps[step] += correct.item()
        total_samples += inputs.size(0)
        total_loss += loss.item()

        # Restore original forward methods
        if decoder_layers is not None:
            for i, layer in enumerate(decoder_layers):
                cross_attn = layer.cross_attn
                key = f'decoder_layer_{i}_cross_attn'
                cross_attn.forward = original_forwards[key] # restore original forward method
                delattr(cross_attn, '_ablation_src_tokens') # clean up the attribute

    accuracies = {step: total_correct_by_steps[step] / total_samples for step in eval_steps}
    avg_loss = total_loss / total_samples

    return accuracies, avg_loss


def compute_levels(sequence, dictionary):
    """
    Compute the level (number of 1s - number of 0s) at each position in the sequence.
    
    Args:
        sequence: tensor of token ids
        dictionary: mapping from tokens to symbols
        
    Returns:
        levels: list of levels at each position
    """
    levels = []
    current_level = 0
    
    for token_id in sequence:
        # Convert token id to symbol
        symbol = dictionary.get(token_id.item(), str(token_id.item()))
        
        # Update level based on symbol (assuming '(' increases level, ')' decreases)
        if symbol == 1:
            current_level += 1
        elif symbol == 0:
            current_level -= 1
        
        levels.append(current_level)
    
    return levels


def extract_encoder_states(model, device, dataloader, dictionary, max_samples=1000):
    """
    Extract encoder hidden states and corresponding levels from the dataset.
    
    Args:
        model: the transformer model
        dataloader: data loader containing input sequences
        dictionary: token to symbol mapping
        max_samples: maximum number of samples to process
        device: computation device (cpu or cuda)
        
    Returns:
        all_states: tensor of shape (total_positions, hidden_dim)
        all_positions: tensor of position indices
    """
    model.eval()
    all_states = []
    all_positions = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= max_samples:
                break
                
            inputs, targets = batch
            inputs = inputs.to(device)
            
            # Get encoder states
            if model.architecture != 'decoder_only':
                # Forward through encoder only
                src_emb = model.src_embedding(inputs)
                src_emb = model.src_pos_encoding(src_emb)
                
                # Pass through encoder layers
                encoder_output = src_emb
                for layer in model.encoder:
                    encoder_output = layer(encoder_output)
                
                # encoder_output shape: (batch_size, seq_len, hidden_dim)
                batch_size, seq_len, hidden_dim = encoder_output.shape
                
                # Process each sequence in the batch
                for i in range(batch_size):
                    if sample_count >= max_samples:
                        break
                        
                    sequence = inputs[i]
                    states = encoder_output[i]  # (seq_len, hidden_dim)
                    
                    # Compute levels for this sequence
                    levels = compute_levels(sequence, dictionary)
                    
                    # Store states, levels, and positions
                    for pos in range(seq_len):
                        all_states.append(states[pos])
                        all_positions.append(pos)
                    
                    sample_count += 1
            else:
                print("This probing analysis is designed for encoder-decoder models.")
                return None, None, None
    
    # Convert to tensors
    all_states = torch.stack(all_states)  # (total_positions, hidden_dim)
    all_positions = torch.tensor(all_positions, dtype=torch.long)  # (total_positions,)
    
    return all_states

def extract_levels(model, dataloader, dictionary, device, max_samples=1000):
    """
    Extract encoder hidden states and corresponding levels from the dataset.
    
    Args:
        model: the transformer model
        dataloader: data loader containing input sequences
        dictionary: token to symbol mapping
        max_samples: maximum number of samples to process
        device: computation device (cpu or cuda)
        
    Returns:
        all_levels: tensor of shape (total_positions,)
    """
    model.eval()
    all_states = []
    all_levels = []
    all_positions = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= max_samples:
                break
                
            inputs, _ = batch
            inputs = inputs.to(device)
            
            # encoder_output shape: (batch_size, seq_len, hidden_dim)
            batch_size, seq_len = inputs.shape

            # Process each sequence in the batch
            for i in range(batch_size):
                if sample_count >= max_samples:
                    break
                    
                sequence = inputs[i]
                
                # Compute levels for this sequence
                levels = compute_levels(sequence, dictionary)
                
                # Store states, levels, and positions
                for pos in range(seq_len):
                    all_levels.append(levels[pos])
                
                sample_count += 1
                
    all_levels = torch.tensor(all_levels, dtype=torch.long)  # (total_positions,)

    return all_levels

def extract_encoder_states_and_levels(model, dataloader, dictionary, device, max_samples=1000):
    """
    Extract encoder hidden states and corresponding levels from the dataset.
    
    Args:
        model: the transformer model
        dataloader: data loader containing input sequences
        dictionary: token to symbol mapping
        max_samples: maximum number of samples to process
        
    Returns:
        all_states: tensor of shape (total_positions, hidden_dim)
        all_levels: tensor of shape (total_positions,)
        all_positions: tensor of position indices
    """
    model.eval()
    all_states = []
    all_levels = []
    all_positions = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= max_samples:
                break
                
            inputs, targets = batch
            inputs = inputs.to(device)
            
            # Get encoder states
            if model.architecture == 'encoder_decoder':
                # Forward through encoder only
                src_emb = model.src_embedding(inputs)
                src_emb = model.src_pos_encoding(src_emb)
                
                # Pass through encoder layers
                encoder_output = src_emb
                for layer in model.encoder:
                    encoder_output = layer(encoder_output)
                
                # encoder_output shape: (batch_size, seq_len, hidden_dim)
                batch_size, seq_len, hidden_dim = encoder_output.shape
                
                # Process each sequence in the batch
                for i in range(batch_size):
                    if sample_count >= max_samples:
                        break
                        
                    sequence = inputs[i]
                    states = encoder_output[i]  # (seq_len, hidden_dim)
                    
                    # Compute levels for this sequence
                    levels = compute_levels(sequence, dictionary)
                    
                    # Store states, levels, and positions
                    for pos in range(seq_len):
                        all_states.append(states[pos])
                        all_levels.append(levels[pos])
                        all_positions.append(pos)
                    
                    sample_count += 1
            else:
                print("This probing analysis is designed for encoder-decoder models.")
                return None, None, None
    
    # Convert to tensors
    all_states = torch.stack(all_states)  # (total_positions, hidden_dim)
    all_levels = torch.tensor(all_levels, dtype=torch.long)  # (total_positions,)
    all_positions = torch.tensor(all_positions, dtype=torch.long)  # (total_positions,)
    
    return all_states, all_levels, all_positions