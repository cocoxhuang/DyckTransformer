from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

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

def analyze_cross_attention_all_steps(model, examples, ex_idx=0, att_head=2, cmap='Blues', figsize=(16, 4), show_diagonal=False):
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
    
    # Calculate grid dimensions: 3 subplots per row
    cols = 3
    rows = (num_steps + cols - 1) // cols  # Ceiling division
    
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
        
        ax = axes_flat[step]
        im = ax.imshow(crs_att.cpu().numpy(), cmap=cmap, aspect='auto')
        ax.set_title(f'Step {step}', fontsize=9)
        ax.set_xlabel('Input Pos', fontsize=8)
        
        # Only show y-label on leftmost column
        if step % cols == 0:
            ax.set_ylabel('Gen Pos', fontsize=8)
        else:
            ax.set_ylabel('')
            
        ax.tick_params(axis='both', which='major', labelsize=7)
        
        if show_diagonal:
            height, width = crs_att.shape
            if height > 1 and width > 1:  # Only draw diagonal if we have a proper 2D matrix
                ax.plot([1, width-1], [height-2, 0], 'r--', linewidth=1, alpha=0.8)
    
    # Hide unused subplots
    for i in range(num_steps, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.4, wspace=0.3)
    
    cbar = fig.colorbar(im, ax=axes_flat[:num_steps], shrink=0.7, aspect=30, pad=0.02)
    cbar.set_label('Attention Score', fontsize=9)
    fig.suptitle(f'Cross-Attention Head {att_head} for Example {ex_idx} (all steps)', fontsize=14, y=0.96)
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
        if token == 2:  # 0 in Dyck path (up step)
            y_coords.append(y_coords[-1] + 1)
            x_coords.append(x_coords[-1])
        elif token == 3:  # 1 in Dyck path (right step)
            y_coords.append(y_coords[-1])
            x_coords.append(x_coords[-1] + 1)
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
    outputs = model.generate_with_attention_tracking(
        src=inputs.unsqueeze(0),
        tgt=targets.unsqueeze(0)[:, :-max_new_tokens],
        max_new_tokens=max_new_tokens,
        temperature=1,
        top_k=1
    )
    
    generated = outputs["generated_tokens"][0]
    
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
    ax1.set_title(f'Dyck Paths (Example {ex_idx})', fontsize=12)
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Height')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cross-attention (top-right)
    ax2 = axes[0, 1]
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
    
    im2 = ax2.imshow(crs_att.cpu().numpy(), cmap=cmap, aspect='auto')
    ax2.set_title(f'Cross-Attention (Step {step}, Head {cross_att_head})', fontsize=12)
    ax2.set_xlabel('Input Position')
    ax2.set_ylabel('Generated Position')
    
    # 3. Encoder self-attention (bottom-left)
    ax3 = axes[1, 0]
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
    
    im3 = ax3.imshow(enc_att.cpu().numpy(), cmap=cmap, aspect='auto')
    ax3.set_title(f'Encoder Self-Attention (Head {encoder_att_head})', fontsize=12)
    ax3.set_xlabel('Key Position')
    ax3.set_ylabel('Query Position')
    
    # 4. Decoder self-attention (bottom-right)
    ax4 = axes[1, 1]
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