"""Visualization utilities for geometric policy and VLA debugging.

Creates visualizations of predicted actions with geometric feasibility overlays,
and comprehensive VLA debug panels for attention analysis, action trajectories,
and cross-attention diagnostics.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.transforms import Affine2D
import io
from PIL import Image


def draw_pusht_block(ax, x, y, angle, color='#1f77b4', alpha=0.8):
    """Draw a T-shaped block as in PushT environment.

    Args:
        ax: matplotlib axis
        x, y: center position
        angle: rotation in radians
        color: block color
        alpha: transparency
    """
    # T-shape dimensions (matching PushT)
    # Horizontal bar (top of T)
    h_width = 100
    h_height = 20
    # Vertical bar (stem of T)
    v_width = 20
    v_height = 80

    # Create T-shape vertices (centered at origin)
    # Top bar
    top_bar = np.array([
        [-h_width/2, h_height/2],
        [h_width/2, h_height/2],
        [h_width/2, -h_height/2],
        [-h_width/2, -h_height/2],
    ])

    # Bottom stem (offset down from origin)
    stem_offset = h_height/2 + v_height/2
    bottom_stem = np.array([
        [-v_width/2, -h_height/2],
        [v_width/2, -h_height/2],
        [v_width/2, -h_height/2 - v_height],
        [-v_width/2, -h_height/2 - v_height],
    ])

    # Rotate and translate
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    top_bar_rot = top_bar @ rotation_matrix.T + np.array([x, y])
    bottom_stem_rot = bottom_stem @ rotation_matrix.T + np.array([x, y])

    # Draw both parts
    top_patch = Polygon(top_bar_rot, closed=True, facecolor=color,
                       edgecolor='black', linewidth=2, alpha=alpha)
    stem_patch = Polygon(bottom_stem_rot, closed=True, facecolor=color,
                        edgecolor='black', linewidth=2, alpha=alpha)

    ax.add_patch(top_patch)
    ax.add_patch(stem_patch)


def visualize_geometric_predictions(
    obs_batch,
    action_pred_batch,
    geom_features_batch,
    num_samples=4,
):
    """Visualize predicted actions with geometric feasibility overlays.

    Args:
        obs_batch: [B, obs_horizon, 5] state observations
        action_pred_batch: [B, horizon, 2] predicted actions
        geom_features_batch: dict with geometric features [B, ...]
        num_samples: number of samples to visualize

    Returns:
        PIL Image with visualizations
    """
    # Validate inputs
    if geom_features_batch is None:
        raise ValueError("geom_features_batch is None")

    required_keys = ['dist_to_block', 'will_contact', 'penetration_depth', 'contact_normal_alignment']
    missing_keys = [k for k in required_keys if k not in geom_features_batch]
    if missing_keys:
        raise ValueError(f"Missing keys in geom_features_batch: {missing_keys}. Available keys: {list(geom_features_batch.keys())}")

    batch_size = min(obs_batch.shape[0], num_samples)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for idx in range(batch_size):
        ax = axes[idx]

        # Extract state (latest timestep)
        state = obs_batch[idx, -1].cpu().numpy()  # [5]
        agent_x, agent_y, block_x, block_y, block_angle = state

        # Extract predicted action (first timestep)
        action = action_pred_batch[idx, 0].detach().cpu().numpy()  # [2]
        pred_x, pred_y = action

        # Extract geometric features
        dist_to_block = geom_features_batch['dist_to_block'][idx].item()
        will_contact = geom_features_batch['will_contact'][idx].item()
        penetration = geom_features_batch['penetration_depth'][idx].item()
        push_quality = geom_features_batch['contact_normal_alignment'][idx].item()

        # Set up plot (PushT is 512x512 world)
        ax.set_xlim(0, 512)
        ax.set_ylim(0, 512)
        ax.set_aspect('equal')
        ax.set_facecolor('#f0f0f0')  # Light gray background like PushT
        ax.grid(True, alpha=0.2, color='white', linewidth=1.5)

        # Draw goal zone (top-right corner target)
        goal_circle = Circle(
            (412, 412),  # PushT goal position
            radius=50,
            color='lightgreen',
            alpha=0.2,
            linewidth=2,
            edgecolor='green',
            linestyle='--',
            fill=True
        )
        ax.add_patch(goal_circle)
        ax.text(412, 412, 'GOAL', ha='center', va='center',
                fontsize=10, fontweight='bold', color='green', alpha=0.5)

        # Draw T-shaped block (actual PushT shape)
        draw_pusht_block(ax, block_x, block_y, block_angle, color='#ff6b6b', alpha=0.95)

        # Draw current agent position (PushT agent is a small circle)
        agent_circle = Circle(
            (agent_x, agent_y),
            radius=12,
            color='#4ecdc4',
            alpha=0.7,
            edgecolor='#2a9d8f',
            linewidth=2,
            label='Current Agent'
        )
        ax.add_patch(agent_circle)

        # Draw predicted action position (color-coded by geometric feasibility)
        # Green = good push, Orange = contact but poor angle, Red = no contact
        if will_contact > 0.5 and push_quality > 0.5:
            color = '#2ecc71'  # green
            status = 'Good Push'
        elif will_contact > 0.5:
            color = '#f39c12'  # orange
            status = 'Poor Angle'
        else:
            color = '#e74c3c'  # red
            status = 'No Contact'

        # Draw arrow from current to predicted position (thicker, more visible)
        dx, dy = pred_x - agent_x, pred_y - agent_y
        arrow_length = np.sqrt(dx**2 + dy**2)

        if arrow_length > 1:  # Only draw arrow if movement is significant
            ax.annotate('', xy=(pred_x, pred_y), xytext=(agent_x, agent_y),
                       arrowprops=dict(arrowstyle='->', color=color, lw=3,
                                     alpha=0.8, mutation_scale=25))

        # Draw predicted position
        pred_circle = Circle(
            (pred_x, pred_y),
            radius=12,
            color=color,
            alpha=0.6,
            edgecolor='black',
            linewidth=2,
            linestyle='--',
            fill=True
        )
        ax.add_patch(pred_circle)

        # Add text with geometric metrics
        metrics_text = (
            f"Dist: {dist_to_block:.1f}px\n"
            f"Contact: {will_contact:.2f}\n"
            f"Penetr: {penetration:.1f}px\n"
            f"Quality: {push_quality:.2f}"
        )
        ax.text(
            0.02, 0.98, metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white',
                     edgecolor='gray', alpha=0.9, pad=0.5)
        )

        # Title with status
        title_color = color
        ax.set_title(f'Sample {idx+1}: {status}', fontsize=12,
                    fontweight='bold', color=title_color, pad=10)

        # Add legend in bottom right
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4ecdc4', edgecolor='#2a9d8f', label='Agent'),
            Patch(facecolor='#ff6b6b', edgecolor='black', label='T-Block'),
            Patch(facecolor='lightgreen', edgecolor='green', label='Goal', alpha=0.3),
            Patch(facecolor=color, edgecolor='black', linestyle='--', label='Predicted', alpha=0.6)
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.9)

    # Hide unused subplots
    for idx in range(batch_size, 4):
        axes[idx].axis('off')

    # Add overall title
    fig.suptitle('PushT Geometric Policy Predictions', fontsize=16,
                fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def create_geometric_heatmap(obs_batch, geom_features_batch, num_samples=4):
    """Create heatmap showing geometric feasibility across the workspace.

    Args:
        obs_batch: [B, obs_horizon, 5] state observations
        geom_features_batch: dict with geometric features
        num_samples: number of samples to show

    Returns:
        PIL Image with heatmap
    """
    batch_size = min(obs_batch.shape[0], num_samples)

    fig, axes = plt.subplots(1, batch_size, figsize=(4*batch_size, 4))
    if batch_size == 1:
        axes = [axes]

    for idx in range(batch_size):
        ax = axes[idx]

        # Extract state
        state = obs_batch[idx, -1].cpu().numpy()
        agent_x, agent_y, block_x, block_y, block_angle = state

        # Create grid for heatmap
        grid_size = 64
        x = np.linspace(0, 512, grid_size)
        y = np.linspace(0, 512, grid_size)
        X, Y = np.meshgrid(x, y)

        # Compute distance-based heatmap
        distances = np.sqrt((X - block_x)**2 + (Y - block_y)**2)
        heatmap = 1.0 / (1.0 + distances / 100.0)  # Closer = brighter

        # Plot heatmap
        im = ax.imshow(
            heatmap,
            extent=[0, 512, 0, 512],
            origin='lower',
            cmap='RdYlGn',
            alpha=0.6
        )

        # Overlay block and agent
        ax.plot(block_x, block_y, 'bs', markersize=10, label='Block')
        ax.plot(agent_x, agent_y, 'ko', markersize=8, label='Agent')

        ax.set_xlim(0, 512)
        ax.set_ylim(0, 512)
        ax.set_title(f'Feasibility Heatmap {idx+1}')
        ax.legend(loc='upper right', fontsize=8)
        plt.colorbar(im, ax=ax, label='Feasibility')

    plt.tight_layout()

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def visualize_vla_predictions(
    current_images,
    future_image,
    attn_weights,
    actions,
    num_samples=4,
):
    """Visualize VLA predictions with current/future images and attention.

    Args:
        current_images: [B, To, C, H, W] current observation images
        future_image: [B, C, H, W] rendered future images
        attn_weights: List of dicts with attention weights from each layer
        actions: [B, Ta, act_dim] predicted actions
        num_samples: Number of samples to visualize

    Returns:
        PIL Image with visualizations
    """
    batch_size = min(current_images.shape[0], num_samples)
    To = current_images.shape[1]  # Number of observation frames

    # Create figure: 2 rows x num_samples columns
    fig, axes = plt.subplots(2, batch_size, figsize=(4 * batch_size, 8))
    if batch_size == 1:
        axes = axes.reshape(2, 1)

    for idx in range(batch_size):
        # Top row: Current image(s) with attention overlay
        ax_current = axes[0, idx]

        # Show latest current image
        current_img = current_images[idx, -1].detach().cpu().numpy()  # [C, H, W]
        current_img = np.transpose(current_img, (1, 2, 0))  # [H, W, C]

        # Denormalize if needed (assuming [0, 1] range)
        if current_img.max() <= 1.0:
            current_img = np.clip(current_img, 0, 1)

        ax_current.imshow(current_img)
        ax_current.set_title(f'Sample {idx+1}: Current', fontsize=10, fontweight='bold')
        ax_current.axis('off')

        # Overlay attention heatmap if available
        if attn_weights and len(attn_weights) > 0:
            # Get attention from last layer to current image
            last_layer_attn = attn_weights[-1]  # Last decoder layer
            if 'current' in last_layer_attn:
                # Attention weights shape depends on average_attn_weights setting:
                #   Default (averaged): [B, Ta, N_tokens]
                #   Not averaged:       [B, n_heads, Ta, N_tokens]
                attn = last_layer_attn['current'][idx].detach().cpu().numpy()

                # Average down to [N_tokens] regardless of input shape
                if attn.ndim == 1:
                    attn_map = attn  # Already [N]
                elif attn.ndim == 2:
                    attn_map = attn.mean(axis=0)  # [Ta, N] -> [N]
                else:
                    attn_map = attn.mean(axis=tuple(range(attn.ndim - 1)))  # [*, N] -> [N]

                # Reshape to spatial grid (excluding CLS token)
                n_tokens = len(attn_map)
                # For CLIP ViT-B/16 with 224x224: 14x14 = 196 patches (+1 CLS = 197)
                if n_tokens == 197:  # CLIP ViT-B/16
                    attn_map = attn_map[1:]  # Remove CLS token
                    grid_size = 14
                elif n_tokens == 50:  # CLIP ViT-B/32
                    attn_map = attn_map[1:]
                    grid_size = 7
                else:
                    grid_size = int(np.sqrt(n_tokens))

                attn_grid = attn_map[:grid_size**2].reshape(grid_size, grid_size)

                # Resize attention map to image size
                from scipy.ndimage import zoom
                h, w = current_img.shape[:2]
                attn_resized = zoom(attn_grid, (h / grid_size, w / grid_size), order=1)

                # Overlay as heatmap
                ax_current.imshow(attn_resized, cmap='jet', alpha=0.4, interpolation='bilinear')

        # Bottom row: Future rendered image
        ax_future = axes[1, idx]

        future_img = future_image[idx].detach().cpu().numpy()  # [C, H, W]
        future_img = np.transpose(future_img, (1, 2, 0))  # [H, W, C]

        if future_img.max() <= 1.0:
            future_img = np.clip(future_img, 0, 1)

        ax_future.imshow(future_img)
        ax_future.set_title(f'Rendered Action', fontsize=10, fontweight='bold')
        ax_future.axis('off')

        # Overlay action trajectory
        if actions is not None:
            action_traj = actions[idx].detach().cpu().numpy()  # [Ta, 2]
            # Assuming actions are normalized to [0, 512] range
            # Scale to image coordinates
            img_h, img_w = future_img.shape[:2]
            traj_x = action_traj[:, 0] * (img_w / 512.0)
            traj_y = action_traj[:, 1] * (img_h / 512.0)

            # Plot trajectory
            ax_future.plot(traj_x, traj_y, 'r-', linewidth=2, alpha=0.7, label='Action Trajectory')
            ax_future.scatter(traj_x[0], traj_y[0], c='green', s=100, marker='o', label='Start', zorder=5)
            ax_future.scatter(traj_x[-1], traj_y[-1], c='red', s=100, marker='x', label='End', zorder=5)
            ax_future.legend(loc='upper right', fontsize=8)

    # Overall title
    fig.suptitle('VLA Cross-Attention: Current vs Future with Attention Overlay',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)

    return img


def visualize_vla_debug(
    net,
    sampled_actions=None,
    gt_actions=None,
    render_state=None,
    num_samples=4,
):
    """Comprehensive VLA debug visualization.

    Creates a multi-panel figure showing:
    - Row 0: Current + Future images side by side with draft action overlay
    - Row 1: Draft head spatial attention heatmap on current image
    - Row 2: Decoder modality contribution per layer (output L2 norm based)
    - Row 3: Action trajectory (predicted vs GT if available)

    Args:
        net: VLACrossAttentionPolicy with stored debug attributes
        sampled_actions: [B, Ta, act_dim] final sampled actions (unnormalized, pixel coords)
        gt_actions: [B, Ta, act_dim] ground truth actions (unnormalized, pixel coords), optional
        render_state: [B, To, 5] or [B, 5] raw state, optional (uses net._render_state if None)
        num_samples: Number of samples to visualize

    Returns:
        PIL Image with debug visualization
    """
    # Gather all stored data from the network
    current_images = getattr(net, '_last_current_images', None)
    future_image = getattr(net, '_last_future_image', None)
    attn_weights = getattr(net, '_last_attn_weights', None)
    velocity_pred = getattr(net, '_last_velocity_pred', None)
    draft_action = getattr(net, '_last_draft_action', None)
    draft_attn_weights = getattr(net, '_last_draft_attn_weights', None)
    state_tokens = getattr(net, '_last_state_tokens', None)

    if current_images is None:
        raise ValueError("No debug data stored in network. Run a forward pass first.")

    batch_size = min(current_images.shape[0], num_samples)
    has_future = future_image is not None
    has_draft_attn = draft_attn_weights is not None
    n_layers = len(attn_weights) if attn_weights else 0

    # --- Layout: 4 rows ---
    # Row 0: Current + Future images
    # Row 1: Draft head attention heatmap on current image
    # Row 2: Decoder modality contribution (output norm)
    # Row 3: Action trajectory on workspace
    n_cols = batch_size
    n_rows = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for idx in range(batch_size):
        # === ROW 0: Images ===
        ax_img = axes[0, idx]

        # Show current image
        current_img = current_images[idx, -1].detach().cpu().numpy()  # [C, H, W]
        current_img = np.transpose(current_img, (1, 2, 0))  # [H, W, C]
        current_img = np.clip(current_img, 0, 1)
        ax_img.imshow(current_img)

        # Overlay future image side by side if available
        if has_future:
            future_img = future_image[idx].detach().cpu().numpy()
            future_img = np.transpose(future_img, (1, 2, 0))
            future_img = np.clip(future_img, 0, 1)
            h, w = current_img.shape[:2]
            composite = np.concatenate([current_img, future_img], axis=1)
            ax_img.clear()
            ax_img.imshow(composite)
            ax_img.axvline(x=w, color='white', linewidth=2, linestyle='--')
            ax_img.text(w * 0.5, 5, 'Current', color='white', fontsize=9,
                       ha='center', fontweight='bold',
                       bbox=dict(facecolor='black', alpha=0.5, pad=2))
            ax_img.text(w * 1.5, 5, 'Future', color='white', fontsize=9,
                       ha='center', fontweight='bold',
                       bbox=dict(facecolor='black', alpha=0.5, pad=2))

            if draft_action is not None:
                da = draft_action[idx].detach().cpu().numpy()
                da_x = da[0] * (w / 512.0) + w
                da_y = da[1] * (h / 512.0)
                ax_img.scatter(da_x, da_y, c='lime', s=120, marker='*',
                             edgecolors='black', linewidths=1, zorder=10)
                ax_img.text(da_x + 3, da_y - 3, 'draft', color='lime',
                           fontsize=7, fontweight='bold')
        else:
            ax_img.text(0.5, 0.02, 'No renderer (norender mode)', color='yellow',
                       fontsize=8, ha='center', transform=ax_img.transAxes,
                       bbox=dict(facecolor='black', alpha=0.5, pad=2))

        ax_img.axis('off')
        ax_img.set_title(f'Sample {idx + 1}', fontsize=10, fontweight='bold')

        # === ROW 1: Draft head spatial attention heatmap ===
        ax_draft = axes[1, idx]

        if has_draft_attn:
            # draft_attn_weights: [B, 1, N_current] where N_current = To * 197
            attn = draft_attn_weights[idx, 0].detach().cpu().numpy()  # [N_current]
            n_tokens = len(attn)
            tokens_per_frame = 197  # CLIP ViT-B/16: 196 patches + 1 CLS

            # Average attention across To frames, reshape to spatial grid
            n_frames = n_tokens // tokens_per_frame
            attn_frames = attn.reshape(n_frames, tokens_per_frame)
            attn_avg = attn_frames.mean(axis=0)  # [197]
            attn_patches = attn_avg[1:]  # Remove CLS token → [196]
            grid_size = 14  # sqrt(196) for ViT-B/16
            attn_grid = attn_patches.reshape(grid_size, grid_size)

            # Show current image with attention overlay
            ax_draft.imshow(current_img)
            from scipy.ndimage import zoom
            h, w = current_img.shape[:2]
            attn_resized = zoom(attn_grid, (h / grid_size, w / grid_size), order=1)
            ax_draft.imshow(attn_resized, cmap='jet', alpha=0.5, interpolation='bilinear')
            ax_draft.set_title('Draft Head: Where is it looking?', fontsize=9,
                             fontweight='bold')
        else:
            ax_draft.text(0.5, 0.5, 'No draft head (norender mode)',
                        ha='center', va='center', transform=ax_draft.transAxes,
                        fontsize=9, color='gray')
            ax_draft.set_title('Draft Head Attention', fontsize=9, fontweight='bold')

        ax_draft.axis('off')

        # === ROW 2: Decoder modality contribution (output L2 norm) ===
        ax_attn = axes[2, idx]

        if attn_weights and n_layers > 0:
            current_norms = []
            future_norms = []
            state_norms = []

            for layer_attn in attn_weights:
                c_norm = layer_attn['current_out_norm'][idx].item() if 'current_out_norm' in layer_attn else 0.0
                f_norm = layer_attn['future_out_norm'][idx].item() if 'future_out_norm' in layer_attn else 0.0
                s_norm = layer_attn['state_out_norm'][idx].item() if 'state_out_norm' in layer_attn else 0.0
                current_norms.append(c_norm)
                future_norms.append(f_norm)
                state_norms.append(s_norm)

            # Convert to percentages
            current_pcts = []
            future_pcts = []
            state_pcts = []
            for c, f, s in zip(current_norms, future_norms, state_norms):
                total = c + f + s
                if total > 0:
                    current_pcts.append(c / total * 100)
                    future_pcts.append(f / total * 100)
                    state_pcts.append(s / total * 100)
                else:
                    current_pcts.append(0)
                    future_pcts.append(0)
                    state_pcts.append(0)

            # Stacked bar chart
            x = np.arange(n_layers)
            width = 0.6
            bottom = np.zeros(n_layers)

            ax_attn.bar(x, current_pcts, width, label='Current img',
                       color='#4ecdc4', bottom=bottom)
            bottom += current_pcts
            if any(f > 0 for f in future_pcts):
                ax_attn.bar(x, future_pcts, width, label='Future img',
                           color='#ff6b6b', bottom=bottom)
                bottom += future_pcts
            if any(s > 0 for s in state_pcts):
                ax_attn.bar(x, state_pcts, width, label='State tokens',
                           color='#f9a825', bottom=bottom)
                bottom += state_pcts

            ax_attn.set_xlabel('Decoder Layer', fontsize=8)
            ax_attn.set_ylabel('Contribution %', fontsize=8)
            ax_attn.set_title('Decoder: Modality Contribution (output norm)',
                            fontsize=9, fontweight='bold')
            ax_attn.set_xticks(x)
            ax_attn.set_xticklabels([f'L{i}' for i in range(n_layers)], fontsize=7)
            ax_attn.legend(fontsize=7, loc='upper right')
            ax_attn.set_ylim(0, 105)

            # Annotate percentages on bars
            for i in range(n_layers):
                # Current img %
                if current_pcts[i] > 8:
                    ax_attn.text(i, current_pcts[i] / 2, f'{current_pcts[i]:.0f}%',
                               ha='center', va='center', fontsize=6, fontweight='bold')
                # Future img %
                if future_pcts[i] > 8:
                    y_pos = current_pcts[i] + future_pcts[i] / 2
                    ax_attn.text(i, y_pos, f'{future_pcts[i]:.0f}%',
                               ha='center', va='center', fontsize=6, fontweight='bold')
                # State %
                if state_pcts[i] > 8:
                    y_pos = current_pcts[i] + future_pcts[i] + state_pcts[i] / 2
                    ax_attn.text(i, y_pos, f'{state_pcts[i]:.0f}%',
                               ha='center', va='center', fontsize=6, fontweight='bold')
        else:
            ax_attn.text(0.5, 0.5, 'No attention data', ha='center', va='center',
                        transform=ax_attn.transAxes)
            ax_attn.axis('off')

        # === ROW 3: Action trajectory on workspace ===
        ax_traj = axes[3, idx]
        ax_traj.set_xlim(0, 512)
        ax_traj.set_ylim(0, 512)
        ax_traj.set_aspect('equal')
        ax_traj.set_facecolor('#f0f0f0')
        ax_traj.invert_yaxis()  # Match image coordinates

        # Draw state info (block + agent positions)
        state_data = render_state if render_state is not None else getattr(net, '_render_state', None)
        if state_data is not None:
            s = state_data[idx].detach().cpu().numpy() if torch.is_tensor(state_data) else state_data[idx]
            if s.ndim == 2:
                s = s[-1]  # Last frame
            agent_x, agent_y, block_x, block_y, block_angle = s[:5]
            draw_pusht_block(ax_traj, block_x, block_y, block_angle,
                           color='#1f77b4', alpha=0.5)
            agent_circle = Circle((agent_x, agent_y), radius=15,
                                facecolor='#4ecdc4', alpha=0.8, edgecolor='black',
                                linewidth=1.5)
            ax_traj.add_patch(agent_circle)
            ax_traj.text(agent_x, agent_y - 20, 'agent', fontsize=7,
                        ha='center', color='#2a9d8f')

        # Plot sampled action trajectory
        if sampled_actions is not None:
            traj = sampled_actions[idx]
            if torch.is_tensor(traj):
                traj = traj.detach().cpu().numpy()
            ax_traj.plot(traj[:, 0], traj[:, 1], 'r-o', markersize=3,
                        linewidth=2, alpha=0.8, label='Predicted')
            ax_traj.scatter(traj[0, 0], traj[0, 1], c='red', s=80,
                          marker='s', zorder=10, edgecolors='black')

        # Plot GT trajectory
        if gt_actions is not None:
            gt = gt_actions[idx]
            if torch.is_tensor(gt):
                gt = gt.detach().cpu().numpy()
            ax_traj.plot(gt[:, 0], gt[:, 1], 'g-o', markersize=3,
                        linewidth=2, alpha=0.8, label='GT')
            ax_traj.scatter(gt[0, 0], gt[0, 1], c='green', s=80,
                          marker='s', zorder=10, edgecolors='black')

        # Plot draft action
        if draft_action is not None:
            da = draft_action[idx].detach().cpu().numpy()
            ax_traj.scatter(da[0], da[1], c='lime', s=150, marker='*',
                          edgecolors='black', linewidths=1.5, zorder=10,
                          label='Draft')

        ax_traj.legend(fontsize=7, loc='upper right')
        ax_traj.set_title('Action Trajectory (512x512 workspace)', fontsize=9,
                         fontweight='bold')
        ax_traj.grid(True, alpha=0.2)

    fig.suptitle('VLA Debug: Images | Draft Attn | Modality Contribution | Actions',
                fontsize=13, fontweight='bold', y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)

    return img
