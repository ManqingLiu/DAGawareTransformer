import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from typing import Dict, Optional, Tuple


# Import the DAGTransformer from your code
class CustomTransformerEncoderLayer(nn.Module):
    """Custom transformer encoder layer without layer normalization"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                               batch_first=batch_first)

        # Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Self-attention block with attention weights return
        attn_output, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                                   key_padding_mask=src_key_padding_mask,
                                                   is_causal=is_causal,
                                                   average_attn_weights=False)
        src = src + self.dropout1(attn_output)

        # Feedforward block
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)

        return src, attn_weights


class DAGTransformerWithAttention(nn.Module):
    """Modified DAGTransformer that returns attention weights for visualization"""

    def __init__(self,
                 dag: Dict,
                 network_width: int,
                 embedding_dim: int,
                 feedforward_dim: int,
                 num_heads: int,
                 num_layers: int,
                 dropout_rate: float,
                 input_layer_depth: int,
                 encoder_weight: float,
                 activation: str,
                 use_layernorm: bool = False,
                 name: str = None):

        super(DAGTransformerWithAttention, self).__init__()
        self.input_nodes = dag['input_nodes']
        self.output_nodes = dag['output_nodes']
        self.edges = dag['edges']
        self.node_ids = dag['node_ids']
        self.id2node = {v: k for k, v in self.node_ids.items()}

        self.num_nodes = len(self.node_ids.keys())
        self.network_width = network_width
        self.embedding_dim = embedding_dim
        self.feedforward_dim = feedforward_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.encoder_weight = encoder_weight
        self.activation = activation
        self.use_layernorm = use_layernorm
        self.name = name

        # Build adjacency matrix
        self.adj_matrix = torch.zeros(self.num_nodes, self.num_nodes)
        for source_node_name in self.edges.keys():
            source_node_id = self.node_ids[source_node_name]
            for target_node in self.edges[source_node_name]:
                target_node_id = self.node_ids[target_node]
                self.adj_matrix[source_node_id, target_node_id] = 1

        # Create attention mask
        self.attn_mask = ~(self.adj_matrix.bool().T)

        # Create layer list similar to original
        self.layer_list = nn.ModuleList()
        for i in range(input_layer_depth):
            if i == 0:
                self.layer_list.append(nn.Linear(self.num_nodes, self.network_width))
            else:
                self.layer_list.append(nn.Linear(self.network_width, self.network_width))
            self.layer_list.append(nn.ReLU())
            self.layer_list.append(nn.Dropout(self.dropout_rate))
        self.layer_list.append(nn.Linear(self.network_width, 1))

        # Input embedding layer
        self.input_embedding = nn.Linear(1, embedding_dim)

        # Create custom encoder layers that return attention weights
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=self.num_heads,
                dim_feedforward=self.feedforward_dim,
                dropout=self.dropout_rate,
                activation=self.activation,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        # Output projection
        self.embed_to_scalar = nn.Linear(embedding_dim * (self.num_nodes - 1), 1)

    def forward_with_attention(self, x, mask=False, estimator="g-formula"):
        """Forward pass that returns attention weights for visualization"""

        # Combine all inputs
        combined_input = torch.stack([x[node].float() for node in self.input_nodes.keys()], dim=1).squeeze(-1)

        # Create node embeddings
        node_embeddings = []
        for node in self.input_nodes.keys():
            node_input = x[node].float().unsqueeze(-1)
            node_embedding = self.input_embedding(node_input)
            node_embeddings.append(node_embedding)

        # Stack embeddings
        transformer_input = torch.stack(node_embeddings, dim=1)

        # Process through encoder layers and collect attention weights
        attention_weights_all_layers = []
        current_input = transformer_input

        for layer_idx, encoder_layer in enumerate(self.encoder_layers):
            if mask:
                # Create attention mask for this batch
                batch_size = current_input.size(0)
                attn_mask = self.attn_mask.repeat(batch_size * self.num_heads, 1, 1)
                attn_mask = attn_mask.to(current_input.device)
                current_input, attn_weights = encoder_layer(current_input, src_mask=attn_mask)
            else:
                current_input, attn_weights = encoder_layer(current_input)

            attention_weights_all_layers.append(attn_weights)

        # Continue with original forward logic for output
        transformer_output = current_input

        if estimator == "g-formula":
            transformer_output_embeddings = transformer_output[:, :-1, :].view(transformer_output.size(0), -1)
            transformer_output_scalar = self.embed_to_scalar(transformer_output_embeddings)
            combined_input_final = torch.cat([combined_input[:, :-1], transformer_output_scalar * self.encoder_weight],
                                             dim=1)

            for layer in self.layer_list[:-1]:
                combined_input_final = layer(combined_input_final)
            node_output_y = self.layer_list[-1](combined_input_final)

            node_outputs = {'y': node_output_y}
        else:
            # Simplified for visualization - implement other estimators as needed
            node_outputs = {'y': transformer_output.mean(dim=-1)}

        return node_outputs, attention_weights_all_layers


def create_sample_dag():
    """Create a sample DAG structure for demonstration"""
    dag = {
        'input_nodes': {'X1': 0, 'X2': 1, 'A': 2, 'Y': 3},
        'output_nodes': {'y': 0},
        'edges': {
            'X1': ['A', 'Y'],
            'X2': ['A', 'Y'],
            'A': ['Y']
        },
        'node_ids': {'X1': 0, 'X2': 1, 'A': 2, 'Y': 3},
        'nodes': ['X1', 'X2', 'A', 'Y']
    }
    return dag


def create_sample_data(dag, batch_size=4):
    """Create sample input data"""
    data = {}
    for node_name, node_id in dag['input_nodes'].items():
        if node_name == 'A':  # Treatment (binary)
            data[node_name] = torch.randint(0, 2, (batch_size,)).float()
        elif node_name == 'Y':  # Outcome (continuous)
            data[node_name] = torch.randn(batch_size)
        else:  # Confounders (continuous)
            data[node_name] = torch.randn(batch_size)
    return data


def visualize_dag_attention(model, sample_data, dag, layer_idx=0, head_idx=0, sample_idx=0):
    """
    Visualize attention weights from the DAG-aware transformer
    """
    model.eval()

    with torch.no_grad():
        # Get attention weights
        outputs, attention_weights_all_layers = model.forward_with_attention(
            sample_data, mask=True, estimator="g-formula"
        )

        # Extract attention weights for specified layer and head
        attn_weights = attention_weights_all_layers[layer_idx]  # Shape: [batch, heads, seq, seq]
        attn_weights_head = attn_weights[sample_idx, head_idx].cpu().numpy()  # [seq, seq]

        # Handle NaN values by replacing them with 0.0
        attn_weights_head = np.nan_to_num(attn_weights_head, nan=0.0, posinf=0.0, neginf=0.0)

        # Get node names using the model's id2node mapping
        node_names = [model.id2node[i] for i in range(len(model.node_ids))]

        # Get adjacency matrix for comparison
        adj_matrix = model.adj_matrix.cpu().numpy()
        attn_mask = model.attn_mask.cpu().numpy()

    return attn_weights_head, adj_matrix, attn_mask, node_names


def create_dag_attention_visualization():
    """
    Create visualization of actual DAG-aware attention from the model
    """
    # Create sample DAG and model
    dag = create_sample_dag()

    model = DAGTransformerWithAttention(
        dag=dag,
        network_width=64,
        embedding_dim=32,
        feedforward_dim=64,
        num_heads=2,
        num_layers=2,
        dropout_rate=0.1,
        input_layer_depth=2,
        encoder_weight=0.1,
        activation="relu",
        use_layernorm=False
    )

    # Create sample data
    sample_data = create_sample_data(dag, batch_size=4)

    # Get attention weights
    attn_weights, adj_matrix, attn_mask, node_names = visualize_dag_attention(
        model, sample_data, dag, layer_idx=0, head_idx=0, sample_idx=0
    )

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Causal DAG Adjacency Matrix
    ax1 = axes[0]
    sns.heatmap(adj_matrix, annot=True, fmt='.0f', cmap='RdYlBu_r',
                xticklabels=node_names, yticklabels=node_names,
                cbar_kws={'label': 'Edge Exists'}, ax=ax1, vmin=0, vmax=1,
                square=True, linewidths=0.5)
    ax1.set_title('(a) Causal DAG\nAdjacency Matrix', fontweight='bold', fontsize=12)
    ax1.set_xlabel('To Node')
    ax1.set_ylabel('From Node')

    # Plot 2: Attention Mask
    ax2 = axes[1]
    sns.heatmap(attn_mask.astype(int), annot=True, fmt='d', cmap='RdGy',
                xticklabels=node_names, yticklabels=node_names,
                cbar_kws={'label': 'Masked (1) / Allowed (0)'}, ax=ax2,
                square=True, linewidths=0.5)
    ax2.set_title('(b) DAG-Derived\nAttention Mask', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Key Position')
    ax2.set_ylabel('Query Position')

    # Plot 3: Actual Attention Weights
    ax3 = axes[2]
    # Ensure very small values are displayed as 0.000 instead of being hidden
    attn_weights_display = np.where(attn_weights < 1e-6, 0.0, attn_weights)
    sns.heatmap(attn_weights_display, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=node_names, yticklabels=node_names,
                cbar_kws={'label': 'Attention Weight'}, ax=ax3,
                square=True, linewidths=0.5, vmin=0.0)
    ax3.set_title('(c) DAG-Aware\nAttention Weights', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Key Position')
    ax3.set_ylabel('Query Position')

    # Highlight causal relationships in attention weights
    for i in range(len(node_names)):
        for j in range(len(node_names)):
            if attn_weights[i, j] > 0.01:  # Only highlight significant attention
                if adj_matrix[j, i] == 1:  # Causal edge exists (note transpose)
                    rect = Rectangle((j, i), 1, 1, linewidth=3,
                                     edgecolor='orange', facecolor='none', alpha=0.8)
                    ax3.add_patch(rect)
                elif i == j:  # Self-attention
                    rect = Rectangle((j, i), 1, 1, linewidth=2,
                                     edgecolor='cyan', facecolor='none', alpha=0.6)
                    ax3.add_patch(rect)

    # Add legend to the attention weights plot
    causal_patch = mpatches.Patch(color='none', ec='orange', linewidth=3,
                                  label='Causal Edge')
    self_patch = mpatches.Patch(color='none', ec='cyan', linewidth=2,
                                label='Self-Attention')
    #ax3.legend(handles=[causal_patch, self_patch],
    #           loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)

    plt.tight_layout()

    return fig, model, attn_weights, adj_matrix, attn_mask


def create_single_attention_heatmap():
    """
    Create a clean single heatmap for paper publication
    """
    # Create model and get attention weights
    dag = create_sample_dag()

    model = DAGTransformerWithAttention(
        dag=dag,
        network_width=64,
        embedding_dim=32,
        feedforward_dim=64,
        num_heads=1,  # Single head for cleaner visualization
        num_layers=1,
        dropout_rate=0.0,  # No dropout for deterministic results
        input_layer_depth=2,
        encoder_weight=0.1,
        activation="relu",
        use_layernorm=False
    )

    # Create sample data
    sample_data = create_sample_data(dag, batch_size=1)

    # Get attention weights
    attn_weights, adj_matrix, attn_mask, node_names = visualize_dag_attention(
        model, sample_data, dag, layer_idx=0, head_idx=0, sample_idx=0
    )

    # Create single heatmap
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Main heatmap
    # Ensure very small values are displayed as 0.000 instead of being hidden
    attn_weights_display = np.where(attn_weights < 1e-6, 0.0, attn_weights)
    im = sns.heatmap(attn_weights_display, annot=True, fmt='.3f', cmap='viridis',
                     xticklabels=node_names, yticklabels=node_names,
                     cbar_kws={'label': 'Attention Weight'}, ax=ax,
                     square=True, linewidths=1, annot_kws={'fontsize': 11}, vmin=0.0)

    # Customize labels
    ax.set_xlabel('Key Node', fontsize=12, fontweight='bold')
    ax.set_ylabel('Query Node', fontsize=12, fontweight='bold')

    # Highlight patterns
    for i in range(len(node_names)):
        for j in range(len(node_names)):
            if attn_weights[i, j] > 0.01:
                if adj_matrix[j, i] == 1:  # Causal relationship
                    rect = Rectangle((j, i), 1, 1, linewidth=4,
                                     edgecolor='orange', facecolor='none')
                    ax.add_patch(rect)
                elif i == j:  # Self-attention
                    rect = Rectangle((j, i), 1, 1, linewidth=3,
                                     edgecolor='red', facecolor='none')
                    ax.add_patch(rect)

    # Remove the text annotations/labels that were causing clutter

    plt.tight_layout()

    # Print some statistics
    print(f"Attention Statistics:")
    print(f"Max attention weight: {np.max(attn_weights):.3f}")
    print(f"Number of non-zero weights: {np.sum(attn_weights > 0.001)}")
    print(f"Sparsity: {np.sum(attn_weights < 0.001) / attn_weights.size * 100:.1f}%")

    return fig, attn_weights


# Main execution
if __name__ == "__main__":
    import os
    from datetime import datetime

    # Create results directory if it doesn't exist
    results_dir = "experiments/results"
    os.makedirs(results_dir, exist_ok=True)

    print("Creating DAG-aware attention visualization from actual model...")

    # Create comprehensive visualization
    fig_comp, model, attn_weights, adj_matrix, attn_mask = create_dag_attention_visualization()

    print(f"\nModel Architecture:")
    print(f"- Number of nodes: {len(model.node_ids)}")
    print(f"- Embedding dimension: {model.embedding_dim}")
    print(f"- Number of heads: {model.num_heads}")
    print(f"- Number of layers: {model.num_layers}")

    print(f"\nDAG Structure:")
    print(f"- Edges: {model.edges}")
    print(f"- Adjacency matrix shape: {adj_matrix.shape}")
    print(f"- Attention mask sparsity: {np.sum(attn_mask) / attn_mask.size * 100:.1f}%")

    # Create simple version for paper
    print("\nCreating publication-ready single heatmap...")
    fig_simple, attn_simple = create_single_attention_heatmap()

    # Save figures
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save comprehensive visualization
    comp_filename = os.path.join(results_dir, f"dag_attention_comprehensive_{timestamp}.png")
    fig_comp.savefig(comp_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved comprehensive visualization to: {comp_filename}")

    # Save publication-ready version (high quality for paper)
    pub_filename = os.path.join(results_dir, f"dag_attention_publication_{timestamp}.png")
    fig_simple.savefig(pub_filename, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Saved publication-ready heatmap to: {pub_filename}")

    # Also save as PDF for LaTeX (vector format)
    pdf_filename = os.path.join(results_dir, f"dag_attention_publication_{timestamp}.pdf")
    fig_simple.savefig(pdf_filename, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF version for LaTeX to: {pdf_filename}")

    # Save attention weights as numpy array for further analysis
    weights_filename = os.path.join(results_dir, f"attention_weights_{timestamp}.npy")
    np.save(weights_filename, attn_simple)
    print(f"Saved attention weights array to: {weights_filename}")

    # Save model configuration and statistics
    stats_filename = os.path.join(results_dir, f"attention_stats_{timestamp}.txt")
    with open(stats_filename, 'w') as f:
        f.write("DAG-Aware Transformer Attention Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model Architecture:\n")
        f.write(f"- Number of nodes: {len(model.node_ids)}\n")
        f.write(f"- Embedding dimension: {model.embedding_dim}\n")
        f.write(f"- Number of heads: {model.num_heads}\n")
        f.write(f"- Number of layers: {model.num_layers}\n\n")
        f.write(f"DAG Structure:\n")
        f.write(f"- Edges: {model.edges}\n")
        f.write(f"- Node IDs: {model.node_ids}\n")
        f.write(f"- Adjacency matrix shape: {adj_matrix.shape}\n")
        f.write(f"- Attention mask sparsity: {np.sum(attn_mask) / attn_mask.size * 100:.1f}%\n\n")
        f.write(f"Attention Statistics:\n")
        f.write(f"- Max attention weight: {np.max(attn_simple):.3f}\n")
        f.write(f"- Number of non-zero weights: {np.sum(attn_simple > 0.001)}\n")
        f.write(f"- Sparsity: {np.sum(attn_simple < 0.001) / attn_simple.size * 100:.1f}%\n")
    print(f"Saved statistics and configuration to: {stats_filename}")