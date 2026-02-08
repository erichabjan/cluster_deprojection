import jax
from jax import random

import flax.linen as nn
import jraph
from typing import Callable, Sequence
import jax.numpy as jnp

import functools

class MLP(nn.Module):
    """Multi-layer perceptron with customizable activation."""
    
    feature_sizes: Sequence[int]
    activation: Callable = nn.gelu
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x, deterministic):
        for i, features in enumerate(self.feature_sizes[:-1]):
            x = nn.Dense(features)(x)
            x = self.activation(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        
        # Final layer without activation
        x = nn.Dense(self.feature_sizes[-1])(x)
        return x


class PairNorm(nn.Module):
    """PairNorm normalization layer."""
    
    @nn.compact
    def __call__(self, features, rescale_factor=1.0):
        eps = 1e-5
        
        # Center features
        feature_mean = jnp.mean(features, axis=0, keepdims=True)
        feature_centered = features - feature_mean
        
        # L2 norm per node
        feature_l2 = jnp.sqrt(jnp.sum(jnp.square(feature_centered), axis=1, keepdims=True))
        
        # Mean L2 norm across all nodes
        feature_l2_mean = jnp.sqrt(jnp.mean(jnp.square(feature_l2)))
        
        # Normalize
        features_normalized = (feature_centered / (feature_l2 + eps) * feature_l2_mean * rescale_factor)
        
        return features_normalized

class GraphNorm(nn.Module):
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x, counts, deterministic):

        N_total = x.shape[0]
        G = counts.shape[0] # max number of graphs

        starter    = jnp.zeros((N_total,), dtype=jnp.int32)
        # indices where each graph starts
        offsets    = jnp.cumsum(counts) - counts
        starter    = starter.at[offsets].set(1)
        seg_ids    = jnp.cumsum(starter) - 1

        # graph mean
        sum_per_g  = jax.ops.segment_sum(x, seg_ids, G)
        mean_per_g = sum_per_g / counts[:, None]
        centred    = x - mean_per_g[seg_ids]

        # graph standard deviation
        sq_sum_per_g = jax.ops.segment_sum(centred**2, seg_ids, G)
        var_per_g    = sq_sum_per_g / counts[:, None]
        std_per_g    = jnp.sqrt(var_per_g + self.eps)
        x_hat        = centred / std_per_g[seg_ids]

        # learnable affine transform
        gamma = self.param("gamma", nn.initializers.ones,  (x.shape[-1],))
        beta = self.param("beta",  nn.initializers.zeros, (x.shape[-1],))

        return gamma * x_hat + beta




def get_node_mlp_updates(mlp_feature_sizes: Sequence[int], dropout_rate: float = 0.0, 
                        name: str = None) -> Callable:
    """Get node update function."""
    
    def update_fn(nodes, sent_attributes, received_attributes, globals, 
                 deterministic=True):
        if received_attributes is not None:
            inputs = jnp.concatenate([nodes, received_attributes, globals], axis=1)
        else:
            inputs = jnp.concatenate([nodes, globals], axis=1)
        
        return MLP(mlp_feature_sizes, dropout_rate=dropout_rate, name=name)(
            inputs, deterministic=deterministic
        )
    
    return update_fn


def get_edge_mlp_updates(mlp_feature_sizes: Sequence[int], dropout_rate: float = 0.0,
                        relative_updates: bool = False, name: str = None) -> Callable:
    """Get edge update function."""
    
    def update_fn(edges, senders, receivers, globals, deterministic=True):
        if edges is not None:
            if relative_updates:
                inputs = jnp.concatenate([edges, senders - receivers, globals], axis=1)
            else:
                inputs = jnp.concatenate([edges, senders, receivers, globals], axis=1)
        else:
            if relative_updates:
                inputs = jnp.concatenate([senders - receivers, globals], axis=1)
            else:
                inputs = jnp.concatenate([senders, receivers, globals], axis=1)
        
        return MLP(mlp_feature_sizes, dropout_rate=dropout_rate, name=name)(
            inputs, deterministic=deterministic
        )
    
    return update_fn


def get_attention_logit_fn(hidden_size: int, dropout_rate: float = 0.0, 
                          name: str = None) -> Callable:
    """Get attention logits function."""
    
    def attention_logit_fn(edges, senders, receivers, globals, deterministic=True):
        inputs = jnp.concatenate([edges, senders, receivers, globals], axis=-1)
        return MLP([hidden_size, 1], dropout_rate=dropout_rate, name=name)(
            inputs, deterministic=deterministic
        )
    
    return attention_logit_fn


class GraphConvNet(nn.Module):
    """Graph Convolutional Network for velocity prediction."""
    
    latent_size: int
    hidden_size: int
    num_mlp_layers: int
    message_passing_steps: int
    skip_connections: bool = True
    edge_skip_connections: bool = True
    norm: str = "layer"
    attention: bool = False
    shared_weights: bool = False
    relative_updates: bool = False
    output_dim: int = 3
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, deterministic) -> jraph.GraphsTuple:
        """Apply GNN to predict unobserved fields."""
        
        mlp_feature_sizes = [self.hidden_size] * self.num_mlp_layers + [self.latent_size]
        
        # Initial embedding
        if graph.edges is None:
            mlp_node_embed = MLP(mlp_feature_sizes, dropout_rate=self.dropout_rate)
            embed_node_fn = functools.partial(mlp_node_embed, deterministic=deterministic)
            embedder = jraph.GraphMapFeatures(embed_node_fn=embed_node_fn)

        else:
            mlp_node_embed = MLP(mlp_feature_sizes, dropout_rate=self.dropout_rate)
            mlp_edge_embed = MLP(mlp_feature_sizes, dropout_rate=self.dropout_rate)

            embed_node_fn = functools.partial(mlp_node_embed, deterministic=deterministic)
            embed_edge_fn = functools.partial(mlp_edge_embed, deterministic=deterministic)

            embedder = jraph.GraphMapFeatures(embed_node_fn=embed_node_fn, embed_edge_fn=embed_edge_fn)
        
        graph = embedder(graph)

        if graph.edges is None:
            num_edges = graph.senders.shape[0]  # jraph uses senders/receivers for edge count
            graph = graph._replace(edges=jnp.zeros((num_edges, self.latent_size), graph.nodes.dtype))
        
        # Ensure globals are properly shaped
        if graph.globals.ndim > 1:
            graph = graph._replace(globals=graph.globals.reshape(graph.globals.shape[0], -1))
        

        # Apply normalization
        if self.norm == "layer":
            norm = nn.LayerNorm()
        elif self.norm == "pair":
            norm = PairNorm()
        elif self.norm == "graph":
            norm = GraphNorm()
        else:
            norm = lambda x: x  # Identity
        
        if self.shared_weights:
            node_mlp_shared = MLP(mlp_feature_sizes, dropout_rate=self.dropout_rate,
                                  name="update_node_fn_shared")
            edge_mlp_shared = MLP(mlp_feature_sizes, dropout_rate=self.dropout_rate,
                                  name="update_edge_fn_shared")
            if self.attention:
                attn_mlp_shared = MLP([self.hidden_size, 1], dropout_rate=self.dropout_rate,
                                       name="attention_logit_fn_shared")

            def update_node_fn_shared(nodes, sent, recv, globals, deterministic=True):
                inputs = (jnp.concatenate([nodes, recv, globals], axis=1)
                  if recv is not None else jnp.concatenate([nodes, globals], axis=1))
                return node_mlp_shared(inputs, deterministic=deterministic)

            def update_edge_fn_shared(edges, senders, receivers, globals, deterministic=True):
                if edges is not None:
                    inputs = (jnp.concatenate([edges, senders - receivers, globals], axis=1)
                      if self.relative_updates
                      else jnp.concatenate([edges, senders, receivers, globals], axis=1))
                else:
                    inputs = (jnp.concatenate([senders - receivers, globals], axis=1)
                      if self.relative_updates
                      else jnp.concatenate([senders, receivers, globals], axis=1))
                return edge_mlp_shared(inputs, deterministic=deterministic)

            if self.attention:
                def attention_logit_fn_shared(edges, s, r, g, deterministic=True):
                    inputs = jnp.concatenate([edges, s, r, g], axis=-1)
                    return attn_mlp_shared(inputs, deterministic=deterministic)


        # Message passing steps
        for step in range(self.message_passing_steps):
            attn_kwargs = {}
            if self.attention:
                if self.shared_weights:
                    def attn_logit_fn(edges, s, r, g, deterministic=True):
                        inputs = jnp.concatenate([edges, s, r, g], axis=-1)
                        return attn_mlp_shared(inputs, deterministic=deterministic)
                else:
                    # per-step attention MLP
                    attention_logit_fn_step = get_attention_logit_fn(
                        self.hidden_size, dropout_rate=self.dropout_rate, name=f"attention_logit_fn_{step}")
                        
                    def attn_logit_fn(*args, deterministic=deterministic, fn=attention_logit_fn_step):
                        return fn(*args, deterministic=deterministic)

                def attn_reduce_fn(edges, weights):
                    return edges * weights

                attn_kwargs = dict(
                    attention_logit_fn=lambda *args, f=attn_logit_fn: f(*args, deterministic=deterministic),
                    attention_reduce_fn=attn_reduce_fn,)

            if self.shared_weights:
                graph_net = jraph.GraphNetwork(
                    update_node_fn=lambda *args: update_node_fn_shared(*args, deterministic=deterministic),
                    update_edge_fn=lambda *args: update_edge_fn_shared(*args, deterministic=deterministic),
                    **attn_kwargs,)

            else:
                update_node_fn = get_node_mlp_updates(
                    mlp_feature_sizes, 
                    dropout_rate=self.dropout_rate,
                    name=f"update_node_fn_{step}"
                )
                update_edge_fn = get_edge_mlp_updates(
                    mlp_feature_sizes,
                    dropout_rate=self.dropout_rate,
                    relative_updates=self.relative_updates,
                    name=f"update_edge_fn_{step}"
                )
                attention_logit_fn = (
                    get_attention_logit_fn(
                        self.hidden_size,
                        dropout_rate=self.dropout_rate,
                        name=f"attention_logit_fn_{step}"
                    ) if self.attention else None
                )
                
                graph_net = jraph.GraphNetwork(
                    update_node_fn=lambda *args: update_node_fn(*args, deterministic=deterministic),
                    update_edge_fn=lambda *args: update_edge_fn(*args, deterministic=deterministic),
                    **attn_kwargs,
                )
            
            # Apply graph network with skip connections
            if self.skip_connections:
                new_graph = graph_net(graph)
                graph = graph._replace(
                    nodes=graph.nodes + new_graph.nodes,
                    edges=(
                        new_graph.edges if graph.edges is None or not self.edge_skip_connections
                        else graph.edges + new_graph.edges
                    ),
                )
            else:
                graph = graph_net(graph)
            
            if graph.nodes is not None:
                if self.norm == 'graph':
                    graph = graph._replace(nodes=norm(graph.nodes, graph.n_node, deterministic))
                else:
                    graph = graph._replace(nodes=norm(graph.nodes))
            if graph.edges is not None:
                if self.norm == 'graph':
                    graph = graph._replace(edges=norm(graph.edges, graph.n_edge, deterministic))
                else: 
                    graph = graph._replace(edges=norm(graph.edges))
        
        # Final decoder to output dimension
        decode_node_mlp = MLP([self.hidden_size] * self.num_mlp_layers + [self.output_dim], dropout_rate=self.dropout_rate)
        decoder = jraph.GraphMapFeatures(embed_node_fn=functools.partial(decode_node_mlp, deterministic=deterministic))
        
        return decoder(graph)