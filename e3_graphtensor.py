"""
graph_hamiltonian_isomorphism.py

Implements an isomorphism between Graph Neural Networks (GNNs) and Hamiltonian-based
Graph Neural Networks (HamGNN) by expressing GNN message passing as a Hamiltonian
dynamics simulation. This prototype demonstrates how standard GNN operations
(message passing, aggregation, and update) can be reformulated as the time evolution
of a quantum-inspired Hamiltonian system, enabling the use of symplectic integrators
for more stable and physically meaningful graph learning.

The core isomorphism maps:
- Node features → Quantum state vectors
- Message passing → Hamiltonian interaction terms
- Aggregation → State superposition
- Update → Unitary time evolution
"""

import tensorflow as tf
import tensorflow_gnn as tfgnn
from typing import Optional, Callable, Dict, Any
import numpy as np

class HamiltonianGNN(tf.keras.layers.Layer):
    """
    A GNN layer that implements message passing via Hamiltonian dynamics.
    The layer evolves node states according to a learned Hamiltonian,
    preserving symplectic structure and enabling physically-constrained learning.
    """

    def __init__(self,
                 units: int,
                 hamiltonian_fn: Optional[Callable] = None,
                 time_steps: int = 5,
                 step_size: float = 0.1,
                 **kwargs):
        """
        Args:
            units: Dimension of node state vectors
            hamiltonian_fn: Function that computes the Hamiltonian matrix from graph structure.
                            If None, uses a default pairwise interaction model.
            time_steps: Number of symplectic integration steps
            step_size: Time step size for integration
        """
        super().__init__(**kwargs)
        self.units = units
        self.time_steps = time_steps
        self.step_size = step_size
        self.hamiltonian_fn = hamiltonian_fn or self._default_hamiltonian

        # Learnable parameters for the Hamiltonian
        self.interaction_strength = self.add_weight(
            name='interaction_strength',
            shape=(1,),
            initializer='ones',
            trainable=True
        )

    def _default_hamiltonian(self, graph: tfgnn.GraphTensor) -> tf.Tensor:
        """
        Default Hamiltonian construction: pairwise interactions between connected nodes.
        H = Σ_{i,j} A_{ij} * (I ⊗ I - X ⊗ X - Y ⊗ Y - Z ⊗ Z) * strength
        where A is the adjacency matrix and X,Y,Z are Pauli matrices.
        """
        # Get adjacency matrix (sparse)
        adj = tfgnn.keras.layers.AdjacencyToSparseMatrix()(graph)

        # Construct Pauli matrices for the interaction term
        eye = tf.eye(self.units, dtype=tf.float32)
        pauli_x = tf.constant([[0, 1], [1, 0]], dtype=tf.float32)
        pauli_y = tf.constant([[0, -1j], [1j, 0]], dtype=tf.complex64)
        pauli_z = tf.constant([[1, 0], [0, -1]], dtype=tf.float32)

        # Tile Pauli matrices to match feature dimension
        pauli_x = tf.tile(pauli_x[None, :, :], [self.units // 2, 1, 1])
        pauli_y = tf.tile(pauli_y[None, :, :], [self.units // 2, 1, 1])
        pauli_z = tf.tile(pauli_z[None, :, :], [self.units // 2, 1, 1])

        # Kronecker products for interaction terms
        interaction = tf.eye(self.units * self.units, dtype=tf.float32)
        for pauli in [pauli_x, pauli_y, pauli_z]:
            kron = tf.linalg.LinearOperatorKronecker(
                [tf.linalg.LinearOperatorFullMatrix(pauli),
                 tf.linalg.LinearOperatorFullMatrix(pauli)]
            ).to_dense()
            interaction -= kron

        # Scale by adjacency and interaction strength
        hamiltonian = tf.sparse.sparse_dense_matmul(
            adj,
            tf.reshape(interaction, [self.units * graph.num_nodes, self.units * graph.num_nodes])
        )
        hamiltonian = tf.reshape(hamiltonian, [graph.num_nodes, self.units, graph.num_nodes, self.units])
        hamiltonian = tf.transpose(hamiltonian, [0, 2, 1, 3])  # [N, N, F, F]

        return hamiltonian * self.interaction_strength

    def _symplectic_integrator(self, state: tf.Tensor, hamiltonian: tf.Tensor) -> tf.Tensor:
        """
        Performs symplectic integration (leapfrog) of the Hamiltonian dynamics.
        Args:
            state: Node states [N, F] (real and imaginary parts concatenated)
            hamiltonian: Hamiltonian matrix [N, N, F, F]
        Returns:
            Updated state after time evolution
        """
        # Split state into position (q) and momentum (p) components
        half = self.units // 2
        q = state[:, :half]
        p = state[:, half:]

        # Reshape Hamiltonian for efficient computation
        h_flat = tf.reshape(hamiltonian, [-1, self.units, self.units])

        for _ in range(self.time_steps):
            # Half step for momentum
            hq = tf.einsum('bij,bj->bi', h_flat, q)
            p = p - 0.5 * self.step_size * hq

            # Full step for position
            hp = tf.einsum('bij,bj->bi', h_flat, p)
            q = q + self.step_size * hp

            # Half step for momentum
            hq = tf.einsum('bij,bj->bi', h_flat, q)
            p = p - 0.5 * self.step_size * hq

        return tf.concat([q, p], axis=-1)

    def call(self, graph: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
        """
        Performs Hamiltonian-based message passing on the input graph.
        Args:
            graph: Input GraphTensor with node features
        Returns:
            GraphTensor with updated node features
        """
        # Get node features and ensure proper shape
        node_features = graph.node_sets['nodes'][tfgnn.HIDDEN_STATE]
        if len(node_features.shape) == 2:
            node_features = tf.expand_dims(node_features, 0)

        # Initialize state with real and imaginary parts (position and momentum)
        if node_features.shape[-1] != self.units:
            # Project to desired dimension if needed
            projection = tf.keras.layers.Dense(self.units, use_bias=False)
            state = projection(node_features)
        else:
            state = node_features

        # Pad state to have even dimension for symplectic structure
        if self.units % 2 != 0:
            state = tf.pad(state, [[0, 0], [0, 0], [0, 1]])
            self.units += 1

        # Compute Hamiltonian from graph structure
        hamiltonian = self.hamiltonian_fn(graph)

        # Evolve state according to Hamiltonian dynamics
        updated_state = self._symplectic_integrator(state, hamiltonian)

        # Project back to original feature dimension if needed
        if updated_state.shape[-1] != node_features.shape[-1]:
            projection = tf.keras.layers.Dense(node_features.shape[-1], use_bias=False)
            updated_state = projection(updated_state)

        # Return updated graph
        return graph.replace_features(
            node_sets_fn=lambda node_set, node_set_name: {
                tfgnn.HIDDEN_STATE: updated_state[0] if len(updated_state.shape) == 3 else updated_state
            }
        )

def build_hamiltonian_gnn_model(graph_schema: tfgnn.GraphSchema,
                               node_feature_dim: int,
                               output_dim: int) -> tf.keras.Model:
    """
    Builds a complete GNN model using Hamiltonian layers.
    Args:
        graph_schema: Schema describing the graph structure
        node_feature_dim: Dimension of input node features
        output_dim: Dimension of output predictions
    Returns:
        Compiled Keras model
    """
    input_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

    # Input layer
    input_layer = tfgnn.keras.layers.GraphUpdate(
        node_sets_fn=lambda node_set, node_set_name: {
            tfgnn.HIDDEN_STATE: tf.keras.layers.Dense(64)(node_set[tfgnn.HIDDEN_STATE])
        },
        next_state_fn=HamiltonianGNN(units=64, time_steps=3)
    )

    # Readout layer
    readout = tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name='nodes',
        feature_name=tfgnn.HIDDEN_STATE
    )
    output_layer = tf.keras.layers.Dense(output_dim)

    # Build model
    inputs = tf.keras.layers.Input(type_spec=input_spec)
    x = input_layer(inputs)
    outputs = output_layer(readout(x))
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

if __name__ == '__main__':
    """
    Self-test demonstrating the Hamiltonian GNN isomorphism.
    Creates a small synthetic graph and verifies that:
    1. Hamiltonian construction works
    2. Symplectic integration preserves state norm
    3. Message passing produces meaningful updates
    """
    # Create a simple graph with 3 nodes and 2 edges
    node_features = tf.constant([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=tf.float32)
    edge_src = tf.constant([0, 1], dtype=tf.int32)
    edge_dst = tf.constant([1, 2], dtype=tf.int32)
    edge_set = tfgnn.EdgeSet.from_fields(
        sizes=[2],
        features={},
        adjacency=tfgnn.Adjacency.from_indices(
            source=('nodes', edge_src),
            target=('nodes', edge_dst)
        )
    )
    node_set = tfgnn.NodeSet.from_fields(
        sizes=[3],
        features={tfgnn.HIDDEN_STATE: node_features}
    )
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={'nodes': node_set},
        edge_sets={'edges': edge_set}
    )

    # Test Hamiltonian construction
    hgnn_layer = HamiltonianGNN(units=4)  # 4D state (2D position + 2D momentum)
    hamiltonian = hgnn_layer.hamiltonian_fn(graph)
    print("Hamiltonian shape:", hamiltonian.shape)
    print("Hamiltonian is Hermitian:", tf.reduce_all(
        tf.abs(hamiltonian - tf.linalg.adjoint(hamiltonian)) < 1e-6
    ).numpy())

    # Test symplectic integration
    initial_state = tf.concat([
        node_features,
        tf.zeros_like(node_features)  # Initial momentum
    ], axis=-1)
    updated_state = hgnn_layer._symplectic_integrator(initial_state, hamiltonian)

    # Verify symplectic property (norm preservation)
    initial_norm = tf.reduce_sum(initial_state ** 2, axis=-1)
    updated_norm = tf.reduce_sum(updated_state ** 2, axis=-1)
    print("Norm preservation error:", tf.reduce_max(tf.abs(initial_norm - updated_norm)).numpy())

    # Test full layer
    output_graph = hgnn_layer(graph)
    output_features = output_graph.node_sets['nodes'][tfgnn.HIDDEN_STATE]
    print("Input features:\n", node_features.numpy())
    print("Output features:\n", output_features.numpy())

    # Verify meaningful updates
    feature_change = tf.reduce_mean(tf.abs(output_features - node_features))
    print("Average feature change:", feature_change.numpy())
    assert feature_change > 0.01, "Features should change meaningfully"