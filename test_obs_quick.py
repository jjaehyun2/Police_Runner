"""Quick validation test for ObservationBuilder."""
from pursuit_evasion_rl.env.road_network import RoadNetwork
from pursuit_evasion_rl.env.observations import ObservationBuilder
import numpy as np

# Create a simple fixed network
config = {
    'network_mode': 'fixed',
    'fixed_edges': [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 3)]
}
network = RoadNetwork(config)
print(f'Nodes: {network.num_nodes}, Max degree: {network.max_degree}')
print(f'Boundary nodes: {network.boundary_nodes}')

# Create ObservationBuilder (3 police + 1 fugitive = 4 agents)
obs_builder = ObservationBuilder(network, num_police=3, max_degree=network.max_degree)

# Build observation space
obs_space = obs_builder.build_observation_space()
print(f'Observation space: {obs_space}')

# Test get_observation
positions = {'police_0': 0, 'police_1': 1, 'police_2': 2, 'fugitive': 4}
obs = obs_builder.get_observation('police_0', positions, current_step=5)
print(f'my_position: {obs["my_position"]}')
print(f'neighbors: {obs["neighbors"]}')
print(f'other_positions: {obs["other_positions"]}')
print(f'nearest_boundary_dist: {obs["nearest_boundary_dist"]}')
print(f'current_step: {obs["current_step"]}')

# Verify shapes
assert obs["my_position"].shape == (1,), f"Expected (1,), got {obs['my_position'].shape}"
assert obs["neighbors"].shape == (network.max_degree,), f"Expected ({network.max_degree},), got {obs['neighbors'].shape}"
assert obs["other_positions"].shape == (3,), f"Expected (3,), got {obs['other_positions'].shape}"
assert obs["nearest_boundary_dist"].shape == (1,), f"Expected (1,), got {obs['nearest_boundary_dist'].shape}"
assert obs["current_step"].shape == (1,), f"Expected (1,), got {obs['current_step'].shape}"

# Verify values
assert obs["my_position"][0] == 0, "my_position should be 0"
assert obs["current_step"][0] == 5, "current_step should be 5"

# Check neighbors padding (node 0 has neighbors [1, 2], max_degree=3)
neighbors_0 = network.get_neighbors(0)
print(f'Node 0 actual neighbors: {neighbors_0}')
for i, n in enumerate(neighbors_0):
    assert obs["neighbors"][i] == n, f"Neighbor mismatch at index {i}"
for i in range(len(neighbors_0), network.max_degree):
    assert obs["neighbors"][i] == -1, f"Padding should be -1 at index {i}"

# Test with removed agent (position = -1)
positions_with_removed = {'police_0': 0, 'police_1': -1, 'police_2': 2, 'fugitive': 4}
obs2 = obs_builder.get_observation('police_0', positions_with_removed, current_step=10)
print(f'\nWith removed agent:')
print(f'other_positions: {obs2["other_positions"]}')
assert -1 in obs2["other_positions"], "Removed agent should have position -1"

# Test removed agent's own observation
obs3 = obs_builder.get_observation('police_1', positions_with_removed, current_step=10)
print(f'\nRemoved agent observation:')
print(f'my_position: {obs3["my_position"]}')
print(f'neighbors: {obs3["neighbors"]}')
print(f'nearest_boundary_dist: {obs3["nearest_boundary_dist"]}')
assert obs3["my_position"][0] == -1, "Removed agent position should be -1"
assert np.all(obs3["neighbors"] == -1), "Removed agent neighbors should all be -1"
assert obs3["nearest_boundary_dist"][0] == network.num_nodes, "Removed agent boundary dist should be num_nodes"

# Verify observation fits within observation space
assert obs_space.contains(obs), "Observation should be within observation space"

print('\n=== All validation tests passed! ===')
