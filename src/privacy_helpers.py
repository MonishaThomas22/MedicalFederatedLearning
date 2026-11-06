
import secrets

import torch
from opacus import PrivacyEngine


NOISE_MULTIPLIER = 1.0
MAX_GRAD_NORM = 1.0
DELTA = 1e-5


class ClientPrivacyAccountant:

    def __init__(self):
        self.engine = PrivacyEngine(
            accountant="prv",
            secure_mode=False
        )

    def make_private(self, model, optimizer, data_loader):
        return self.engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=NOISE_MULTIPLIER,
            max_grad_norm=MAX_GRAD_NORM,
        )

    def get_epsilon(self):
        return self.engine.get_epsilon(delta=DELTA)


def get_original_model_state(model):
    if hasattr(model, "_module"):
        return model._module.state_dict()

    return model.state_dict()


def secure_aggregate_weighted_deltas(
    global_state,
    client_states,
    sample_counts
):
    if not client_states:
        raise ValueError("No client updates provided.")

    if len(client_states) != len(sample_counts):
        raise ValueError("Client counts do not match.")

    if any(count <= 0 for count in sample_counts):
        raise ValueError("Sample counts must be positive.")

    total_samples = sum(sample_counts)
    client_count = len(client_states)

    # Generate one common seed per client pair.
    pair_seeds = {}

    for i in range(client_count):
        for j in range(i + 1, client_count):
            pair_seeds[(i, j)] = secrets.randbits(63)

    aggregated_state = {}

    for key, global_tensor in global_state.items():

        if not torch.is_floating_point(global_tensor):
            aggregated_state[key] = global_tensor.clone()
            continue

        total_delta = torch.zeros_like(global_tensor)

        for client_index, client_state in enumerate(client_states):

            weight = sample_counts[client_index] / total_samples

            client_delta = (
                client_state[key].to(global_tensor.device)
                - global_tensor
            ) * weight

            for other_index in range(client_count):

                if client_index == other_index:
                    continue

                first = min(client_index, other_index)
                second = max(client_index, other_index)

                seed = pair_seeds[(first, second)]

                generator = torch.Generator(device="cpu")
                generator.manual_seed(seed)

                mask = torch.randn(
                    global_tensor.shape,
                    generator=generator,
                    dtype=torch.float32
                ).to(
                    device=global_tensor.device,
                    dtype=global_tensor.dtype
                )

                if client_index < other_index:
                    client_delta += mask
                else:
                    client_delta -= mask

            total_delta += client_delta

        aggregated_state[key] = global_tensor + total_delta

    return aggregated_state
