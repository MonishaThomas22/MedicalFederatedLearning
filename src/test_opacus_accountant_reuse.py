import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine

def train_one_step(engine):
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    dataset = TensorDataset(
        torch.randn(100, 4),
        torch.randint(0, 2, (100,))
    )
    loader = DataLoader(dataset, batch_size=10)

    private_model, private_optimizer, private_loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        rand_on_empty=True
    )

    inputs, labels = next(iter(private_loader))

    private_optimizer.zero_grad()
    loss = nn.CrossEntropyLoss()(private_model(inputs), labels)
    loss.backward()
    private_optimizer.step()

    return engine.get_epsilon(delta=1e-5)

engine = PrivacyEngine(accountant="prv", secure_mode=False)

epsilon_after_round_1 = train_one_step(engine)
print("Epsilon after first optimizer:", epsilon_after_round_1)
print("Accountant history:", engine.accountant.history)

epsilon_after_round_2 = train_one_step(engine)
print("Epsilon after second optimizer:", epsilon_after_round_2)
print("Accountant history:", engine.accountant.history)

print("Privacy accountant accumulated:",
      epsilon_after_round_2 > epsilon_after_round_1)
