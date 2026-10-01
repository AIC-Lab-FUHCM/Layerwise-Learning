import torch
from pathlib import Path

run_dir = (
    Path(__file__).resolve().parent
    / 'checkpoints'
    / 'ecg_b_tcn_run_02'
)

checkpoint = torch.load(
    run_dir / 'best.pth',
    map_location='cpu'
)

print('Best epoch:', checkpoint['epoch'] + 1)
print('Best loss:', checkpoint['best_loss'])
print('Config:', checkpoint['config'])

loss_history = torch.load(
    run_dir / 'loss_history.pt',
    map_location='cpu'
)

print('\nLoss history:')

for epoch, loss in enumerate(loss_history, start=1):
    print(f'Epoch {epoch}: {loss:.6f}')