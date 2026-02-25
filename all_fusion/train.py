import torch

import torch.optim as optim
from torch.utils.data import DataLoader
from ccpb_mlp import CCPB_MLP, MLPLoss
from four_model import load_all_models
from fusion_data_processor import FusionDataProcessor
import numpy as np
import os



def train_mlp(inputs, targets, device, epochs=100, batch_size=32):
    print("\n[MLP Train] Starting training...")

    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CCPB_MLP().to(device)
    criterion = MLPLoss(lambda_val=1.5) 

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)

            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(loader):.4f}, LR: {scheduler.get_last_lr()[0]:.1e}")

    print("[MLP Train] Done.")
    torch.save(model.state_dict(), "ccpb_mlp_best.pth")
    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_configs = {
        'faster_rcnn': {'path': './model_best_faster_rcnn.pth', 'num_classes': 2},
        'mask_rcnn': {'path': './model_best_mask_rcnn.pth', 'num_classes': 2},
        'ssd': {'path': './ssd300-best.pth', 'num_classes': 2},
        'yolov5': {'path': './yolov5_best.pt'}
    }


    models = load_all_models(device, model_configs)


    dataset_root = "/VOCdevkit/VOC2007"

    processor = FusionDataProcessor(models, device)

    if os.path.exists("mlp_train_data.pth"):
        print("[Main] Loading cached MLP training data...")
        data = torch.load("mlp_train_data.pth")
        mlp_inputs, mlp_targets = data['inputs'], data['targets']
        anchors = data['anchors']
        print(f"  -> Loaded {len(mlp_inputs)} samples.")
        print(f"  -> Anchors: {anchors}")
    else:

        if len(models) > 0 and os.path.exists(dataset_root):
            mlp_inputs, mlp_targets, anchors = processor.generate_mlp_data(dataset_root)
        else:
            print("[Main] Mocking data for demonstration...")

            mlp_inputs = torch.randn(1000, 24)
            mlp_targets = torch.randn(1000, 6)
            anchors = np.random.rand(8, 2)


    mlp_model = train_mlp(mlp_inputs, mlp_targets, device, epochs=100)


    print("\n[Main] Running CCPB-Fusion Inference demo...")
    mlp_model.eval()
    with torch.no_grad():
        sample_input = torch.randn(1, 24).to(device)
        output = mlp_model(sample_input)

        # 解析输出
        # [dx, dy, dw, dh, cls, score]
        final_box = output[0, :4]
        final_score = torch.sigmoid(output[0, 5])

        print(f"  Input features: {sample_input.cpu().numpy()[0][:6]} ...")
        print(f"  Fused Box: {final_box.cpu().numpy()}, Score: {final_score.item():.4f}")


if __name__ == "__main__":
    main()
