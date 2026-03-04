import os
import numpy as np
from torch.optim import lr_scheduler
import torch
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import time
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from models.network.net_stage1 import net_stage1


class Trainer_stage1:
    def __init__(self, opt):
        self.model = net_stage1()

        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"Total model parameters: {total_params:.2f}M")

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
        trainable_ratio = (trainable_params / total_params) * 100
        print(f"Trainable parameters: {trainable_params:.2f}M ({trainable_ratio:.2f}%)")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.stage1_learning_rate, betas=(0.9, 0.999))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=opt.stage1_lr_decay_step, gamma=opt.stage1_lr_decay_factor)
        self.scaler = GradScaler()

        self.best_val_loss = float('inf')

        self.lambdas = opt.lambdas

    def shuffle_patches(self, data, patch_size):
        batch_size, C, H, W = data.shape
        assert H % patch_size == 0 and W % patch_size == 0

        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        num_patches = num_patches_h * num_patches_w

        # shape: [B, C, 14, 14, 16, 16]
        patches = data.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

        # shape: [B, 196, C, 16, 16]
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, num_patches, C, patch_size, patch_size)
        
        # idx shape: [B, 196]
        idx = torch.stack([torch.randperm(num_patches) for _ in range(batch_size)]).to(data.device)

        # [B, 196, C, 16, 16]
        patches = patches[torch.arange(batch_size).unsqueeze(1), idx]

        patches = patches.reshape(batch_size, num_patches_h, num_patches_w, C, patch_size, patch_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, C, H, W)

        return patches

    # 三块 cls 层面 L2 拉近，使用最后一层的 BCELoss，三块的 L2Loss
    def setting1(self, data, mod_data, target, criterion):
        result, differ_cls_tokens, cls_tokens, mod_cls_tokens = self.model(data, mod_data)
        main_loss = criterion(result.squeeze(1), target.type(torch.float32))
        # l2_loss : cls_tokens 与 mod_cls_tokens
        l2_loss_items = []
        for item in zip(cls_tokens, mod_cls_tokens):
            l2_loss_items.append(torch.mean((item[0] - item[1]) ** 2))
        l2_loss = sum(l2_loss_items)
        return self.lambdas[0] * main_loss + self.lambdas[1] * l2_loss
    
    # 三块 cls 层面 L2 拉近，使用三层的 BCELoss，三块的 L2Loss
    def setting2(self, data, mod_data, target, criterion):
        result, logits_by_differ_cls_tokens, cls_tokens, mod_cls_tokens = self.model(data, mod_data)
        main_loss = sum([criterion(item.squeeze(1), target.type(torch.float32)) for item in logits_by_differ_cls_tokens])
        # l2_loss : cls_tokens 与 mod_cls_tokens
        l2_loss_items = []
        for item in zip(cls_tokens, mod_cls_tokens):
            l2_loss_items.append(torch.mean((item[0] - item[1]) ** 2))
        l2_loss = sum(l2_loss_items)
        return self.lambdas[0] * main_loss + self.lambdas[1] * l2_loss
    
    # 
    def setting3(self, data, mod_data, target, criterion):
        result, logits_by_differ_cls_tokens, cls_tokens, mod_cls_tokens = self.model(data, mod_data)
        main_loss = sum([criterion(item.squeeze(1), target.type(torch.float32)) for item in logits_by_differ_cls_tokens])
        new_loss = criterion(result.squeeze(1), target.type(torch.float32))
        # l2_loss : cls_tokens 与 mod_cls_tokens
        l2_loss_items = []
        for item in zip(cls_tokens, mod_cls_tokens):
            l2_loss_items.append(torch.mean((item[0] - item[1]) ** 2))
        l2_loss = sum(l2_loss_items)
        # return self.lambdas[0] * main_loss + self.lambdas[1] * l2_loss + new_loss
        return self.lambdas[1] * l2_loss + new_loss

    def train_epoch(self, dataloader: DataLoader, criterion):
        total_loss = 0.0
        total_batches = 0

        running_loss = 0.0
        batch_number = 0

        self.model.to(self.device)
        self.model.train()

        for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
            data, target = data.to(self.device), target.to(self.device)
            mod_data = self.shuffle_patches(data, self.model.vision_patch_size)
            self.optimizer.zero_grad()

            with autocast():
                loss = self.setting3(data, mod_data, target, criterion)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            total_loss += loss.item()
            batch_number += 1
            total_batches += 1

        return total_loss / (total_batches + 1)

    def validate_epoch(self, dataloader: DataLoader, criterion, epoch: int, writer: SummaryWriter = None):
        self.model.to(self.device)
        self.model.eval()
        running_loss = 0.0
        dataset_preds = []
        dataset_targets = []

        for data, target in tqdm(dataloader):
            data, target = data.to(self.device), target.to(self.device)

            with torch.no_grad():
                with autocast():
                    pre, _, _, _ = self.model(data)
                    loss = criterion(pre.squeeze(1), target.type(torch.float32))
                    running_loss += loss.item()
                    pre_prob = pre.cpu().numpy()
                    target = target.cpu().numpy()
                    dataset_preds.append(pre_prob)
                    dataset_targets.append(target)
        dataset_preds = np.concatenate(dataset_preds)
        dataset_targets = np.concatenate(dataset_targets)

        acc = accuracy_score(dataset_targets, dataset_preds > 0)
        ap = average_precision_score(dataset_targets, dataset_preds)

        if writer is not None:
            writer.add_scalar('Loss/Validation', running_loss / len(dataloader), epoch)
            writer.add_scalar('Accuracy', acc, epoch)
            writer.add_scalar('Average Precision', ap, epoch)

        return running_loss / len(dataloader), acc, ap

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, criterion, num_epochs: int,
              checkpoint_dir: str = None, writer: SummaryWriter = None):
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            print(f"Training" + "-" * 60)
            time.sleep(1)
            train_loss = self.train_epoch(train_dataloader, criterion)
            print(f"Validating" + "*" * 60)
            time.sleep(1)
            val_loss, acc, ap = self.validate_epoch(val_dataloader, criterion, epoch, writer=writer)

            print(
                f'{time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())}\nTrain Epoch: {epoch+1}: \n'
                f'train loss: {train_loss}\nval_loss:{val_loss}\nacc:{acc}\nap:{ap}')

            os.makedirs(checkpoint_dir, exist_ok=True)
            if (epoch+1) % 1 == 0:
                checkpoint_path_1 = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                }, checkpoint_path_1)
                print(f'Model checkpoint saved to {checkpoint_path_1}')

            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss and checkpoint_dir is not None:
                best_val_loss = val_loss
                checkpoint_path_2 = os.path.join(checkpoint_dir, f'intermediate_model_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                }, checkpoint_path_2)
                print(f'Model checkpoint saved to {checkpoint_path_2}')

            self.scheduler.step()

        print('Training complete.')
