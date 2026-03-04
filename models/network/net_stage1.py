import torch
import torch.nn as nn
import torch.nn.functional as F

from models.network.clip import clip


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class net_stage1(nn.Module):
    def __init__(self, dim=768, drop_rate=0.5, output_dim=1):
        super(net_stage1, self).__init__()

        # lode the frozen CLIP-ViT with wsgm trainable
        self.backbone, self.vision_patch_size, self.vision_width, self.embed_dim, _ = clip.load('ViT-L/14', device='cpu')
        params = []
        for name, p in self.backbone.named_parameters():
            if ("Adapter_modules" in name and "visual" in name) or name == "fc.weight" or name == "fc.bias":
                params.append(name)
            else:
                p.requires_grad = False
        print(params)

        self.ln_post = LayerNorm(dim)

        self.fc = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(dim, output_dim)
        )

        # new
        scale = self.vision_width ** -0.5

        self.projs = nn.ParameterList([
            nn.Parameter(scale * torch.randn(self.vision_width, self.embed_dim))
            for _ in range(3)
        ])

        self.fcs = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(drop_rate),
                nn.Linear(dim, output_dim)
            )
            for _ in range(3)
        ])

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 1)
        self.fc3 = nn.Linear(3, 1)


    def forward(self, x, mod_x=None):
        return self.setting3(x, mod_x)

    def setting1(self, x, mod_x=None):
        cls_tokens, mod_cls_tokens = [], []
        differ_cls_tokens = []

        proj_result, tokens = self.backbone.encode_image(x)
        if mod_x is not None:
            mod_proj_result, mod_tokens = self.backbone.encode_image(mod_x)

            keys = list(tokens.keys())
            for idx in range(len(keys)):
                cls_tokens.append(tokens[keys[idx]])
                mod_cls_tokens.append(mod_tokens[keys[idx]])
                differ_cls_token = tokens[keys[idx]] - mod_tokens[keys[idx]]
                
                differ_cls_tokens.append(differ_cls_token)

            result = self.fc(proj_result - mod_proj_result)
        else:
            result = self.fc(proj_result)
        
        return result, differ_cls_tokens, cls_tokens, mod_cls_tokens
    
    def setting2(self, x, mod_x=None):
        cls_tokens, mod_cls_tokens = [], []
        differ_cls_tokens, logits_by_differ_cls_tokens = [], []

        last_token, tokens = self.backbone.encode_image(x)
        if mod_x is not None:
            mod_last_token, mod_tokens = self.backbone.encode_image(mod_x)

            keys = list(tokens.keys())
            for idx in range(len(keys)):
                cls_tokens.append(tokens[keys[idx]])
                mod_cls_tokens.append(mod_tokens[keys[idx]])
                differ_cls_token = tokens[keys[idx]] - mod_tokens[keys[idx]]
                
                differ_cls_tokens.append(differ_cls_token)
                logits_by_differ_cls_tokens.append(
                    self.fcs[idx](
                        differ_cls_token @ self.projs[idx]
                    )
                )

            result = logits_by_differ_cls_tokens[-1]
        else:
            result = self.fcs[-1](
                last_token @ self.projs[-1]
            )
        
        return result, logits_by_differ_cls_tokens, cls_tokens, mod_cls_tokens

    def setting3(self, x, mod_x=None):
        cls_tokens, mod_cls_tokens = [], []
        differ_cls_tokens, logits_by_differ_cls_tokens = [], []
        if mod_x is not None:
            last_token, tokens = self.backbone.encode_image(x)
            mod_last_token, mod_tokens = self.backbone.encode_image(mod_x)

            keys = list(tokens.keys())
            for idx in range(len(keys)):
                cls_tokens.append(tokens[keys[idx]])
                mod_cls_tokens.append(mod_tokens[keys[idx]])
                differ_cls_token = tokens[keys[idx]] - mod_tokens[keys[idx]]
                
                differ_cls_tokens.append(differ_cls_token)
                logits_by_differ_cls_tokens.append(
                    self.fcs[idx](
                        differ_cls_token @ self.projs[idx]
                    )
                )
            
            stacked = torch.stack(differ_cls_tokens, dim=0)  # num_layers, batch_size, hidden_size

            stacked = F.gelu(self.fc1(stacked))
            stacked = self.fc2(stacked).squeeze().permute(1, 0)
            result = self.fc3(stacked).view(-1).unsqueeze(1)
        else:
            last_token, tokens = self.backbone.encode_image(x)

            cls = []
            keys = list(tokens.keys())
            for idx in range(len(keys)):
                cls.append(tokens[keys[idx]])
            tokens = torch.stack(cls, dim=0)
            tokens = F.gelu(self.fc1(tokens))
            tokens = self.fc2(tokens).squeeze().permute(1, 0)
            result = self.fc3(tokens).view(-1).unsqueeze(1)
        return result, logits_by_differ_cls_tokens, cls_tokens, mod_cls_tokens


