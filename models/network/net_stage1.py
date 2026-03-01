import torch
import torch.nn as nn
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

    def forward(self, x, mod_x=None):
        return self.setting1(x, mod_x)

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


