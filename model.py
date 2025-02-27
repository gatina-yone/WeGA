import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.hub.set_dir('data_path')
device = torch.device("cuda:")


class BasicBlock(nn.Module):
    def __init__(self, in_channel, s):
        super(BasicBlock, self).__init__()
        self.s = s
        self.conv1 = nn.Conv2d(in_channel, in_channel * s, kernel_size=3, stride=s, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel * s)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channel * s, in_channel * s, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channel * s)
        if self.s == 2:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, in_channel * s, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(in_channel * s)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.s == 2: 
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        )if project_out else nn.Identity()

    def forward(self, x_q, x_kv):
        b, seq_len_q, dim = x_q.size()  # seq_len_q = n
        _, seq_len_kv, _ = x_kv.size()

        qkv = self.to_qkv(torch.cat([x_q, x_kv], dim = 1))
        q, k, v = qkv.split(self.inner_dim, dim = -1)
        q = q[:,  :seq_len_q].reshape(b, seq_len_q, self.heads, self.dim_head).transpose(1,2)  # q:n, len(q+korv), inner_dim
        k = k[:, seq_len_q: ].reshape(b, seq_len_kv, self.heads, self.dim_head).transpose(1,2)
        v = v[:, seq_len_q: ].reshape(b, seq_len_kv, self.heads, self.dim_head).transpose(1,2)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, seq_len_q, self.inner_dim)
        return self.to_out(out)
    
class CrossVitBlock(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x1, x2):
        for attn, ff in self.layers:
            x = attn(x1, x2) + x1
            x = ff(x) + x
    
        return self.norm(x)
    
class CrossVit(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim,  dim_head = 8, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.align_scale = nn.Sequential(
            nn.Linear(384, dim),           
            nn.LayerNorm(dim)
        )
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = CrossVitBlock(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = 'cls'

    def forward(self, x1, x2):  # x2: 1*65*384
        x2 = self.align_scale(x2)

        x = self.transformer(x1, x2)

        return x

class Aug_network(nn.Module):
    def __init__(self, n_class=1, zero_init_residual=True):
        super(Aug_network, self).__init__()
        self.dinov2_vits14 = torch.hub.load('', 'dinov2_vits14', source='local')

        self.classifier = nn.Linear(384, 2)
        self.relu = nn.ReLU(inplace = True)

        self.fc_imgProb = nn.Linear(2, 1)
        self.sigmoid_img = nn.Sigmoid()

        self.fc_Combine = nn.Linear(8, 4) 
        self.fc_combine_2 = nn.Linear(4, n_class)  
        self.sigmoid_combined = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')   
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):   
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, patch_all, target_layers):
        layer_outputs = []

        x = self.dinov2_vits14.prepare_tokens_with_masks(patch_all)
        for i, blk in enumerate(self.dinov2_vits14.blocks):
            x = blk(x)
            if i in target_layers:
                layer_outputs.append(x)
        
        x_norm = self.dinov2_vits14.norm(x)
        cls_tokens = x_norm[:, 0]
        x = self.classifier(cls_tokens)
        x = self.relu(x)

        img_pred = self.fc_imgProb(x)
        img_pred = self.sigmoid_img(img_pred)

        return img_pred, layer_outputs

class Main_network(nn.Module):
    def __init__(self, n_class=1, zero_init_residual=True):
        super(Main_network, self).__init__()
        self.classifier1 = nn.Linear(768, 2)

        self.classifier2 = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(768, 2, kernel_size=1)
        )

        self.cls = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 2, kernel_size=1)
        )
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=1) 
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            BasicBlock(in_channel=64, s=1), 
            BasicBlock(in_channel=64, s=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(in_channel=64, s=2),
            BasicBlock(in_channel=128, s=1),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(in_channel=128, s=2),
            BasicBlock(in_channel=256, s=1),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(in_channel=256, s=1),
            BasicBlock(in_channel=256, s=1),
        )
        self.conv_input1 = nn.Conv2d(in_channels=64, out_channels=48, kernel_size=1)
        self.conv_output1 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=1)
        self.align_scale_2 = nn.Sequential(
            nn.Linear(256, 768),           
            nn.LayerNorm(768)
        )

        self.linear_layer = nn.Linear(768, 48)
        self.deconv_layer = nn.ConvTranspose2d(in_channels=48, out_channels=48, kernel_size=4, stride=4)
        self.crossvit_block = nn.ModuleList([
            CrossVit(768, 3, 12, 768, 64, 0., 0.),
            CrossVit(768, 3, 12, 768, 64, 0., 0.),
            CrossVit(768, 6, 12, 768, 64, 0., 0.)
        ])
        self.relu = nn.ReLU(inplace = True)
        self.fc_imgProb = nn.Linear(2, 1)
        self.sigmoid_img = nn.Sigmoid()

        self.fc_Combine = nn.Linear(6, 4) 
        self.fc_combine_2 = nn.Linear(4, n_class)
        self.sigmoid_combined = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')   
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):   
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, patch, patch_LD, patch_SD, patch_RD, patch_adc, block_idx, aux_features):
        x = self.conv1(patch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.conv_input1(x)
        x = self.embedding(x, 4)
        x = self.crossvit_block[0](x , aux_features[0])
        x = self.unembedding1(x, 8, 8)
        x = self.conv_output1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feas_88 = x
        logits_map = self.cls(feas_88)

        x = self.embedding(x, 1)
        x = self.align_scale_2(x)
        x = self.crossvit_block[2](x, aux_features[2])
        
        cls_tokens = x[:, 0]

        # for i, blk in enumerate(self.dinov2_vitb14.blocks):
        #     x = blk(x)
        #     if i in block_idx:
        #         x = self.crossvit_block[idx](x, aux_features[idx])
        #         idx += 1
        
        # x_norm = self.dinov2_vitb14.norm(x)
        # cls_tokens = x_norm[:, 0]      # 1*768
        # patch_token = x_norm[:, 1 :]   # 1*64*768

        # logits_map = patch_token.permute(0, 2, 1).reshape(1, 768, 8, 8)
        # logits_map = self.classifier2(logits_map)      

        img_fea = self.classifier1(cls_tokens)
        img_pred = self.fc_imgProb(img_fea)
        img_pred = self.sigmoid_img(img_pred)

        combined_fea = torch.cat((img_fea[0], patch_LD, patch_SD, patch_RD, patch_adc))
        combined_fea = self.fc_Combine(combined_fea)
        combined_fea = self.relu(combined_fea)  # RELU
        combined_pred = self.fc_combine_2(combined_fea)
        combined_pred = self.sigmoid_combined(combined_pred)
        combined_pred = combined_pred.unsqueeze(0)

        return img_pred, combined_pred, logits_map
    
    def embedding(self, input_tensor, patch_size):

        batch_size, channels, height, width = input_tensor.shape
        num_patches_y = height // patch_size
        num_patches_x = width // patch_size

        patches = input_tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(batch_size, channels, num_patches_y, num_patches_x, patch_size, patch_size)

        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, num_patches_y * num_patches_x, -1)

        cls_token = torch.zeros((batch_size, 1, patches.shape[2]), device=input_tensor.device)
        output_tensor = torch.cat((cls_token, patches), dim=1)

        return output_tensor
    

    def unembedding1(self, output_tensor, num_patches_y, num_patches_x):
        output_tensor = output_tensor[:, 1:, :] 
        output_tensor = self.linear_layer(output_tensor)  

        output_tensor = output_tensor.view(1, 64, 48)
        
        output_tensor = output_tensor.permute(0, 2, 1)
        output_tensor = output_tensor.view(1, 48, num_patches_y, num_patches_x) 

        output_tensor = self.deconv_layer(output_tensor) 

        return output_tensor
    
    def unembedding2(self, output_tensor, num_patches_y, num_patches_x):
        output_tensor = output_tensor[:, 1:, :] 
        linear_layer = nn.Linear(768, 48) 
        output_tensor = linear_layer(output_tensor) 

        output_tensor = output_tensor.view(1, 64, 48)
        
        output_tensor = output_tensor.permute(0, 2, 1) 
        output_tensor = output_tensor.view(1, 48, num_patches_y, num_patches_x) 

        deconv_layer = nn.ConvTranspose2d(in_channels=48, out_channels=48, kernel_size=4, stride=4)
        output_tensor = deconv_layer(output_tensor) 

        return output_tensor

class Combined_network(nn.Module):
    def __init__(self, num_class):
        super(Combined_network, self).__init__()
        self.main_network = Main_network(num_class)
        self.auxiliary_network = Aug_network(num_class)
        
    def forward(self, current_patch, remaining_iamge, p_patch_LD, p_patch_SD, p_patch_RD, p_patch_adc):
        [auxiliary_output, aux_features] = self.auxiliary_network(remaining_iamge, [1, 5, 9])
        [img_pred, combined_pred, cls_map] = self.main_network_0214(current_patch, p_patch_LD, p_patch_SD, p_patch_RD, p_patch_adc, [1, 5, 9], aux_features)
        return img_pred, combined_pred, cls_map, auxiliary_output

    def get_ra_loss(self, logits, label, th_bg=0.31, bg_fg_gap=0.1):
        n, _, _, _ = logits.size()
        cls_logits = torch.softmax(logits, dim=1)
        var_logits = torch.var(cls_logits, dim=1)
        norm_var_logits = self.normalize_feat(var_logits)

        bg_mask = (norm_var_logits < th_bg).float()
        fg_mask = (norm_var_logits > (th_bg + bg_fg_gap)).float()
        cls_map = logits[torch.arange(n), label.long(), ...]
        cls_map = torch.sigmoid(cls_map)

        ra_loss = torch.mean(cls_map * bg_mask + (1 - cls_map) * fg_mask)
        return ra_loss

    def normalize_feat(self,feat):
        n, fh, fw = feat.size()
        feat = feat.view(n, -1)
        min_val, _ = torch.min(feat, dim=-1, keepdim=True)
        max_val, _ = torch.max(feat, dim=-1, keepdim=True)
        norm_feat = (feat - min_val) / (max_val - min_val + 1e-15)
        norm_feat = norm_feat.view(n, fh, fw)

        return norm_feat

