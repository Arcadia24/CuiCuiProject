import torch
import torch.nn as nn

class PathEmbed(nn.Module):
    def __init__(self, img_size : int, patch_size : int, in_chans : int = 3, embed_dim : int = 768) -> None:
        """Patch Embedding Module

        Args:
            img_size (int): Image size 
            patch_size (int): Image size
            in_chans (int, optional): number of channels of the image. Defaults to 3.
            embed_dim (int, optional): Imbedding dimension. Defaults to 768.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size= patch_size,
            stride = patch_size
        )
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """Run forward pass

        Args:
            x (torch.Tensor): shape (n_samples, in_chans, img_size, img_size)

        Returns:
            torch.Tensor: shape (n_samples, n_patches, embed_dim)
        """
        x = self.proj(x) # (n_smaples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2) # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2) # (n_samples, n_patches, embed_dim)
        
        return x
    
    
class Attention(nn.Module):
    def __init__(self, dim : int, n_heads : int = 12, qkv_bias : bool = True, attn_p : float = 0., proj_p : float = 0.) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """Run forward pass

        Args:
            x (torch.Tensor): shape (n_samples, n_patches + 1, dim)
            +1 for class token
        Returns:
            torch.Tensor: shape (n_samples, n_patches + 1, dim)
        """
        n_samples, n_tokens, dim = x.shape
        
        if dim != self.dim:
            raise ValueError
        
        qkv = self.qkv(x) #shape -> (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim 
        ) #sahpe -> (n_samples, n_patches + 1, 3, head_dim)
        qkv = qkv.permute( 2, 0, 3, 1, 4) #shape-> (3, n_samples, n_heads, n_patches + 1, head_dim)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        #dot product between Q and K
        k_t = k.transpose(-2, -1)
        dp = ( q @ k_t) * self.scale
        attn = dp.softmax(dim = 1) #softmax to get probability
        attn = self.attn_drop(attn)
        
        #dot product between QK and V
        weighted_avg = attn @ v
        weighted_avg = weighted_avg.transpose(
            1, 2
        )
        weighted_avg = weighted_avg.flatten(2) #concat
        x = self.proj(weighted_avg)
        x = self.proj_drop(x)
        
        return x
    
class MLP(nn.Module):
    def __init__(self, in_features : int, hidden_feature : int, out_features : int, p : float= 0.) -> None:
        """MLP

        Args:
            in_features (int): In dim
            hidden_feature (int): hidden layer
            out_features (int): out dim
            p (float, optional): probability of dropout. Defaults to 0..
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_feature),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden_feature, out_features),
            nn.Dropout(p)
        )
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
        
class Block(nn.Module):
    def __init__(self, dim : int, n_heads : int, mlp_ratio : float = 4.0, qkv_bias : bool = True, p : float = .0, attn_p : float = .0) -> None:
        super().__init__()
        self.res = nn.Sequential(
            nn.LayerNorm(dim, eps = 1e-6),
            Attention(
                dim,
                n_heads= n_heads,
                qkv_bias= qkv_bias,
                attn_p= attn_p,
                proj_p= p
            )
        )
        self.addn = nn.Sequential(
            nn.LayerNorm(dim, eps = 1e-6),
            MLP(
                in_features= dim,
                hidden_feature= int(dim * mlp_ratio),
                out_features= dim
            )
        )
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): shape (n_samples, n_patches + 1, dim)

        Returns:
            torch.Tensor: shape (n_samples, n_patches + 1, dim)
        """
        x = x + self.res(x)
        x = x + self.addn(x)
        
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self,
                img_size : int = 384,
                patch_size : int = 16,
                in_chans : int = 3,
                n_classes : int = 1000,
                embed_dim : int = 768,
                depth : int = 12,
                n_heads : int = 12,
                mlp_ratio : float = 4.,
                qkv_bias : bool = True,
                p : float = 0.,
                attn_p : float = 0.
                ) -> None:
        super().__init__()
        
        self.patch_embed = PathEmbed(
            img_size= img_size,
            patch_size= patch_size,
            in_chans= in_chans,
            embed_dim= embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim= embed_dim,
                    n_heads= n_heads,
                    mlp_ratio= mlp_ratio,
                    qkv_bias= qkv_bias,
                    p= p,
                    attn_p= attn_p, 
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps = 1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """Run forward pass

        Args:
            x (torch.Tensor): shape(n_samples, in_chans, img_size, img_size)

        Returns:
            torch.Tensor: shape(n_samples, n_classes)
        """
        n_samples = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(
            n_samples, -1, -1
        )
        x = torch.cat((cls_token, x), dim = 1)
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        cls_token_final = x[:,0]
        x = self.head(cls_token_final)
        
        return x