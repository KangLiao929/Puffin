from torch import nn
import torch


class ResnetBlock(nn.Module):
    """
    Standard Residual Block to help the network learn deeper spatial features
    while preserving the original signal.
    """
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class CameraConditionEncoder(nn.Module):
    def __init__(self, in_channels=7, out_channels=16, mid_channels=64,
                 num_res_blocks=3, ray_downsampled=False):
        """
        ray_downsampled:
            Default routing flag remembered by the module so callers can use
            `self.cond_fuser(cam, mask)` without re-passing it every time.
            Overridable per-call via the `ray_downsampled` kwarg of `forward`.
        """
        super().__init__()
        self.ray_downsampled = bool(ray_downsampled)
        
        # Path A: Used when input is ORIGINAL resolution (needs 8x downsample).
        # Anti-aliased downsample: Conv3 features then AvgPool2 blurs before
        # the 2x decimation. Replaces the original Conv4/stride2 chain whose
        # integer subsampling grid produced patch-level checkerboard artifacts
        # on high-frequency inputs (ray map directions, perspective field).
        self.downsample_path = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels // 4, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(mid_channels // 4, mid_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(mid_channels // 2, mid_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AvgPool2d(2),
        )
        
        # Path B: Used when input is ALREADY downsampled 8x
        self.direct_path = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.SiLU()
        )
        
        res_layers = []
        for _ in range(num_res_blocks):
            res_layers.append(ResnetBlock(mid_channels))
        
        self.backbone = nn.Sequential(*res_layers)
        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0)
        
        self.init_weights()

    def init_weights(self):
        # 1. Kaiming initialization for all general convolution layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m != self.out_conv:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 2. Final projection: ControlNet-style zero-init so cam contribution
        #    starts at zero and learns via gradient from the cam-to-latent path.
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def forward(self, cam, mask, ray_downsampled=None):
        """Apply cond_fuser to [cam, mask].

        cam:  [C_cam, H, W]   or [N, C_cam, H, W]
        mask: [C_mask, H, W]  or [N, C_mask, H, W]
        Returns matching rank with channel = out_channels (C_lat).

        ray_downsampled:
            None (default) -> use `self.ray_downsampled` set at construction.
            bool           -> override per call.
        """
        if ray_downsampled is None:
            ray_downsampled = self.ray_downsampled

        # Cat cam + mask along the channel dim, handle 3D / 4D.
        if cam.dim() == 3:
            x = torch.cat([cam, mask], dim=0)[None]  # [1, C+M, H, W]
            input_was_3d = True
        else:
            x = torch.cat([cam, mask], dim=1)         # [N, C+M, H, W]
            input_was_3d = False

        # Route the input through the appropriate initial spatial processing path
        if ray_downsampled:
            x = self.direct_path(x)
        else:
            x = self.downsample_path(x)

        # Extract deep features
        x = self.backbone(x)

        # Project to target dimension
        out = self.out_conv(x)
        return out[0] if input_was_3d else out