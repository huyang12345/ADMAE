import torch
import torch.nn.functional as F

def gradient(x):
    h, w = x.shape[-2:]
    dx = torch.zeros_like(x)
    dy = torch.zeros_like(x)
    dx[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    dy[..., :, :-1] = x[..., :, 1:] - x[..., :, :-1]
    return dx, dy

def msgms_loss(img1, img2, level_weights=None):
    if level_weights is None:
        level_weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    device = img1.device
    assert img1.size() == img2.size()
    batch_size, num_channels, height, width = img1.size()
    factor = 2.0 ** (len(level_weights) - 1)

    # Gaussian pyramid
    g_img1 = [img1]
    g_img2 = [img2]
    for i in range(1, len(level_weights)):
        img1_down = F.avg_pool2d(g_img1[-1], kernel_size=2, stride=2)
        img2_down = F.avg_pool2d(g_img2[-1], kernel_size=2, stride=2)
        g_img1.append(img1_down)
        g_img2.append(img2_down)

    # Gradient magnitude weight map
    weight_maps = []
    for i, (img1_i, img2_i) in enumerate(zip(g_img1, g_img2)):
        w_i = torch.tensor(level_weights[i]).to(device)
        dx1_i, dy1_i = gradient(img1_i)
        dx2_i, dy2_i = gradient(img2_i)
        gm1_i = torch.sqrt(dx1_i ** 2 + dy1_i ** 2 + 1e-6)
        gm2_i = torch.sqrt(dx2_i ** 2 + dy2_i ** 2 + 1e-6)
        weight_map_i = (2.0 * gm1_i * gm2_i + 1e-6) / (gm1_i ** 2 + gm2_i ** 2 + 1e-6)
        weight_map_i = w_i * weight_map_i
        weight_maps.append(weight_map_i)

    # Multi-scale structural similarity
    cs_map = None
    for i, (img1_i, img2_i, weight_map_i) in enumerate(zip(g_img1, g_img2, weight_maps)):
        img1_var = img1_i - torch.mean(img1_i, dim=(2, 3), keepdim=True)
        img2_var = img2_i - torch.mean(img2_i, dim=(2, 3), keepdim=True)
        img12_cov = torch.mean(img1_var * img2_var, dim=(2, 3), keepdim=True)
        img1_var = torch.mean(img1_var ** 2, dim=(2, 3), keepdim=True)
        img2_var = torch.mean(img2_var ** 2, dim=(2, 3), keepdim=True)
        cs_map_i = (2.0 * img12_cov + 1e-6) / (img1_var + img2_var + 1e-6)
        if cs_map is None:
            cs_map = torch.mean(weight_map_i * cs_map_i)
        else:
            cs_map = cs_map + torch.mean(weight_map_i * cs_map_i)

    return abs(1-cs_map)
# x1 = torch.rand(3,3,244,244)
# x2 = torch.rand(3,3,244,244)
#
# msgms_loss(x1,x2)