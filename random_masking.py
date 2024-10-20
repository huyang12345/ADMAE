import torch

def forward_decoder(x, ids_restore):
    # embed tokens
    # x N*K,L/K=l,D
    x = self.decoder_embed(x)
    NK,l,D =x.shape
    N,L = ids_restore.shape
    K = int(NK/N)
    # append mask tokens to sequence
    mask_token = torch.zeros(1,1,D)
    mask_tokens = mask_token.repeat(N, L + 1 - l, 1) # x里面还包括一个位置编码
    x_ = x.reshape(K,N,l,D)
    x_1 = torch.zeros(K,N,L,D)
    step = int(L/K)
    for i in range(K):
        x_ = torch.cat((mask_tokens[:,0:i*step,:],x_[i][:, :, :], mask_tokens[:,0:(K-i)*step,:]), dim=1)  # no cls token
        x_1[i] = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    K = int(1/(1-mask_ratio))
    x_masked = torch.zeros([K,N,int(L/K),D],device=x.device)
    mask = torch.zeros([K,N,L],device=x.device)

    for i in range(K):
        step = int(L/K)
        ids = ids_shuffle[:, i*step:(i+1)*step]
        x_masked[i] = torch.gather(x, dim=1, index=ids.unsqueeze(-1).repeat(1, 1, D))  # select patch
        # generate the binary mask: 0 is keep, 1 is remove
        m_i = torch.ones(N,L)
        m_i[:,i*step:(i+1)*step] = 0
        mask[i] = torch.gather(m_i,dim=1,index=ids_restore)
        #ids_tmp[i] = ids_shuffle[:, i*step:(i+1)*step]
    mask = mask.reshape(-1,L)
    x_masked = x_masked.reshape(-1,int(L/K),D)
    #ids_tmp = ids_tmp.reshape(-1,L)
    #ids_restore = torch.argsort(ids_tmp, dim=1)
    # # keep the first subset
    # ids_keep = ids_shuffle[:, :len_keep]
    # x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    #
    # # generate the binary mask: 0 is keep, 1 is remove
    # mask = torch.ones([N, L], device=x.device)
    # mask[:, :len_keep] = 0
    # # unshuffle to get the binary mask
    # mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore
x=torch.rand(2,4,4)
x_masked, mask, ids_restore = random_masking(x, 0.75)
print('x_masked shape:{}, mask shape:{}, ids_restore shape:{}'.format(x_masked.shape, mask.shape, ids_restore.shape ))