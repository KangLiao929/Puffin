from torch.optim import AdamW

class CustomAdamW(AdamW):
    def __init__(self, params, weight_decay, *args, **kwargs):
        # Normalize to a list (params may be a dict of name->param, a generator
        # of params, or — with paramwise_cfg — a list of param-group dicts).
        if isinstance(params, dict):
            params = list(params.values())
        else:
            params = list(params)

        # paramwise_cfg path: `params` is a list of param-GROUP dicts, each
        # possibly carrying its own 'lr' (from custom_keys lr_mult). Preserve
        # every group's extra keys (lr!), filter frozen params, and split each
        # group into decay (>=2D) / no-decay (<2D) sub-groups so weight decay
        # still applies WITHOUT discarding the per-group lr.
        if len(params) > 0 and isinstance(params[0], dict):
            optim_groups = []
            for group in params:
                extra = {k: v for k, v in group.items()
                         if k not in ('params', 'weight_decay')}
                g_params = [p for p in group['params'] if p.requires_grad]
                decay = [p for p in g_params if p.dim() >= 2]
                nodecay = [p for p in g_params if p.dim() < 2]
                if decay:
                    optim_groups.append({'params': decay, 'weight_decay': weight_decay, **extra})
                if nodecay:
                    optim_groups.append({'params': nodecay, 'weight_decay': 0.0, **extra})

            # Summary by lr so the paramwise lr_mult is verifiable in the log.
            from collections import defaultdict
            by_lr = defaultdict(lambda: [0, 0])  # lr -> [n_tensors, n_params]
            for g in optim_groups:
                lr = g.get('lr', kwargs.get('lr'))
                by_lr[lr][0] += len(g['params'])
                by_lr[lr][1] += sum(p.numel() for p in g['params'])
            for lr in sorted(by_lr, key=lambda x: (x is None, x)):
                n_t, n_p = by_lr[lr]
                print(f"[CustomAdamW] lr={lr}: {n_t} tensors, {n_p:,} params")

            super().__init__(params=optim_groups, *args, **kwargs)
            return

        # flat path (no paramwise_cfg): original behavior.
        params = [p for p in params if p.requires_grad]
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for p in params if p.dim() >= 2]
        nodecay_params = [p for p in params if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        super().__init__(params=optim_groups, *args, **kwargs)


class ParamWiseAdamW(AdamW):
    def __init__(self, params, *args, **kwargs):
        assert isinstance(params, list)
        for param in params:
            assert isinstance(param, dict)
            assert isinstance(param['params'], list)
            assert len(param['params']) == 1

            if param['params'][0].ndim == 1:
                param['weight_decay'] = 0.0

        super().__init__(params=params, *args, **kwargs)
