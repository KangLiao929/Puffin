from mmengine.hooks import Hook


class DepthTransitionHook(Hook):
    """Stage-4 RGB-preserving depth-transition scheduler.

    Each train iteration this updates two runtime scalars on the Puffin model
    from the current iteration, so the model forward / loss can ramp the new
    depth pathway in WITHOUT disturbing the warm-started stage-3 RGB:

    - ``model.depth_loss_weight``: weight on the depth-view flow-matching loss,
      ramped 0 -> ``model.depth_loss_weight_max`` over
      ``model.depth_loss_warmup_iters`` micro-steps (only when
      ``model.depth_loss_split`` is on). Starts at 0 so the FIRST backward step
      puts no depth gradient into the shared DiT head (no yellow-tint drift).

    - ``model.transformer.depth_attn_closed``: when True, RGB/text queries are
      blocked from attending to depth-visual keys; stochastically annealed open
      over ``model.depth_attn_open_iters`` micro-steps (only when
      ``model.rgb_depth_attn_isolation`` is on). Closed at iter 0 so step-0 RGB
      attention is identical to stage-3.

    All schedules are in MICRO-steps (``runner.iter``), matching ``warmup_iters``.
    Completely inert for models/configs that do not enable the switches, so old
    experiments are unaffected.
    """

    priority = 'NORMAL'

    def _unwrap(self, runner):
        # unwrap DDP / DeepSpeed / etc. until we reach the Puffin model that
        # carries the transition flags.
        model = runner.model
        while not hasattr(model, 'depth_loss_split') and hasattr(model, 'module'):
            model = model.module
        return model

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        import random
        model = self._unwrap(runner)
        it = runner.iter

        if getattr(model, 'depth_loss_split', False):
            wmax = float(getattr(model, 'depth_loss_weight_max', 1.0))
            wi = int(getattr(model, 'depth_loss_warmup_iters', 0))
            model.depth_loss_weight = wmax * min(1.0, it / wi) if wi > 0 else wmax

        if getattr(model, 'rgb_depth_attn_isolation', False):
            oi = int(getattr(model, 'depth_attn_open_iters', 0))
            if oi < 0:
                # -1 = PERMANENT asymmetric attention: RGB never attends to
                # depth (depth always attends to RGB). Kills the depth->RGB
                # color-leak channel for good; pair it with
                # model.depth_attn_closed_infer=True (or --depth_attn_closed)
                # so inference matches training.
                open_frac = 0.0
            elif oi == 0:
                open_frac = 1.0
            else:
                open_frac = min(1.0, it / oi)
            # rank-consistent stochastic anneal: closed with prob (1 - open_frac).
            # iter-seeded RNG => every rank picks the same state for a given iter.
            closed = random.Random(it).random() >= open_frac
            model.transformer.depth_attn_closed = bool(closed)
            model.depth_attn_open = float(open_frac)
