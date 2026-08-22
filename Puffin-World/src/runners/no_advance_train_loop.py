import logging

from mmengine.logging import print_log
from xtuner.engine.runner import TrainLoop


class _SkipAdvanceIterator:
    """Wrap the real dataloader iterator so the FIRST ``n_skip`` ``next()`` calls
    (mmengine's resume-time "advance dataloader N steps" fast-forward) return
    instantly WITHOUT fetching a batch. Later calls delegate to the real
    iterator, so normal training loads data as usual."""

    def __init__(self, inner, n_skip):
        self._inner = inner
        self._n = int(n_skip)

    def __iter__(self):
        return self

    def __next__(self):
        if self._n > 0:
            self._n -= 1
            return None          # advance step: no data fetch
        return next(self._inner)


class NoAdvanceTrainLoop(TrainLoop):
    """TrainLoop that SKIPS mmengine's resume-time "Advance dataloader N steps"
    fast-forward AND re-seeds the MultiSource sampler so the resumed data is
    FRESH (not a replay of the already-trained samples).

    Why both are needed: on resume mmengine (a) restores the global seed, so the
    sampler reproduces the exact same shuffle, then (b) fetch-and-discards N
    batches to skip past what was already trained. Step (b) is slow and can hang
    for hours on a flaky AOSS dataloader. But if we merely skip (b), the
    reproduced shuffle would REPLAY the same samples from the start. So we also
    shift the sampler seed by the resumed iter -> a different (yet rank-consistent)
    stream -> fresh data, with no fast-forward. ``runner.iter`` is still restored,
    so the LR schedule, optimizer state and DepthTransitionHook ramps continue
    correctly.
    """

    def run(self):
        n_skip = getattr(self, '_iter', 0)
        iterator = getattr(self, 'dataloader_iterator', None)
        if not (n_skip and iterator is not None):
            return super().run()

        reseeded = self._reseed_sampler(n_skip)
        self.dataloader_iterator = _SkipAdvanceIterator(iterator, n_skip)
        print_log(
            f'[NoAdvanceTrainLoop] resume @ iter {n_skip}: skipped the dataloader '
            f'fast-forward; sampler '
            + ('RE-SEEDED -> fresh data.' if reseeded
               else 'NOT re-seeded (resumed data may overlap already-trained data!).')
            + f' Training starts immediately at iter {n_skip}.',
            logger='current', level=logging.WARNING)
        try:
            return super().run()
        finally:
            self.dataloader_iterator = iterator

    def _reseed_sampler(self, shift):
        """Shift the MultiSource sampler's seed by ``shift`` (the resumed iter) and
        rebuild its infinite index generators, so the resumed stream differs from
        the already-trained one. The base seed is synced across ranks and ``shift``
        is identical on every rank, so the new seed stays rank-consistent and the
        per-rank disjoint sharding is preserved."""
        sampler = None
        bs = getattr(self.dataloader, 'batch_sampler', None)
        if bs is not None and getattr(bs, 'sampler', None) is not None:
            sampler = bs.sampler          # MultiSourceBatchSampler -> its .sampler
        else:
            sampler = getattr(self.dataloader, 'sampler', None)

        needed = ('seed', 'source2inds', '_indices_of_rank', 'dataset')
        if sampler is None or not all(hasattr(sampler, a) for a in needed):
            print_log(
                '[NoAdvanceTrainLoop] could not locate a re-seedable MultiSource '
                'sampler; resumed data may overlap already-trained data.',
                logger='current', level=logging.WARNING)
            return False
        try:
            sampler.seed = int(sampler.seed) + int(shift)
            sampler.source2inds = {
                source: sampler._indices_of_rank(len(ds))
                for source, ds in enumerate(sampler.dataset.datasets)
            }
            return True
        except Exception as e:  # noqa: BLE001
            print_log(f'[NoAdvanceTrainLoop] sampler re-seed failed: {e}',
                      logger='current', level=logging.WARNING)
            return False
