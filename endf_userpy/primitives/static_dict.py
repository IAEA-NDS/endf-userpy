"""Identity-hashed dict subclass used as a stable handle for a parsed
ENDF-6 dict.

Two use cases share the same primitive:

* **Preproc caching.** Reconstruction paths (currently MF6 LAW=1;
  future MF sections likely to follow) marshal a parsed subsection
  into a natural-size dataclass. Adaptive-Simpson mu/Eout
  integrators walk the same subsection many times per top-level
  query; caching the marshaled result across those calls is a
  measurable win. Keying the cache on the wrapper's identity binds
  cache-entry lifetime to the wrapper instance: dropping the
  wrapper drops its cache.

* **JAX static-argnums.** ``jax.jit(f, static_argnums=(0,))`` needs
  its static argument to be hashable so it can key the compiled
  kernel cache. A raw ``dict`` is unhashable; ``StaticEndfDict``
  hashes by identity, so passing the same wrapper across calls
  reuses the compiled kernel and re-wrapping forces a recompile.

Subclassing ``dict`` means the wrapper drops into every existing
call site that expects a parsed endf_dict, with no signature
changes needed downstream. Construction shallow-copies the top
level; nested MF/MT subtrees are shared by reference with the
underlying dict.
"""
from __future__ import annotations


class StaticEndfDict(dict):
    """Identity-hashed handle around a parsed ENDF-6 dict.

    Behaves as a ``dict`` for all read access. Identity-based hash
    and equality make it safe as a cache key and as a
    ``jax.jit`` static argument. Carries an instance-attached
    preproc cache (``_preproc_cache``) so caching state lives and
    dies with the wrapper.

    Wrapping is the caller's explicit contract that the dict's
    contents will not change until they re-wrap. Mutating the
    underlying dict in place breaks that contract silently and can
    return stale cached values; the fix is to re-wrap.

    Note: the top level is shallow-copied at construction. Nested
    subtrees (``self[MF]``, ``self[MF][MT]`` and below) are shared
    by reference with the underlying dict, so an in-place mutation
    like ``d[6][mt]['LCT'] = 2`` is visible in both. Only
    top-level rebinding (``d[NEW_MF] = ...`` after wrapping)
    diverges between the wrapper and the underlying dict; no known
    workflow needs this.
    """

    def __init__(self, endf_dict):
        super().__init__(endf_dict)
        self._preproc_cache: dict = {}

    __hash__ = object.__hash__

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    def __repr__(self):
        return f'StaticEndfDict(id=0x{id(self):x}, keys={list(self.keys())})'


def wrap_endf_dict(endf_dict) -> StaticEndfDict:
    """Return a :class:`StaticEndfDict` wrapping ``endf_dict``.

    Idempotent: if ``endf_dict`` is already a :class:`StaticEndfDict`,
    it is returned unchanged. To force cache invalidation after
    mutating the underlying dict in place, call
    :func:`wrap_endf_dict` on the raw dict again to obtain a fresh
    wrapper.
    """
    if isinstance(endf_dict, StaticEndfDict):
        return endf_dict
    return StaticEndfDict(endf_dict)
