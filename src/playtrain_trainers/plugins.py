"""Optional env-provider seam for the trainers.

The trainers are environment-agnostic: they reach environments through
``playtrain.runtime``. Two capabilities are *not* generic and belong to the
consumer:

  * binding-level generalization / seed pools (``resolve_pools``,
    ``SeedSetWrapper``) — these know about a specific game's seed bindings, and
  * backends other than PlayTrain, such as MiniGrid (``make_minigrid_env``).

A consumer repo supplies these by registering a
provider once at import/startup::

    from playtrain_trainers import plugins
    from myrepo import generalization, minigrid_env

    class MyProvider:
        resolve_pools = staticmethod(generalization.resolve_pools)
        SeedSetWrapper = generalization.SeedSetWrapper
        make_minigrid_env = staticmethod(minigrid_env.make_minigrid_env)

    plugins.register_provider(MyProvider())

When no provider is registered, ``resolve_pools`` returns ``None`` (single
fixed env — the default training path), and the two backend-specific helpers
raise a clear error naming the missing provider. This keeps the trainers fully
usable standalone against plain PlayTrain envs, with generalization and MiniGrid
as opt-in extras.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class EnvProvider(Protocol):
    """What a consumer must supply to enable generalization / extra backends."""

    def resolve_pools(self, *args: Any, **kwargs: Any) -> Any: ...
    def SeedSetWrapper(self, *args: Any, **kwargs: Any) -> Any: ...
    def make_minigrid_env(self, *args: Any, **kwargs: Any) -> Any: ...


_provider: Optional[EnvProvider] = None


def register_provider(provider: EnvProvider) -> None:
    """Install the consumer's env provider (see module docstring)."""
    global _provider
    _provider = provider


def get_provider() -> Optional[EnvProvider]:
    return _provider


def _require(name: str) -> EnvProvider:
    if _provider is None:
        raise RuntimeError(
            f"playtrain_trainers.plugins.{name}() needs an env provider, but none is "
            "registered. This code path (generalization pools / MiniGrid "
            "backend) is provided by the consumer; call "
            "playtrain_trainers.plugins.register_provider(...) at startup. Plain "
            "PlayTrain training does not require a provider."
        )
    return _provider


def resolve_pools(*args: Any, **kwargs: Any) -> Any:
    """Resolve generalization seed pools. Returns None when no provider is set
    (i.e. no generalization — single fixed env, the default)."""
    if _provider is None:
        return None
    return _provider.resolve_pools(*args, **kwargs)


def SeedSetWrapper(*args: Any, **kwargs: Any) -> Any:
    return _require("SeedSetWrapper").SeedSetWrapper(*args, **kwargs)


def make_minigrid_env(*args: Any, **kwargs: Any) -> Any:
    return _require("make_minigrid_env").make_minigrid_env(*args, **kwargs)
