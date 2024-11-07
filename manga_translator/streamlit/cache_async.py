from functools import _make_key, lru_cache, _lru_cache_wrapper
import streamlit as st
from typing import TypeVar, overload, Callable, Any, Coroutine
import asyncio
import functools
import inspect
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

R = TypeVar("R")


def _get_session_cache() -> dict[str, Callable]:
    if "#cached_async_functions" not in st.session_state:
        st.session_state["#cached_async_functions"] = {}
    return st.session_state["#cached_async_functions"]


def _hash_function(f: Callable) -> str:
    logger.debug("module: %s", f.__module__)
    logger.debug("file: %s", inspect.getabsfile(f))
    logger.debug("srclines: %s", inspect.getsourcelines(f))
    lines, lineno = inspect.getsourcelines(f)
    h = hash(
        (
            f.__module__,
            inspect.getabsfile(f),
            lineno,
            tuple(lines),
            # inspect.getsource(f),
            # inspect.getsourcefile(f)
        )
    )
    return f"{f.__module__}:{inspect.getabsfile(f)}:{lineno} ${h}"


def cache_async(
    f: Callable[..., Coroutine[Any, Any, R]], cache_size: int = 128
) -> Callable[..., asyncio.Task[R]]:
    """_summary_

    Args:
        f (Callable[..., Coroutine[Any, Any, R]]): an async functoin

    Returns:
        Callable[..., asyncio.Task[R]]: a per (session, callee)
        callee is identified by , therefore mu
    """
    f_id = _hash_function(f)

    def wrapped(*args, **kwargs):
        """the outmost function

        Returns:
            asyncio.Task[R]: maybe-cached asyncio.Task, created by caching calls to f()
        """
        cache = _get_session_cache()
        if f_id not in cache:
            logger.debug("creating lru cache for %s", f_id)

            def run_for_task(*args, **kwargs):
                return asyncio.create_task(f(*args, **kwargs))

            cache[f_id] = functools.lru_cache(maxsize=cache_size, typed=True)(
                run_for_task
            )
        return cache[f_id](*args, **kwargs)
        # nested levels: wrapped >> (lru-ed run_for_task) >> run_for_task >> f

    return wrapped
