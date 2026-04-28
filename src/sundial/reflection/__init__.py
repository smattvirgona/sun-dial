"""Reflection synthesis — chart facts + retrieved corpus → headline + body.

Uses the Anthropic SDK with prompt caching (system + voice guide cached;
chart facts and corpus snippets in the cached prefix; question variable).
"""
from .prompt import build_messages, build_system
from .synthesize import synthesize
from .types import Citation, Reflection, ReflectionRequest

__all__ = [
    "build_messages",
    "build_system",
    "synthesize",
    "Citation",
    "Reflection",
    "ReflectionRequest",
]
