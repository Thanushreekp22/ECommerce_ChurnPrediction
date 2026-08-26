"""
backend/integrations/platforms/__init__.py

Platform adapters that pull orders from real e-commerce backends and convert
them into the canonical :class:`CanonicalTransaction` stream used by the
normalizer / storage layer, so platform purchases flow into the same
customer-feature and churn-scoring pipeline as CSV imports.
"""

from .shopify import ShopifyAdapter
from .woocommerce import WooCommerceAdapter
from .mock import MockStoreAdapter

__all__ = [
    "ShopifyAdapter",
    "WooCommerceAdapter",
    "MockStoreAdapter",
]