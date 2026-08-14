"""CR-Pipeline: Configuration Management.

Provides:
- Schema-based configuration validation
- Configuration templating and inheritance
- Default value resolution
- Type coercion and normalization
- Configuration merging and overrides
- Environment variable substitution
- Configuration versioning
"""

from .validation import (
    ConfigSchema,
    ConfigValidator,
    ConfigurationManager,
    ConfigTemplate,
    TemplateRegistry,
    ValidationRule,
    ValidationType,
    ValidationError,
)

__all__ = [
    "ConfigSchema",
    "ConfigValidator",
    "ConfigurationManager",
    "ConfigTemplate",
    "TemplateRegistry",
    "ValidationRule",
    "ValidationType",
    "ValidationError",
]
