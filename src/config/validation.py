"""Configuration validation and templating for CR-Pipeline.

Provides:
- Schema-based configuration validation
- Configuration templating and inheritance
- Default value resolution
- Type coercion and normalization
- Configuration merging and overrides
- Environment variable substitution
- Configuration versioning
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Validation Types
# =============================================================================


class ValidationType(Enum):
    """Supported validation types."""
    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
    BOOLEAN = auto()
    LIST = auto()
    DICT = auto()
    ENUM = auto()
    RANGE = auto()
    PATTERN = auto()
    PATH = auto()
    URL = auto()
    EMAIL = auto()
    JSON = auto()


@dataclass
class ValidationError:
    """Represents a validation error."""
    field: str
    message: str
    value: Any = None
    severity: str = "error"  # error, warning, info

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "field": self.field,
            "message": self.message,
            "value": str(self.value) if self.value is not None else None,
            "severity": self.severity,
        }


@dataclass
class ValidationRule:
    """A single validation rule."""
    field: str
    vtype: ValidationType
    required: bool = False
    default: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    enum_values: Optional[List[str]] = None
    pattern: Optional[str] = None
    path_exists: bool = False
    transform: Optional[Callable] = None
    message: str = ""

    def validate(self, value: Any) -> List[ValidationError]:
        """Validate a value against this rule.

        Args:
            value: Value to validate.

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []

        # Check required
        if value is None:
            if self.required:
                errors.append(ValidationError(
                    field=self.field,
                    message=f"Required field '{self.field}' is missing",
                    severity="error",
                ))
            elif self.default is not None:
                return []  # Use default
            else:
                errors.append(ValidationError(
                    field=self.field,
                    message=f"Field '{self.field}' is None",
                    value=value,
                    severity="warning",
                ))
            return errors

        # Apply transform if specified
        if self.transform:
            try:
                value = self.transform(value)
            except Exception as e:
                errors.append(ValidationError(
                    field=self.field,
                    message=f"Transform failed: {e}",
                    value=value,
                    severity="error",
                ))
                return errors

        # Type validation
        if self.vtype == ValidationType.INTEGER:
            if not isinstance(value, int):
                errors.append(ValidationError(
                    field=self.field,
                    message=f"Expected integer, got {type(value).__name__}",
                    value=value,
                    severity="error",
                ))

        elif self.vtype == ValidationType.FLOAT:
            try:
                value = float(value)
            except (TypeError, ValueError):
                errors.append(ValidationError(
                    field=self.field,
                    message=f"Cannot convert to float: {value}",
                    value=value,
                    severity="error",
                ))

        elif self.vtype == ValidationType.BOOLEAN:
            if isinstance(value, str):
                value = value.lower() in ("true", "1", "yes", "y")
            elif not isinstance(value, bool):
                errors.append(ValidationError(
                    field=self.field,
                    message=f"Expected boolean, got {type(value).__name__}",
                    value=value,
                    severity="error",
                ))

        elif self.vtype == ValidationType.STRING:
            if not isinstance(value, str):
                errors.append(ValidationError(
                    field=self.field,
                    message=f"Expected string, got {type(value).__name__}",
                    value=value,
                    severity="error",
                ))

        elif self.vtype == ValidationType.ENUM:
            if self.enum_values and value not in self.enum_values:
                errors.append(ValidationError(
                    field=self.field,
                    message=f"Value '{value}' not in allowed values: {self.enum_values}",
                    value=value,
                    severity="error",
                ))

        elif self.vtype == ValidationType.RANGE:
            if self.min_value is not None and value < self.min_value:
                errors.append(ValidationError(
                    field=self.field,
                    message=f"Value {value} below minimum {self.min_value}",
                    value=value,
                    severity="error",
                ))
            if self.max_value is not None and value > self.max_value:
                errors.append(ValidationError(
                    field=self.field,
                    message=f"Value {value} above maximum {self.max_value}",
                    value=value,
                    severity="error",
                ))

        elif self.vtype == ValidationType.PATTERN:
            if self.pattern and not re.match(self.pattern, str(value)):
                errors.append(ValidationError(
                    field=self.field,
                    message=f"Value '{value}' doesn't match pattern '{self.pattern}'",
                    value=value,
                    severity="error",
                ))

        elif self.vtype == ValidationType.PATH:
            if self.path_exists and not Path(str(value)).exists():
                errors.append(ValidationError(
                    field=self.field,
                    message=f"Path '{value}' does not exist",
                    value=value,
                    severity="error",
                ))

        # Apply message if specified
        if errors and self.message:
            for error in errors:
                error.message = self.message

        return errors


# =============================================================================
# Configuration Schema
# =============================================================================


@dataclass
class ConfigSchema:
    """Schema definition for configuration validation.

    Attributes:
        name: Schema name.
        version: Schema version.
        rules: List of validation rules.
        required_fields: List of required field names.
        allowed_fields: Set of allowed field names (None = all allowed).
    """
    name: str
    version: str = "1.0"
    rules: List[ValidationRule] = field(default_factory=list)
    required_fields: List[str] = field(default_factory=list)
    allowed_fields: Optional[Set[str]] = None

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule.

        Args:
            rule: Validation rule to add.
        """
        self.rules.append(rule)

    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[ValidationError]]:
        """Validate a configuration dictionary.

        Args:
            config: Configuration to validate.

        Returns:
            Tuple of (is_valid, errors).
        """
        errors = []

        # Check required fields
        for field_name in self.required_fields:
            if field_name not in config or config[field_name] is None:
                errors.append(ValidationError(
                    field=field_name,
                    message=f"Required field '{field_name}' is missing",
                    severity="error",
                ))

        # Check allowed fields
        if self.allowed_fields is not None:
            for key in config.keys():
                if key not in self.allowed_fields:
                    errors.append(ValidationError(
                        field=key,
                        message=f"Unknown field '{key}'",
                        severity="warning",
                    ))

        # Validate each rule
        for rule in self.rules:
            value = config.get(rule.field)
            rule_errors = rule.validate(value)
            errors.extend(rule_errors)

        is_valid = all(e.severity != "error" for e in errors)
        return is_valid, errors

    def get_defaults(self) -> Dict[str, Any]:
        """Get default values from schema.

        Returns:
            Dictionary of default values.
        """
        defaults = {}
        for rule in self.rules:
            if rule.default is not None:
                defaults[rule.field] = rule.default
        return defaults

    def to_dict(self) -> dict:
        """Convert schema to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "rules": [
                {
                    "field": r.field,
                    "type": r.vtype.name,
                    "required": r.required,
                    "default": r.default,
                    "min_value": r.min_value,
                    "max_value": r.max_value,
                    "enum_values": r.enum_values,
                    "pattern": r.pattern,
                }
                for r in self.rules
            ],
            "required_fields": self.required_fields,
        }


# =============================================================================
# Configuration Validator
# =============================================================================


class ConfigValidator:
    """Validates and normalizes configurations.

    Supports:
    - Schema-based validation
    - Type coercion
    - Default value resolution
    - Environment variable substitution
    - Configuration merging
    """

    def __init__(self, schemas: Optional[Dict[str, ConfigSchema]] = None):
        """Initialize the validator.

        Args:
            schemas: Dictionary of schema name to ConfigSchema.
        """
        self.schemas = schemas or {}
        self._validation_history: List[dict] = []

    def register_schema(self, schema: ConfigSchema) -> None:
        """Register a configuration schema.

        Args:
            schema: Schema to register.
        """
        self.schemas[schema.name] = schema

    def validate(
        self,
        config: Dict[str, Any],
        schema_name: Optional[str] = None,
        strict: bool = True,
    ) -> Tuple[bool, List[ValidationError]]:
        """Validate a configuration.

        Args:
            config: Configuration to validate.
            schema_name: Schema to use for validation.
            strict: Whether to fail on warnings.

        Returns:
            Tuple of (is_valid, errors).
        """
        if schema_name and schema_name in self.schemas:
            schema = self.schemas[schema_name]
        elif schema_name:
            logger.warning(f"Schema '{schema_name}' not found")
            return True, []
        else:
            # No schema, just check for obvious issues
            errors = self._basic_validation(config)
            return len([e for e in errors if e.severity == "error"]) == 0, errors

        is_valid, errors = schema.validate(config)

        # Apply defaults
        defaults = schema.get_defaults()
        for key, value in defaults.items():
            if key not in config or config[key] is None:
                config[key] = value

        # Log validation result
        self._validation_history.append({
            "schema": schema_name,
            "is_valid": is_valid,
            "errors": [e.to_dict() for e in errors],
            "timestamp": time.time(),
        })

        if not is_valid:
            error_msgs = [f"{e.field}: {e.message}" for e in errors if e.severity == "error"]
            logger.error(f"Validation failed for schema '{schema_name}':")
            for msg in error_msgs:
                logger.error(f"  - {msg}")

        return is_valid, errors

    def apply_defaults(self, config: Dict[str, Any],
                       schema_name: str) -> Dict[str, Any]:
        """Apply default values from schema.

        Args:
            config: Configuration to apply defaults to.
            schema_name: Schema to use.

        Returns:
            Configuration with defaults applied.
        """
        if schema_name not in self.schemas:
            logger.warning(f"Schema '{schema_name}' not found")
            return config

        schema = self.schemas[schema_name]
        defaults = schema.get_defaults()

        for key, value in defaults.items():
            if key not in config or config[key] is None:
                config[key] = value

        return config

    def substitute_env_vars(self, config: Any) -> Any:
        """Substitute environment variables in configuration.

        Supports syntax: ${ENV_VAR} or ${ENV_VAR:default}

        Args:
            config: Configuration (can be nested dict/list).

        Returns:
            Configuration with environment variables substituted.
        """
        if isinstance(config, str):
            # Check for environment variable pattern
            pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"
            match = re.search(pattern, config)

            if match:
                env_var = match.group(1)
                default = match.group(2)

                value = os.environ.get(env_var, default)
                if value is None:
                    value = default or ""

                return value

        elif isinstance(config, dict):
            return {k: self.substitute_env_vars(v) for k, v in config.items()}

        elif isinstance(config, list):
            return [self.substitute_env_vars(item) for item in config]

        return config

    def merge_configs(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any],
        deep: bool = True,
    ) -> Dict[str, Any]:
        """Merge two configurations.

        Args:
            base: Base configuration.
            override: Override configuration.
            deep: Whether to do deep merge.

        Returns:
            Merged configuration.
        """
        if not deep:
            result = base.copy()
            result.update(override)
            return result

        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_configs(result[key], value, deep=True)
            else:
                result[key] = value

        return result

    def get_validation_history(self) -> List[dict]:
        """Get validation history."""
        return self._validation_history

    def _basic_validation(self, config: Dict[str, Any]) -> List[ValidationError]:
        """Basic validation without schema."""
        errors = []

        # Check for None values in required positions
        for key, value in config.items():
            if value is None and key.endswith("_size") or key.endswith("_count"):
                errors.append(ValidationError(
                    field=key,
                    message=f"Configuration value '{key}' cannot be None",
                    severity="warning",
                ))

        return errors


# =============================================================================
# Configuration Templates
# =============================================================================


@dataclass
class ConfigTemplate:
    """Configuration template for generating configs.

    Attributes:
        name: Template name.
        description: Template description.
        variables: Template variables with defaults.
        template: Template string/dict.
    """
    name: str
    description: str = ""
    variables: Dict[str, Any] = field(default_factory=dict)
    template: Union[str, Dict[str, Any]] = ""

    def render(self, overrides: Optional[Dict[str, Any]] = None) -> Union[str, Dict[str, Any]]:
        """Render the template with given overrides.

        Args:
            overrides: Variable overrides.

        Returns:
            Rendered template.
        """
        if isinstance(self.template, str):
            result = self.template
            all_vars = {**self.variables, **(overrides or {})}

            for var, value in all_vars.items():
                placeholder = f"{{{var}}}"
                result = result.replace(placeholder, str(value))

            return result

        elif isinstance(self.template, dict):
            result = self._render_dict(self.template, overrides or {})
            return result

        return self.template

    def _render_dict(self, d: Dict[str, Any],
                     overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively render a dictionary template."""
        result = {}

        for key, value in d.items():
            if isinstance(value, str):
                # Check for template variables
                pattern = r"\{([^}]+)\}"
                matches = re.findall(pattern, value)

                if matches:
                    rendered = value
                    for var in matches:
                        var_value = overrides.get(var, self.variables.get(var, ""))
                        rendered = rendered.replace(f"{{{var}}}", str(var_value))
                    result[key] = rendered
                else:
                    result[key] = value

            elif isinstance(value, dict):
                result[key] = self._render_dict(value, overrides)

            elif isinstance(value, list):
                result[key] = [
                    self._render_dict(item, overrides) if isinstance(item, dict)
                    else item.replace("{var}", str(overrides.get("var", "")))
                    if isinstance(item, str) and "{var}" in item
                    else item
                    for item in value
                ]
            else:
                result[key] = value

        return result


class TemplateRegistry:
    """Registry for configuration templates."""

    def __init__(self):
        """Initialize the template registry."""
        self._templates: Dict[str, ConfigTemplate] = {}
        self._load_builtin_templates()

    def _load_builtin_templates(self) -> None:
        """Load built-in templates."""
        # Evolution training template
        self._templates["evolution"] = ConfigTemplate(
            name="evolution",
            description="Standard evolution training configuration",
            variables={
                "population_size": 200,
                "elite_count": 10,
                "max_generations": 100,
                "crossover_rate": 0.7,
                "mutation_rate": 0.05,
                "mutation_std": 0.1,
                "tournament_mode": False,
            },
            template={
                "population": {
                    "size": "{population_size}",
                    "elite_count": "{elite_count}",
                },
                "evolution": {
                    "max_generations": "{max_generations}",
                    "crossover_rate": "{crossover_rate}",
                    "mutation_rate": "{mutation_rate}",
                    "mutation_std": "{mutation_std}",
                },
                "tournament": {
                    "mode": "{tournament_mode}",
                },
            },
        )

        # Tournament template
        self._templates["tournament"] = ConfigTemplate(
            name="tournament",
            description="Tournament evaluation configuration",
            variables={
                "format": "round_robin",
                "matches_per_pair": 4,
                "num_workers": 4,
            },
            template={
                "tournament": {
                    "format": "{format}",
                    "matches_per_pair": "{matches_per_pair}",
                    "num_workers": "{num_workers}",
                },
            },
        )

        # Export template
        self._templates["export"] = ConfigTemplate(
            name="export",
            description="Model export configuration",
            variables={
                "formats": "torch,numpy,json",
                "output_dir": "exports",
            },
            template={
                "export": {
                    "formats": "{formats}",
                    "output_dir": "{output_dir}",
                },
            },
        )

    def register_template(self, template: ConfigTemplate) -> None:
        """Register a template.

        Args:
            template: Template to register.
        """
        self._templates[template.name] = template

    def get_template(self, name: str) -> Optional[ConfigTemplate]:
        """Get a template by name.

        Args:
            name: Template name.

        Returns:
            ConfigTemplate or None.
        """
        return self._templates.get(name)

    def list_templates(self) -> List[str]:
        """List available template names.

        Returns:
            List of template names.
        """
        return list(self._templates.keys())

    def render_template(self, name: str,
                        overrides: Optional[Dict[str, Any]] = None) -> Optional[Union[str, Dict[str, Any]]]:
        """Render a template with overrides.

        Args:
            name: Template name.
            overrides: Variable overrides.

        Returns:
            Rendered template or None.
        """
        template = self._templates.get(name)
        if template:
            return template.render(overrides)
        return None


# =============================================================================
# Configuration Manager
# =============================================================================


class ConfigurationManager:
    """Manages configuration loading, validation, and resolution.

    Features:
    - YAML/JSON config loading
    - Schema validation
    - Template rendering
    - Environment variable substitution
    - Configuration merging
    - Default resolution
    """

    def __init__(self, config_dir: str = "configs"):
        """Initialize the configuration manager.

        Args:
            config_dir: Directory for configuration files.
        """
        self.config_dir = Path(config_dir)
        self.validator = ConfigValidator()
        self.template_registry = TemplateRegistry()
        self._config_cache: Dict[str, Dict[str, Any]] = {}

        # Register default schemas
        self._register_default_schemas()

    def _register_default_schemas(self) -> None:
        """Register default validation schemas."""
        # Evolution schema
        evolution_schema = ConfigSchema(
            name="evolution",
            version="1.0",
            required_fields=["population", "evolution"],
            allowed_fields={"population", "evolution", "tournament", "evaluation"},
        )
        evolution_schema.add_rule(ValidationRule(
            field="population.size",
            vtype=ValidationType.INTEGER,
            required=True,
            min_value=10,
            max_value=1000,
            message="Population size must be between 10 and 1000",
        ))
        evolution_schema.add_rule(ValidationRule(
            field="evolution.max_generations",
            vtype=ValidationType.INTEGER,
            required=True,
            min_value=1,
            max_value=10000,
            message="Max generations must be between 1 and 10000",
        ))
        evolution_schema.add_rule(ValidationRule(
            field="evolution.crossover_rate",
            vtype=ValidationType.FLOAT,
            min_value=0.0,
            max_value=1.0,
            message="Crossover rate must be between 0 and 1",
        ))
        evolution_schema.add_rule(ValidationRule(
            field="evolution.mutation_rate",
            vtype=ValidationType.FLOAT,
            min_value=0.0,
            max_value=1.0,
            message="Mutation rate must be between 0 and 1",
        ))
        self.validator.register_schema(evolution_schema)

        # Tournament schema
        tournament_schema = ConfigSchema(
            name="tournament",
            version="1.0",
            required_fields=["format"],
            allowed_fields={"format", "matches_per_pair", "num_workers"},
        )
        tournament_schema.add_rule(ValidationRule(
            field="format",
            vtype=ValidationType.ENUM,
            required=True,
            enum_values=["round_robin", "single_elim", "double_elim", "league"],
            message="Tournament format must be one of: round_robin, single_elim, double_elim, league",
        ))
        self.validator.register_schema(tournament_schema)

    def load_config(self, filename: str) -> Dict[str, Any]:
        """Load a configuration file.

        Args:
            filename: Configuration filename.

        Returns:
            Configuration dictionary.
        """
        cache_key = filename
        if cache_key in self._config_cache:
            return self._config_cache[cache_key].copy()

        config_path = self.config_dir / filename

        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return {}

        # Load YAML or JSON
        if config_path.suffix in (".yaml", ".yml"):
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
        elif config_path.suffix == ".json":
            with open(config_path) as f:
                config = json.load(f)
        else:
            logger.warning(f"Unsupported config format: {config_path.suffix}")
            return {}

        # Cache
        self._config_cache[cache_key] = config

        return config.copy()

    def validate_config(self, config: Dict[str, Any],
                        schema_name: str) -> Tuple[bool, List[ValidationError]]:
        """Validate a configuration against a schema.

        Args:
            config: Configuration to validate.
            schema_name: Schema name.

        Returns:
            Tuple of (is_valid, errors).
        """
        return self.validator.validate(config, schema_name)

    def resolve_config(self, config: Dict[str, Any],
                       schema_name: Optional[str] = None) -> Dict[str, Any]:
        """Resolve a configuration (validate, apply defaults, substitute env vars).

        Args:
            config: Configuration to resolve.
            schema_name: Optional schema for validation.

        Returns:
            Resolved configuration.
        """
        # Substitute environment variables
        config = self.validator.substitute_env_vars(config)

        # Validate if schema provided
        if schema_name:
            is_valid, errors = self.validate_config(config, schema_name)
            if not is_valid:
                logger.warning(f"Configuration validation warnings for '{schema_name}':")
                for error in errors:
                    if error.severity == "warning":
                        logger.warning(f"  - {error.field}: {error.message}")

        # Apply defaults
        if schema_name:
            config = self.validator.apply_defaults(config, schema_name)

        return config

    def render_template(self, template_name: str,
                        overrides: Optional[Dict[str, Any]] = None) -> Optional[Union[str, Dict[str, Any]]]:
        """Render a configuration template.

        Args:
            template_name: Template name.
            overrides: Variable overrides.

        Returns:
            Rendered template.
        """
        return self.template_registry.render_template(template_name, overrides)

    def list_templates(self) -> List[str]:
        """List available templates.

        Returns:
            List of template names.
        """
        return self.template_registry.list_templates()

    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configurations.

        Args:
            *configs: Configurations to merge (in order of priority).

        Returns:
            Merged configuration.
        """
        if not configs:
            return {}

        result = configs[0].copy()
        for config in configs[1:]:
            result = self.validator.merge_configs(result, config)

        return result
