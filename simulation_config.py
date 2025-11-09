"""Configuration objects and enumerations for the iGame simulation launcher."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class DimensionMode(Enum):
    """Representation of the simulation dimensionality."""

    MODE_2D = auto()
    MODE_3D = auto()

    def label(self) -> str:
        return "2D" if self is DimensionMode.MODE_2D else "3D"


class FieldType(Enum):
    """Types of electromagnetic simulations supported by the launcher."""

    ELECTROSTATIC = auto()
    MAGNETOSTATIC = auto()
    COUPLED = auto()

    def label(self) -> str:
        if self is FieldType.ELECTROSTATIC:
            return "Électrostatique"
        if self is FieldType.MAGNETOSTATIC:
            return "Magnétostatique"
        return "Combinaison"


@dataclass
class SimulationConfig:
    """Simple container storing the user-selected simulation settings."""

    dimension: DimensionMode = DimensionMode.MODE_2D
    field_type: FieldType = FieldType.ELECTROSTATIC

    def set_dimension(self, dimension: DimensionMode) -> None:
        """Change the dimensional mode of the simulation."""

        self.dimension = dimension

    def set_field_type(self, field_type: FieldType) -> None:
        """Change the physical model used by the simulation."""

        self.field_type = field_type

    def describe(self) -> str:
        """Return a short human-readable summary of the configuration."""

        return f"Mode {self.dimension.label()} • {self.field_type.label()}"
