"""Data subpackage."""

from . import preprocessing
from . import adapters
from . import manifest
from . import validators
from . import splits
from . import versioning

# Export key classes for convenience
from .manifest import DatasetManifest, ManifestEntry, DuplicateDetector, ClassBalanceAnalyzer
from .validators import SchemaValidator, FileValidator, DataLeakageValidator, ComprehensiveValidator
from .splits import PatientLevelSplitter, ManifestGenerator
from .versioning import DatasetVersion, ManifestHasher, ReproducibilityCard, ExperimentRegistry

__all__ = [
    'preprocessing',
    'adapters',
    'manifest',
    'validators',
    'splits',
    'versioning',
    'DatasetManifest',
    'ManifestEntry',
    'DuplicateDetector',
    'ClassBalanceAnalyzer',
    'SchemaValidator',
    'FileValidator',
    'DataLeakageValidator',
    'ComprehensiveValidator',
    'PatientLevelSplitter',
    'ManifestGenerator',
    'DatasetVersion',
    'ManifestHasher',
    'ReproducibilityCard',
    'ExperimentRegistry',
]
