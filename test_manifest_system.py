#!/usr/bin/env python3
"""Quick validation test for manifest system."""

import sys
import os

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing manifest system imports...")

try:
    from data import (
        DatasetManifest,
        ManifestEntry,
        DuplicateDetector,
        ClassBalanceAnalyzer,
        SchemaValidator,
        FileValidator,
        DataLeakageValidator,
        ComprehensiveValidator,
        PatientLevelSplitter,
        ManifestGenerator,
        DatasetVersion,
        ManifestHasher,
        ReproducibilityCard,
        ExperimentRegistry,
    )
    print("✅ All manifest system modules imported successfully")
    
    # Test basic functionality
    entry = ManifestEntry(
        patient_id='test_001',
        study_id='study_001',
        aneurysm_id='aneur_001',
        source='synthetic',
        geometry_path='test.stl',
        rupture_status=1
    )
    
    manifest = DatasetManifest('test_manifest')
    success, error = manifest.add_entry(entry)
    
    if success:
        print(f"✅ Created test manifest with {len(manifest.entries)} entry")
        print(f"✅ Statistics: {manifest.statistics()}")
    else:
        print(f"❌ Failed to add entry: {error}")
        sys.exit(1)
    
    # Test version
    version = DatasetVersion("1.0", "synthetic")
    print(f"✅ Created dataset version: {version.version_id}")
    
    # Test reproducibility card
    card = ReproducibilityCard("test_experiment")
    card.set_training_info(
        model_name="PointNet++",
        seed=42,
        loss_function="BCE",
        optimizer="Adam",
        learning_rate=0.001,
        batch_size=32,
        num_epochs=100
    )
    print(f"✅ Created reproducibility card: {card.experiment_id}")
    
    print("\n" + "="*60)
    print("✅ ALL MANIFEST SYSTEM TESTS PASSED")
    print("="*60)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
