#!/usr/bin/env python3
"""
Data Validation Utility for MI Knowledge Hub
Validates JSON files for structure, required fields, and data integrity
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

class DataValidator:
    """Validator for MI Knowledge Hub JSON data files"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.errors = []
        self.warnings = []

    def validate_all(self) -> bool:
        """Validate all JSON data files"""
        print("=" * 60)
        print("MI Knowledge Hub - Data Validation")
        print("=" * 60)

        success = True

        # Validate each JSON file
        files = {
            "papers.json": self.validate_papers,
            "datasets.json": self.validate_datasets,
            "tutorials.json": self.validate_tutorials,
            "tools.json": self.validate_tools,
        }

        for filename, validator_func in files.items():
            filepath = self.data_dir / filename
            print(f"\nValidating {filename}...")

            if not filepath.exists():
                self.errors.append(f"{filename} not found")
                print(f"  ❌ File not found")
                success = False
                continue

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                file_success = validator_func(data, filename)
                if file_success:
                    print(f"  ✅ Valid ({len(data)} entries)")
                else:
                    print(f"  ❌ Validation failed")
                    success = False

            except json.JSONDecodeError as e:
                self.errors.append(f"{filename}: Invalid JSON - {e}")
                print(f"  ❌ Invalid JSON: {e}")
                success = False
            except Exception as e:
                self.errors.append(f"{filename}: {e}")
                print(f"  ❌ Error: {e}")
                success = False

        # Print summary
        print("\n" + "=" * 60)
        print("Validation Summary")
        print("=" * 60)

        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")

        if success and not self.warnings:
            print("\n✅ All validations passed!")

        return success

    def validate_papers(self, data: List[Dict], filename: str) -> bool:
        """Validate papers.json structure"""
        required_fields = ["id", "title", "authors", "year", "journal", "doi", "abstract", "tags", "collected_at"]
        return self._validate_entries(data, filename, required_fields)

    def validate_datasets(self, data: List[Dict], filename: str) -> bool:
        """Validate datasets.json structure"""
        required_fields = ["id", "name", "description", "url", "data_types", "size", "license", "updated_at"]
        return self._validate_entries(data, filename, required_fields)

    def validate_tutorials(self, data: List[Dict], filename: str) -> bool:
        """Validate tutorials.json structure"""
        required_fields = ["id", "title", "description", "level", "difficulty", "estimated_time", "notebook_url", "topics", "prerequisites"]
        return self._validate_entries(data, filename, required_fields)

    def validate_tools(self, data: List[Dict], filename: str) -> bool:
        """Validate tools.json structure"""
        required_fields = ["id", "name", "description", "url", "category", "language", "license", "tags"]
        return self._validate_entries(data, filename, required_fields)

    def _validate_entries(self, data: List[Dict], filename: str, required_fields: List[str]) -> bool:
        """Generic validator for entries"""
        if not isinstance(data, list):
            self.errors.append(f"{filename}: Expected list, got {type(data)}")
            return False

        success = True
        ids_seen = set()

        for i, entry in enumerate(data):
            entry_id = entry.get('id', f'entry_{i}')

            # Check for duplicate IDs
            if entry_id in ids_seen:
                self.errors.append(f"{filename}: Duplicate ID '{entry_id}'")
                success = False
            ids_seen.add(entry_id)

            # Check required fields
            for field in required_fields:
                if field not in entry:
                    self.errors.append(f"{filename}[{entry_id}]: Missing required field '{field}'")
                    success = False
                elif not entry[field]:
                    self.warnings.append(f"{filename}[{entry_id}]: Empty value for '{field}'")

        return success


def main():
    """Main validation function"""
    # Get data directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        sys.exit(1)

    # Run validation
    validator = DataValidator(data_dir)
    success = validator.validate_all()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
