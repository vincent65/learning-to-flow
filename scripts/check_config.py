#!/usr/bin/env python3
"""
Config validation and debugging tool.

Checks if YAML config files are being parsed correctly and helps diagnose
type conversion issues.
"""

import yaml
import argparse
import sys
from pathlib import Path


def check_config(config_path: str, verbose: bool = True):
    """
    Check if config file is valid and all values have correct types.

    Args:
        config_path: Path to YAML config file
        verbose: Print detailed information

    Returns:
        True if config is valid, False otherwise
    """
    if verbose:
        print("=" * 70)
        print(f"Config Validation: {config_path}")
        print("=" * 70)

    # Check file exists
    if not Path(config_path).exists():
        print(f"❌ ERROR: Config file not found: {config_path}")
        return False

    if verbose:
        print(f"\n✓ File exists: {config_path}")

    # Load config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if verbose:
            print("✓ YAML parsed successfully")
    except Exception as e:
        print(f"❌ ERROR: Failed to parse YAML: {e}")
        return False

    # Check expected structure
    is_valid = True
    issues = []

    # Common numeric fields that should be float/int
    numeric_checks = [
        ('training', 'learning_rate', float, "Learning rate should be float"),
        ('training', 'batch_size', int, "Batch size should be int"),
        ('training', 'num_epochs', int, "Number of epochs should be int"),
    ]

    # Optional fields depending on config type
    if 'loss' in config:  # FCLF config
        numeric_checks.extend([
            ('loss', 'temperature', float, "Temperature should be float"),
            ('loss', 'lambda_curl', float, "Lambda curl should be float"),
            ('loss', 'lambda_div', float, "Lambda div should be float"),
            ('training', 'alpha', float, "Alpha should be float"),
        ])

    if verbose:
        print("\nValidating numeric fields:")

    for section, key, expected_type, description in numeric_checks:
        if section not in config:
            continue

        if key not in config[section]:
            continue

        value = config[section][key]
        actual_type = type(value)

        # Check if it's the expected type
        if not isinstance(value, expected_type):
            is_valid = False
            issue = f"  ❌ {section}.{key} = {value} (type: {actual_type.__name__}, expected: {expected_type.__name__})"
            issues.append(issue)
            if verbose:
                print(issue)
                print(f"     {description}")

                # Try to convert
                try:
                    converted = expected_type(value)
                    print(f"     💡 Can be fixed by converting to: {converted}")
                except:
                    print(f"     ⚠️  Cannot convert '{value}' to {expected_type.__name__}")
        else:
            if verbose:
                print(f"  ✓ {section}.{key} = {value} ({actual_type.__name__})")

    # Summary
    print("\n" + "=" * 70)
    if is_valid:
        print("✅ CONFIG IS VALID")
        print("=" * 70)
        return True
    else:
        print("❌ CONFIG HAS ISSUES")
        print("=" * 70)
        print("\nProblems found:")
        for issue in issues:
            print(issue)
        print("\n💡 Fix suggestions:")
        print("  1. Edit the config file manually")
        print("  2. Replace scientific notation (1e-4) with decimal (0.0001)")
        print("  3. Remove quotes around numeric values")
        print("=" * 70)
        return False


def print_full_config(config_path: str):
    """Print the entire parsed config for inspection."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("\n" + "=" * 70)
    print("FULL CONFIG DUMP")
    print("=" * 70)

    def print_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                print_dict(value, indent + 1)
            else:
                type_name = type(value).__name__
                print("  " * indent + f"{key}: {value} ({type_name})")

    print_dict(config)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Validate YAML config files")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")
    parser.add_argument("--dump", action="store_true",
                        help="Dump full config with types")

    args = parser.parse_args()

    # Check config
    is_valid = check_config(args.config, verbose=args.verbose)

    # Dump if requested
    if args.dump:
        print_full_config(args.config)

    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
