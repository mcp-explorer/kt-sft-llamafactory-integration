# Documentation Index

## Main Documentation

1. **[KTRANSFORMERS_BUILD_FIX.md](./KTRANSFORMERS_BUILD_FIX.md)** - Complete documentation
   - Problems we're working on
   - What has been solved
   - What is remaining
   - TODOs
   - Guide to commands

2. **[QUICK_REFERENCE.md](./QUICK_REFERENCE.md)** - Quick reference guide
   - Common commands
   - Error patterns
   - File locations

3. **[SUMMARY.md](./SUMMARY.md)** - Session summary
   - Files created
   - Current status
   - Next steps

## Scripts Location

All scripts are in `scripts/docker/`:

- `build_kt_in_docker.sh` - Build KTransformers
- `check_build_errors.sh` - Check errors
- `inspect_source.sh` - Inspect source code
- `fix_and_build.sh` - Apply fixes and rebuild
- `apply_common_fixes.sh` - Apply common fixes
- `fix_kvcache_attn_structure.py` - Fix structure issues
- `fix_missing_commas.py` - Find missing commas
- `check_function_signatures.py` - Check function signatures
- `README.md` - Script usage guide

## Quick Start

```bash
# Read the main documentation
cat docs/docker/KTRANSFORMERS_BUILD_FIX.md

# Use quick reference
cat docs/docker/QUICK_REFERENCE.md

# Run scripts
cd scripts/docker
./build_kt_in_docker.sh
```

