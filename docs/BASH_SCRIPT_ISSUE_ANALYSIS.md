# Bash Script Issue Analysis: Duplicate Training Processes

**Date:** 2026-01-15  
**Script:** `scripts/training/dpo_ds2_chat_lite_hf.sh`

## Problem

Two training processes were running simultaneously, competing for GPU resources.

## Root Cause Analysis

### The Script Has Protection, But...

The script **DOES** have a check for existing training processes (lines 516-534):

```bash
# Check for existing training processes and kill them
echo "Checking for existing training processes..."
EXISTING_PIDS=$(pgrep -f "llamafactory-cli train" || true)
if [ -n "$EXISTING_PIDS" ]; then
    echo "⚠ Warning: Found existing training processes: $EXISTING_PIDS"
    echo "  Killing existing processes to prevent conflicts..."
    pkill -f "llamafactory-cli train" || true
    sleep 2
    # ... force kill if needed
fi
```

### Why It Failed

1. **Bypassed the Script:**
   - First training: Started via script ✅ (check ran)
   - Second training: Started via **direct command** ❌ (bypassed check)
   - Direct command: `nohup conda run -n deepspeed-z3 ... llamafactory-cli train ...`
   - This completely bypassed the script's safety check

2. **Race Condition:**
   - Script runs training **synchronously** (not in background by default)
   - If script is run with `&` (background), there's a time window where:
     - Process starts but isn't fully registered in process list yet
     - `pgrep` might not find it immediately
     - Second process can start before first is detected

3. **Process Detection Timing:**
   - `pgrep -f "llamafactory-cli train"` might not catch processes that are:
     - Just starting (not fully spawned)
     - Running via `conda run` (different process tree)
     - Using different command paths

## What Happened

1. **17:30** - First training started via script:
   ```bash
   ./scripts/training/dpo_ds2_chat_lite_hf.sh --dataset identity_sean_dpo_improved --yes
   ```
   - Script checked for processes (found none)
   - Started training
   - But had config issues (wrong dataset initially)

2. **17:31** - Second training started via direct command:
   ```bash
   nohup conda run -n deepspeed-z3 ... llamafactory-cli train ...
   ```
   - **Bypassed script entirely**
   - No process check
   - Started while first was still running

3. **Result:** Both processes running simultaneously

## Issues in Script

### 1. Process Detection May Be Too Narrow

```bash
EXISTING_PIDS=$(pgrep -f "llamafactory-cli train" || true)
```

**Problem:**
- Only checks for exact pattern "llamafactory-cli train"
- Might miss processes started via `conda run` wrapper
- Might miss processes in different process trees

**Better approach:**
```bash
# Check for any training-related processes
EXISTING_PIDS=$(pgrep -f "llamafactory.*train\|torchrun.*dpo" || true)
```

### 2. No Check for Background Processes

The script doesn't check if it's being run in background, which could cause issues.

### 3. Training Runs Synchronously

Line 621-626: Training runs synchronously, but if script is backgrounded with `&`, the check might not work properly.

## Recommendations

### 1. Improve Process Detection

```bash
# More comprehensive check
EXISTING_PIDS=$(pgrep -f "llamafactory.*train\|torchrun.*dpo\|train.*dpo" || true)
```

### 2. Add Lock File Mechanism

```bash
LOCK_FILE="/tmp/dpo_training.lock"
if [ -f "$LOCK_FILE" ]; then
    LOCK_PID=$(cat "$LOCK_FILE")
    if ps -p "$LOCK_PID" > /dev/null 2>&1; then
        echo "⚠ Training already running (PID: $LOCK_PID)"
        exit 1
    else
        rm -f "$LOCK_FILE"
    fi
fi
echo $$ > "$LOCK_FILE"
trap "rm -f $LOCK_FILE" EXIT
```

### 3. Check Before Starting

Add check at the very beginning of script, before any setup:

```bash
# Check at start of script
if pgrep -f "llamafactory.*train.*dpo\|torchrun.*dpo" > /dev/null; then
    echo "Error: DPO training already running!"
    echo "  PIDs: $(pgrep -f 'llamafactory.*train.*dpo|torchrun.*dpo')"
    exit 1
fi
```

### 4. Warn About Direct Commands

Add documentation warning against bypassing the script.

## Conclusion

**The script has protection, but:**
- ✅ Works when script is used properly
- ❌ Can be bypassed by running direct commands
- ⚠️ Race condition possible if script is backgrounded

**The real issue:** Training was started via direct command, completely bypassing the script's safety checks.

**Solution:** Always use the script, or improve process detection to catch all training processes regardless of how they're started.
