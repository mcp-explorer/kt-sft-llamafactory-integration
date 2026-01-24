# Docker Build Notes

## Issue: `latest-AVX2` Tag Not Updated

### Problem
The `latest-AVX2` tag on DockerHub (`approachingai/ktransformers:latest-AVX2`) is not being automatically updated when new images are built and pushed.

### Current Build Process

1. **Dockerfile** (`docker/Dockerfile`):
   - Builds with `BUILD_ALL_CPU_VARIANTS=1` (line 8)
   - This correctly builds all CPU variants including AVX2
   - Uses `CPUINFER_BUILD_ALL_VARIANTS=1 ./install.sh build` (line 318)
   - Creates versioned tags like: `sglang-v0.5.6_ktransformers-v0.4.3_x86-intel-multi_cu128_sft_...`

2. **Build Script** (`docker/push-to-dockerhub.sh`):
   - Only pushes versioned tags:
     - Full tag: `sglang-v{ver}_ktransformers-v{ver}_{cpu}_{gpu}_{func}_{timestamp}`
     - Simplified tag: `v{ktransformers-ver}-{cuda}` (e.g., `v0.4.3-cu128`)
   - **Missing**: No step to push `latest-AVX2` tag

3. **GitHub Workflow** (`.github/workflows/docker-image.yml`):
   - Triggers on releases and manual workflow_dispatch
   - Calls `push-to-dockerhub.sh` but doesn't add `latest-AVX2` tag

### Root Cause
The build system is designed to create versioned tags only. There's no mechanism to:
1. Create a `latest-AVX2` tag pointing to the latest AVX2-compatible build
2. Update this tag on each new release

### Solution Options

#### Option 1: Add `latest-AVX2` tag to build script (Recommended)
Modify `push-to-dockerhub.sh` to also push `latest-AVX2` tag:

```bash
# After pushing full and simplified tags, add:
if [ "$CPU_VARIANT" = "x86-intel-multi" ]; then
    LATEST_AVX2_TAG="$REGISTRY/$REPOSITORY:latest-AVX2"
    push_image_with_retry "$TEMP_TAG" "$LATEST_AVX2_TAG"
fi
```

#### Option 2: Add to GitHub Workflow
Add a step in `.github/workflows/docker-image.yml` to tag and push `latest-AVX2`:

```yaml
- name: Tag and push latest-AVX2
  if: steps.params.outputs.should_push == 'true'
  run: |
    docker tag ${{ env.DOCKERHUB_REPO }}:$FULL_TAG ${{ env.DOCKERHUB_REPO }}:latest-AVX2
    docker push ${{ env.DOCKERHUB_REPO }}:latest-AVX2
```

#### Option 3: Manual Update
Manually tag and push after each release:
```bash
docker pull approachingai/ktransformers:v0.4.3-cu128
docker tag approachingai/ktransformers:v0.4.3-cu128 approachingai/ktransformers:latest-AVX2
docker push approachingai/ktransformers:latest-AVX2
```

### Current Status
- **Last Update**: Unknown (user reports not updated since June 2024/2025)
- **Build System**: Creates versioned tags correctly
- **Missing**: Automatic `latest-AVX2` tag update mechanism

### Verification
To check when `latest-AVX2` was last updated:
```bash
curl -s https://hub.docker.com/v2/repositories/approachingai/ktransformers/tags/latest-AVX2/ | jq '.last_updated'
```

### Recommended Action
✅ **IMPLEMENTED**: Option 1 has been implemented in `push-to-dockerhub.sh`

### Implementation Details

**Changes Made to `push-to-dockerhub.sh`:**

1. **Added latest-AVX2 tag push** (after simplified tag push):
   ```bash
   # Push latest-AVX2 tag if building with x86-intel-multi (includes AVX2 support)
   if [ "$CPU_VARIANT" = "x86-intel-multi" ]; then
       LATEST_AVX2_IMAGE="$REGISTRY/$REPOSITORY:latest-AVX2"
       log_step "Pushing latest-AVX2 tag"
       if ! push_image_with_retry "$TEMP_TAG" "$LATEST_AVX2_IMAGE"; then
           log_warning "Failed to push latest-AVX2 tag, but continuing..."
       else
           log_success "Successfully updated latest-AVX2 tag"
       fi
   fi
   ```

2. **Added latest-AVX2 to summary output**:
   - Shows in build summary when `latest-AVX2` tag is pushed
   - Helps verify the tag was created successfully

**How It Works:**
- When building with default `CPU_VARIANT=x86-intel-multi` (which includes AVX2 support)
- After pushing the full versioned tag and simplified tag
- Automatically tags the image as `latest-AVX2` and pushes it
- The tag will be updated on each new release/build

**Next Steps:**
- On the next release or manual build, `latest-AVX2` will be automatically updated
- Users pulling `approachingai/ktransformers:latest-AVX2` will get the latest build
