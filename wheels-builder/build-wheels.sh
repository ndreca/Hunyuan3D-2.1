#!/bin/bash

# Build all wheels using the main cog container for compatibility
set -e

echo "Building all wheels for Hunyuan3D-2.1..."

# Get the parent directory (source code)
PARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Source directory: $PARENT_DIR"
echo "Wheels output directory: $WHEELS_DIR"

# Build the main cog image first
echo "=== Building main cog image ==="
cd "$WHEELS_DIR"
cog build -t hunyuan3d-2.1-wheels --debug

# Clean existing wheels
cd "$WHEELS_DIR"
rm -f *.whl *.so

# Create a temporary container, copy source, build wheels, then copy wheels back
echo "=== Building all wheels in isolated container ==="
CONTAINER_ID=$(docker create --platform linux/amd64 \
  -e "TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;8.9;9.0+PTX" \
  -e FORCE_CUDA=1 \
  -e MAX_JOBS=4 \
  hunyuan3d-2.1-wheels \
  bash -c "
    # Clean and setup source directory
    rm -rf /src/* /src/.*[!.]* 2>/dev/null || true
    cp -r /tmp/source/. /src/
    
    # Activate virtual environment
    source /root/.venv/bin/activate
    
    echo '=== Environment Info ==='
    python --version
    python -m pip --version
    which python
    echo 'Virtual env: $VIRTUAL_ENV'
    
    echo 'Building custom_rasterizer wheel...'
    cd /src/hy3dpaint/custom_rasterizer
    if [ -f setup.py ]; then
        python -m pip wheel . --wheel-dir /tmp/wheels --no-deps --no-build-isolation --verbose
        echo '✅ custom_rasterizer wheel built'
    else
        echo '❌ custom_rasterizer setup.py not found'
        exit 1
    fi
    
    echo 'Compiling mesh_inpaint_processor...'
    cd /src/hy3dpaint/DifferentiableRenderer
    if [ -f compile_mesh_painter.sh ]; then
        bash compile_mesh_painter.sh
        # Verify compilation succeeded
        if ls mesh_inpaint_processor*.so 1> /dev/null 2>&1; then
            echo '✅ mesh_inpaint_processor compiled successfully'
            echo 'ℹ️  .so file will remain in place for direct import'
        else
            echo '❌ mesh_inpaint_processor compilation failed'
            exit 1
        fi
    else
        echo '❌ compile_mesh_painter.sh not found'
        exit 1
    fi
    
    echo 'All wheels and libraries built successfully!'
  ")

# Copy source code into the container
echo "Copying source code into container..."
docker cp "$PARENT_DIR/." "$CONTAINER_ID:/tmp/source/"

# Start the container and run the build
echo "Running build process..."
docker start -a "$CONTAINER_ID"

# Copy wheels back to host
echo "Copying wheels back to host..."
docker cp "$CONTAINER_ID:/tmp/wheels/" "$WHEELS_DIR/"
mv "$WHEELS_DIR/wheels/"*.whl "$WHEELS_DIR/" 2>/dev/null || true
rmdir "$WHEELS_DIR/wheels" 2>/dev/null || true

# Copy compiled .so file to its proper location for commit
echo "Copying mesh_inpaint_processor.so to proper location..."
# Copy all .so files from the container (container is stopped, so we copy the whole directory)
docker cp "$CONTAINER_ID:/src/hy3dpaint/DifferentiableRenderer/" "$PARENT_DIR/hy3dpaint/" 2>/dev/null || true

# Clean up container
docker rm "$CONTAINER_ID"

echo "=== All wheels and libraries built! ==="
echo "Wheels directory contents:"
ls -la *.whl 2>/dev/null || echo "No wheels found"

echo "mesh_inpaint_processor.so status:"
if ls "$PARENT_DIR/hy3dpaint/DifferentiableRenderer/mesh_inpaint_processor"*.so 1> /dev/null 2>&1; then
    echo "✅ mesh_inpaint_processor.so copied to proper location for commit"
    ls -la "$PARENT_DIR/hy3dpaint/DifferentiableRenderer/mesh_inpaint_processor"*.so
else
    echo "❌ mesh_inpaint_processor.so not found - checking what was copied:"
    ls -la "$PARENT_DIR/hy3dpaint/DifferentiableRenderer/" 2>/dev/null || echo "DifferentiableRenderer directory not found"
fi

echo "Done! Wheels are in $WHEELS_DIR, .so file is in place for commit" 