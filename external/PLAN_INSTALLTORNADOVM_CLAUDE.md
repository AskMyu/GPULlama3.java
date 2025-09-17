# TornadoVM Local Installation Plan - Claude Code Implementation Guide

**Generated:** 2025-09-15 (Updated from 2025-09-12)
**Purpose:** Step-by-step executable plan for Claude Code to install TornadoVM 1.1.1 locally with both OpenCL and PTX backends

## IMPORTANT: Claude Implementation Instructions

This plan is designed for Claude Code to execute directly. Each phase contains specific commands that should be run sequentially. Do NOT execute anything until explicitly instructed by the user.

## Executive Summary

Install TornadoVM 1.1.1 from source (`./tornadovm-local-src/`) into local directory (`./tornadovm-local/`). The installation provides GPU acceleration APIs and native libraries with **both OpenCL and PTX backends** for maximum compatibility. PTX backend requires CUDA <= 11.x (CUDA 12+ not supported).

**Working Directory:** Current directory (`./`) is the main project directory containing the TornadoVM installation.

**Note:** If `./tornadovm-local-src/` does not exist or is empty, the TornadoVM source can be cloned from the official repository:
```bash
git clone https://github.com/beehive-lab/TornadoVM.git ./tornadovm-local-src
```

## Pre-Execution Checklist

Before starting, Claude should verify:
- Current working directory: Should be in the main project directory (`./`)
- Source exists: `./tornadovm-local-src/` directory (will be cloned if missing)
- NVIDIA GPU system: CUDA drivers installed and nvidia-smi working
- CUDA version compatibility: CUDA <= 11.x (PTX backend limitation)
- User has confirmed they want to proceed with installation
- User has sudo access if system packages need installation
- Internet connection available (for cloning repository if needed)

## CRITICAL: System Requirements

**⚠️ IMPORTANT Requirements:**
- **Maven 3.9.0+** (Ubuntu default 3.8.7 will fail)
- **GCC >= 10.0** (for successful compilation)
- **CUDA <= 11.x** (PTX backend does not support CUDA 12+)
- **Ubuntu 20.04+** (recommended platform)

## Phase 1: System Verification and Dependencies

### Step 1.1: Check System Dependencies

Run these commands to verify prerequisites:

```bash
# Check current environment - should be in main project directory
pwd

# Verify system requirements
gcc --version | grep -E "gcc.*([1-9][0-9]|10)\." || echo "WARNING: GCC 10.0+ required"
lsb_release -r | grep -E "(20\.04|22\.04|24\.04)" || echo "WARNING: Ubuntu 20.04+ recommended"

# Check CUDA version compatibility
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -o "release [0-9]*\.[0-9]*" | cut -d' ' -f2)
    if [[ $(echo "$CUDA_VERSION" | cut -d'.' -f1) -ge 12 ]]; then
        echo "WARNING: CUDA $CUDA_VERSION detected. PTX backend requires CUDA <= 11.x"
        echo "Consider OpenCL-only build or downgrade CUDA for PTX support"
    else
        echo "CUDA $CUDA_VERSION detected - compatible with PTX backend"
    fi
else
    echo "CUDA not detected - OpenCL backend only"
fi

# Ensure source directory exists - clone if needed
if [ ! -d "./tornadovm-local-src" ] || [ -z "$(ls -A ./tornadovm-local-src 2>/dev/null)" ]; then
    echo "TornadoVM source not found. Cloning from GitHub..."
    git clone https://github.com/beehive-lab/TornadoVM.git ./tornadovm-local-src
else
    echo "TornadoVM source found"
fi

ls -la ./tornadovm-local-src/

# Verify Java 21
java -version

# Check build tools
python3 --version
cmake --version
mvn --version  # Should show Maven 3.9.0+

# Check NVIDIA CUDA drivers
nvidia-smi 2>/dev/null || echo "NVIDIA CUDA not detected"
nvcc --version 2>/dev/null || echo "NVCC not found"

# Check OpenCL
clinfo 2>/dev/null || echo "OpenCL not detected"
```

### Step 1.2: Install Missing Dependencies (if needed)

If any dependencies are missing, run:

```bash
# Ubuntu/Debian systems
sudo apt update
sudo apt install -y gcc g++ git cmake python3 python3-pip python3-venv python3-wget
sudo apt install -y default-jdk

# NVIDIA CUDA development (for PTX backend)
sudo apt install -y nvidia-cuda-dev nvidia-cuda-toolkit

# OpenCL support (required backend)
sudo apt install -y ocl-icd-opencl-dev opencl-headers nvidia-opencl-dev

# Intel GPUs (optional)
# sudo apt install -y intel-opencl-icd
```

### Step 1.3: Install Maven 3.9.6+ (Critical)

```bash
# Remove old Maven
sudo apt remove maven -y 2>/dev/null || true

# Create temporary directory
mkdir -p tmp
cd tmp

# Install Maven 3.9.6
wget https://archive.apache.org/dist/maven/maven-3/3.9.6/binaries/apache-maven-3.9.6-bin.tar.gz
tar -xzf apache-maven-3.9.6-bin.tar.gz
sudo mv apache-maven-3.9.6 /opt/maven-3.9.6
sudo ln -sf /opt/maven-3.9.6/bin/mvn /usr/local/bin/mvn

# Set environment
export M2_HOME=/opt/maven-3.9.6
export PATH=/opt/maven-3.9.6/bin:$PATH

# Verify installation
mvn --version

cd ..
```

### Step 1.4: Set Environment Variables

```bash
# Set JAVA_HOME if not already set
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
echo "JAVA_HOME: $JAVA_HOME"

# Create installation directory
mkdir -p ./tornadovm-local
export TORNADO_ROOT=$(pwd)/tornadovm-local
echo "TORNADO_ROOT: $TORNADO_ROOT"
```

## Phase 2: Build TornadoVM

### Step 2.1: Navigate and Prepare Source

```bash
cd ./tornadovm-local-src

# Check TornadoVM version
cat README.md | grep -E "version|1\.1\.1" | head -5

# Verify installer exists
ls -la bin/tornadovm-installer
```

### Step 2.1A: Apply Custom Modifications (FloatArrayLongFix)

**IMPORTANT**: Check for custom modifications that need to be applied before building:

```bash
# Check if patch directory exists with FloatArray.java modifications
if [ -f "../patch/FloatArray.java" ]; then
    echo "🚀 FLOATARRAY-LONGFIX: Custom FloatArray.java found in patch directory"
    echo "Applying FloatArrayLongFix for large tensor allocation support..."

    # Backup original FloatArray.java
    FLOATARRAY_PATH="tornado-api/src/main/java/uk/ac/manchester/tornado/api/types/arrays/FloatArray.java"
    if [ -f "$FLOATARRAY_PATH" ]; then
        cp "$FLOATARRAY_PATH" "$FLOATARRAY_PATH.backup.$(date +%Y%m%d_%H%M%S)"
        echo "Original FloatArray.java backed up"
    fi

    # Copy modified FloatArray.java to source
    cp "../patch/FloatArray.java" "$FLOATARRAY_PATH"
    echo "Modified FloatArray.java copied to: $FLOATARRAY_PATH"

    # Verify the modification was applied
    if grep -q "FLOATARRAY-LONGFIX" "$FLOATARRAY_PATH"; then
        echo "✓ FloatArrayLongFix modification verified in source"
        echo "  This fix enables large tensor allocations >2GB for models like Gemma2B"
    else
        echo "⚠️  WARNING: FloatArrayLongFix markers not found - modification may not be applied"
    fi
else
    echo "No custom FloatArray.java modifications found in ../patch/ directory"
    echo "Building with standard TornadoVM FloatArray implementation"
fi
```

### Step 2.2: Determine Backend Configuration

Check available GPU backends and build both OpenCL and PTX:

```bash
# Initialize backend list
BACKENDS=""

# Check OpenCL availability (required)
if command -v clinfo &> /dev/null; then
    echo "OpenCL available - adding as backend"
    BACKENDS="opencl"
    OPENCL_AVAILABLE=true
else
    echo "ERROR: OpenCL not available - required for TornadoVM"
    OPENCL_AVAILABLE=false
fi

# Check CUDA/PTX availability
if command -v nvidia-smi &> /dev/null && command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -o "release [0-9]*\.[0-9]*" | cut -d' ' -f2)
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d'.' -f1)

    if [[ $CUDA_MAJOR -lt 12 ]]; then
        echo "CUDA $CUDA_VERSION available - adding PTX backend"
        if [ "$OPENCL_AVAILABLE" = true ]; then
            BACKENDS="${BACKENDS},ptx"
        else
            BACKENDS="ptx"
        fi
        PTX_AVAILABLE=true
    else
        echo "CUDA $CUDA_VERSION detected - PTX backend not supported (requires CUDA <= 11.x)"
        PTX_AVAILABLE=false
    fi
else
    echo "CUDA/PTX not available - OpenCL backend only"
    PTX_AVAILABLE=false
fi

# Check Intel Level Zero (SPIRV) - optional
if command -v sycl-ls &> /dev/null || [ -f /usr/lib/x86_64-linux-gnu/libze_loader.so ]; then
    echo "Intel Level Zero available - adding SPIRV backend"
    BACKENDS="${BACKENDS},spirv"
fi

echo "Selected backends: $BACKENDS"
echo "Primary backend: $(echo $BACKENDS | cut -d',' -f1)"

# Validate at least one backend is available
if [ -z "$BACKENDS" ]; then
    echo "ERROR: No compatible backends found. Cannot proceed."
    exit 1
fi
```

### Step 2.3: Run TornadoVM Installer

Execute the automated installer with both backends:

```bash
# Run installer with available backends
echo "Installing TornadoVM with backends: $BACKENDS"
./bin/tornadovm-installer --jdk jdk21 --backend $BACKENDS --auto-deps

# If installer fails, try manual build
if [ $? -ne 0 ]; then
    echo "Installer failed. Attempting manual build..."
    make clean
    make jdk21 BACKENDS=$BACKENDS
fi
```

### Step 2.4: Verify Build Success

```bash
# Check if build completed
echo "Checking build artifacts..."
ls -la bin/sdk/
ls -la target/tornado-api*.jar
ls -la tornado-matrices/target/tornado-matrices*.jar

# Source generated environment
source setvars.sh

# Test basic functionality
tornado --version
tornado --devices
```

## Phase 3: Create Self-Contained Local Installation

### Step 3.1: Create Installation Structure

```bash
# Navigate back to main project directory
cd ..

# Create complete installation structure
mkdir -p tornadovm-local/{bin,lib,etc,share,lib/native,examples}

echo "Created installation directory structure"
ls -la tornadovm-local/
```

### Step 3.2: Copy All Required Files

```bash
cd tornadovm-local-src

# Copy TornadoVM SDK (complete) - IMPORTANT: Copy as real directory, not symlink
cp -rL bin/sdk ../tornadovm-local/bin/
# Alternative if SDK is symlinked to dist directory:
# cp -rL dist/tornado-sdk/tornado-sdk-*/ ../tornadovm-local/bin/sdk/

# Copy CLI tools including Python utilities
cp bin/tornado ../tornadovm-local/bin/
cp bin/tornado-test ../tornadovm-local/bin/
cp bin/*.py ../tornadovm-local/bin/ 2>/dev/null || true
chmod +x ../tornadovm-local/bin/tornado*

# Ensure SDK directory is self-contained with proper permissions
chmod -R u+rwX,go+rX ../tornadovm-local/bin/sdk/

# Copy all API JARs
find . -name "tornado-api-*.jar" -exec cp {} ../tornadovm-local/lib/ \;
find . -name "tornado-matrices-*.jar" -exec cp {} ../tornadovm-local/lib/ \;
find . -name "tornado-runtime-*.jar" -exec cp {} ../tornadovm-local/lib/ \;
find . -name "tornado-drivers-*.jar" -exec cp {} ../tornadovm-local/lib/ \;

# Copy all native libraries (comprehensive search)
find . -name "*.so" -path "*/build/cmake/*" -exec cp {} ../tornadovm-local/lib/native/ \; 2>/dev/null || true
find . -name "*.so" -path "*/target/*" -exec cp {} ../tornadovm-local/lib/native/ \; 2>/dev/null || true
find . -name "*.so" -path "*/lib/*" -exec cp {} ../tornadovm-local/lib/native/ \; 2>/dev/null || true

# Copy configuration files
cp setvars.sh ../tornadovm-local/etc/original-setvars.sh 2>/dev/null || true

# Copy examples if they exist
cp -r examples ../tornadovm-local/ 2>/dev/null || true

# List copied files for verification
echo "API JARs:"
ls -la ../tornadovm-local/lib/*.jar

echo "Native libraries:"
ls -la ../tornadovm-local/lib/native/

echo "SDK contents:"
ls -la ../tornadovm-local/bin/sdk/
```

### Step 3.3: Create Self-Contained Environment Setup Script

```bash
cat > ../tornadovm-local/etc/setvars.sh << 'EOF'
#!/bin/bash
# TornadoVM Local Installation Environment Setup
# Self-contained installation - no dependency on source directory

# Get absolute path of installation directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export TORNADO_ROOT="$(dirname "$SCRIPT_DIR")"
export TORNADO_SDK="$TORNADO_ROOT/bin/sdk"

# Add TornadoVM tools to PATH
export PATH="$TORNADO_ROOT/bin:$PATH"

# Set library path for native drivers
export LD_LIBRARY_PATH="$TORNADO_ROOT/lib/native:${LD_LIBRARY_PATH:-}"

# Java library path for JNI libraries
export JAVA_LIBRARY_PATH="$TORNADO_ROOT/lib/native:${JAVA_LIBRARY_PATH:-}"

# TornadoVM JVM flags for runtime
export TORNADO_JVM_FLAGS="-XX:+UnlockExperimentalVMOptions -XX:+EnableJVMCI"
export TORNADO_JVM_FLAGS="$TORNADO_JVM_FLAGS -Djava.library.path=$JAVA_LIBRARY_PATH"

# Set classpath for TornadoVM JARs
TORNADO_CLASSPATH="$TORNADO_ROOT/lib/*"
export TORNADO_JVM_FLAGS="$TORNADO_JVM_FLAGS -cp $TORNADO_CLASSPATH"

# Default backend selection
if [ -f "$TORNADO_ROOT/lib/native/libtornado-ptx.so" ]; then
    export TORNADO_DEFAULT_BACKEND="ptx"
    echo "PTX backend available"
elif [ -f "$TORNADO_ROOT/lib/native/libtornado-opencl.so" ]; then
    export TORNADO_DEFAULT_BACKEND="opencl"
    echo "OpenCL backend available"
else
    echo "Warning: No native backends detected"
fi

# Module path for TornadoVM (if SDK has modules)
if [ -d "$TORNADO_SDK/share/java/tornado" ]; then
    export TORNADO_JVM_FLAGS="$TORNADO_JVM_FLAGS --module-path $TORNADO_SDK/share/java/tornado"
fi

echo "TornadoVM environment configured:"
echo "  TORNADO_ROOT: $TORNADO_ROOT"
echo "  TORNADO_SDK: $TORNADO_SDK"
echo "  Default Backend: ${TORNADO_DEFAULT_BACKEND:-none}"
echo "  Native libraries: $TORNADO_ROOT/lib/native"
echo "  API JARs: $TORNADO_ROOT/lib"
echo ""
echo "Usage: tornado --devices"
EOF

chmod +x ../tornadovm-local/etc/setvars.sh
```

## Phase 4: Installation Verification

### Step 4.1: Test Self-Contained Installation

```bash
cd ..

# Source local environment (should work independently)
source ./tornadovm-local/etc/setvars.sh

# Verify installation works without source directory
echo "Testing self-contained installation..."
tornado --version || echo "WARNING: tornado command failed"
tornado --devices || echo "WARNING: no devices detected"

# List installation contents
echo "Installation structure:"
find ./tornadovm-local -type f | head -30
```

### Step 4.2: Run Basic Test

```bash
# Try running a simple test
timeout 30 tornado -m tornado.examples/uk.ac.manchester.tornado.examples.compute.MatrixMultiplication2D 256 2>/dev/null || echo "Example test completed"
```

## Phase 5: Create Documentation and Verification Tools

### Step 5.1: Create Verification Script

```bash
cat > ./verify-tornadovm.sh << 'EOF'
#!/bin/bash
echo "=== TornadoVM Installation Verification ==="

# Get script directory and set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Source environment
source ./tornadovm-local/etc/setvars.sh

echo "1. Installation structure:"
ls -la ./tornadovm-local/

echo -e "\n2. TornadoVM version:"
tornado --version 2>/dev/null || echo "ERROR: tornado command not found"

echo -e "\n3. Available devices:"
timeout 15 tornado --devices 2>/dev/null || echo "ERROR: Device query failed"

echo -e "\n4. API JARs:"
ls -la ./tornadovm-local/lib/*.jar 2>/dev/null || echo "No JAR files found"

echo -e "\n5. Native libraries:"
ls -la ./tornadovm-local/lib/native/ 2>/dev/null || echo "No native libraries found"

echo -e "\n6. Backend availability:"
if [ -f "./tornadovm-local/lib/native/libtornado-ptx.so" ]; then
    echo "✓ PTX backend available"
else
    echo "✗ PTX backend not found"
fi

if [ -f "./tornadovm-local/lib/native/libtornado-opencl.so" ]; then
    echo "✓ OpenCL backend available"
else
    echo "✗ OpenCL backend not found"
fi

echo -e "\n7. Environment variables:"
echo "TORNADO_ROOT: $TORNADO_ROOT"
echo "TORNADO_SDK: $TORNADO_SDK"
echo "TORNADO_DEFAULT_BACKEND: $TORNADO_DEFAULT_BACKEND"
echo "LD_LIBRARY_PATH includes TornadoVM: $(echo $LD_LIBRARY_PATH | grep -o 'tornadovm-local' || echo 'No')"

echo -e "\n=== Verification Complete ==="

# Return success if basic functionality works
if command -v tornado &> /dev/null && tornado --version &> /dev/null; then
    echo "✓ Installation appears successful"
    exit 0
else
    echo "✗ Installation verification failed"
    exit 1
fi
EOF

chmod +x ./verify-tornadovm.sh
```

### Step 5.2: Create Usage Documentation

```bash
cat > ./TORNADOVM_USAGE.md << 'EOF'
# TornadoVM Local Installation Usage Guide

## Quick Start

### 1. Activate TornadoVM Environment
```bash
source ./tornadovm-local/etc/setvars.sh
```

### 2. Verify Installation
```bash
./verify-tornadovm.sh
```

### 3. Check Available Devices
```bash
tornado --devices
```

## Self-Contained Installation

This installation is completely self-contained in `./tornadovm-local/` and includes:
- All TornadoVM JARs and native libraries
- Complete SDK and runtime components
- Environment setup script
- Command-line tools (tornado, tornado-test)

## Backend Usage

### Backend Selection
```bash
# Check which backends are available
ls ./tornadovm-local/lib/native/libtornado-*.so

# Set backend preference (optional)
export TORNADO_BACKEND=ptx      # For NVIDIA GPUs (CUDA <= 11.x)
export TORNADO_BACKEND=opencl   # For broader compatibility
export TORNADO_BACKEND=spirv    # For Intel GPUs
```

### JVM Integration
```bash
# Use in Java applications
java $TORNADO_JVM_FLAGS -cp "your-app.jar:$TORNADO_ROOT/lib/*" YourMainClass
```

## Configuration

### Environment Variables
- `TORNADO_ROOT`: Installation directory
- `TORNADO_SDK`: SDK location
- `TORNADO_DEFAULT_BACKEND`: Preferred backend
- `TORNADO_JVM_FLAGS`: JVM flags for TornadoVM

### Debug Mode
```bash
export TORNADO_DEBUG=true
tornado --jvm="-Dtornado.debug=true" --devices
```

## Troubleshooting

### Common Issues
- **No devices found**: Check GPU drivers and OpenCL/CUDA installation
- **PTX backend issues**: Verify CUDA <= 11.x and nvidia drivers
- **Native library errors**: Check LD_LIBRARY_PATH in setvars.sh
- **JVM crashes**: Ensure Java 21 and proper JVM configuration

### Backend Compatibility
- **OpenCL**: Works with NVIDIA, AMD, Intel GPUs
- **PTX**: NVIDIA GPUs only, requires CUDA <= 11.x
- **SPIRV**: Intel GPUs with Level Zero drivers
EOF
```

## Phase 6: Final Installation Summary

### Step 6.1: Create Installation Summary

```bash
cat > ./TORNADOVM_INSTALLATION_SUMMARY.md << 'EOF'
# TornadoVM Installation Summary

## Installation Details
- **Source**: ./tornadovm-local-src/
- **Installation**: ./tornadovm-local/
- **Version**: TornadoVM 1.1.1
- **Backends**: OpenCL (required), PTX (NVIDIA, CUDA <= 11.x), SPIRV (Intel, optional)
- **Type**: Self-contained local installation
- **Modifications**: Automatically applies FloatArrayLongFix from ../patch/ directory if present

## Key Files Created
- `./tornadovm-local/etc/setvars.sh` - Environment setup (self-contained)
- `./tornadovm-local/bin/tornado` - TornadoVM command-line tool
- `./tornadovm-local/bin/sdk/` - Complete SDK directory (real directory, not symlink)
- `./tornadovm-local/lib/*.jar` - API JARs
- `./tornadovm-local/lib/native/*.so` - Native backend libraries
- `./verify-tornadovm.sh` - Installation verification script
- `./TORNADOVM_USAGE.md` - Usage documentation

## Installation Structure
```
./
├── tornadovm-local-src/     # Source code (for reference)
├── tornadovm-local/         # Self-contained installation
│   ├── bin/                 # Executables and SDK (real directory)
│   │   ├── tornado          # CLI tool
│   │   ├── *.py             # Python utilities
│   │   └── sdk/             # SDK directory (NOT symlink)
│   ├── lib/                 # JAR files
│   ├── lib/native/          # Native libraries
│   └── etc/setvars.sh       # Environment setup
├── verify-tornadovm.sh      # Verification script
├── TORNADOVM_USAGE.md       # Usage guide
└── TORNADOVM_INSTALLATION_SUMMARY.md
```

## Backend Status
- [ ] OpenCL backend built and installed
- [ ] PTX backend built (if CUDA <= 11.x available)
- [ ] SPIRV backend built (if Intel Level Zero available)
- [ ] Native libraries verified
- [ ] Basic functionality tested

## Next Steps
1. Run verification: `./verify-tornadovm.sh`
2. Test GPU acceleration with your applications
3. Configure backend preferences as needed

## Verification Command
```bash
./verify-tornadovm.sh
```

## Usage
```bash
source ./tornadovm-local/etc/setvars.sh
tornado --devices
```
EOF

echo "=== TornadoVM Installation Complete ==="
echo "Installation directory: $(pwd)/tornadovm-local"
echo "Run './verify-tornadovm.sh' to verify installation"
echo "See ./TORNADOVM_USAGE.md for usage instructions"
```

## Claude Execution Notes

When executing this plan:

1. **Sequential Execution**: Run each phase in order, checking success before proceeding
2. **Error Handling**: If a command fails, report the error and ask user how to proceed
3. **User Confirmation**: Before installing system packages (sudo commands), confirm with user
4. **Progress Reporting**: Report completion of each major step
5. **Path Verification**: Ensure working directory is main project directory (`./`)
6. **Self-Contained Focus**: Ensure installation works independently of source directory
7. **SDK Directory**: Ensure SDK is copied as real directory, not symlink, with proper permissions
8. **Modification Check**: Always check for and apply custom modifications from ../patch/ directory before building
9. **Final Verification**: Always run the verification script at the end

## Success Criteria

Installation is successful when:
- [ ] `tornado --version` returns version information
- [ ] `tornado --devices` lists at least one device
- [ ] Both OpenCL and PTX backends built (if compatible CUDA available)
- [ ] Native libraries exist in `./tornadovm-local/lib/native/`
- [ ] API JARs exist in `./tornadovm-local/lib/`
- [ ] Environment script works independently
- [ ] Verification script passes all checks
- [ ] Installation is self-contained in `./tornadovm-local/`

## Rollback Plan

If installation fails:
```bash
# Clean up partial installation
rm -rf ./tornadovm-local
cd ./tornadovm-local-src
make clean
cd ..
# Start over from Phase 1
```

This plan provides Claude Code with all necessary commands and context to successfully install TornadoVM locally as a self-contained installation focused purely on local TornadoVM functionality.
