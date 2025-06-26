#!/bin/bash
# Get the Python extension suffix using Python itself (more portable than python3-config)
PYTHON_EXT_SUFFIX=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
echo "Building mesh_inpaint_processor with extension suffix: $PYTHON_EXT_SUFFIX"
c++ -O3 -Wall -shared -std=c++11 -fPIC `python -m pybind11 --includes` mesh_inpaint_processor.cpp -o mesh_inpaint_processor$PYTHON_EXT_SUFFIX