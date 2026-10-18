#!/bin/bash
# Build the MeshProcess third-party binaries on yj-4 at the commits MeshProcess pins.
# CoACD 2e4842c (JYChen18 fork) -> third_party/CoACD/build/main, ACVD b65c74a -> third_party/ACVD/bin/ACVD.
set -euxo pipefail
W=/home/nas5/sibeenkim/work/_graspoc/dexonomy
CONDA=/home/nas5/sibeenkim/anaconda3
ENV=$CONDA/envs/graspoc_meshbuild

if [ ! -d $ENV ]; then
  $CONDA/bin/conda create -y -p $ENV -c conda-forge --override-channels \
    "cmake>=3.24,<4" make cxx-compiler c-compiler vtk-base git
fi
$CONDA/bin/conda install -y -p $ENV -c conda-forge --override-channels "cmake>=3.24,<4"
# ACVD gets its own env with VTK 9.2, the version the MeshProcess README builds against. conda-forge
# VTK 9.7 splits dependency headers into -devel packages its cmake config then fails to find.
ENV2=$CONDA/envs/graspoc_acvdbuild
if [ ! -d $ENV2 ]; then
  $CONDA/bin/conda create -y -p $ENV2 -c conda-forge --override-channels \
    python=3.10 "vtk-base=9.2.6" "cmake>=3.24,<4" make cxx-compiler c-compiler
fi
# vtk-base exports VTK::IOFFMPEG in its cmake targets but ships that library in vtk-io-ffmpeg.
# Its Qt5 find wants GL/gl.h, in libgl-devel, and its Boost find wants the 1.85.0 headers.
$CONDA/bin/conda install -y -p $ENV2 -c conda-forge --override-channels "vtk-base=9.2.6" "vtk-io-ffmpeg=9.2.6" libgl-devel "libboost-devel=1.85.0"
export PATH=$ENV/bin:$PATH
cmake --version

cd $W/MeshProcess/third_party
rmdir CoACD ACVD 2>/dev/null || true
[ -d CoACD/.git ] || mv $W/CoACD CoACD
[ -d ACVD/.git ] || mv $W/ACVD ACVD
git -C CoACD checkout -q 2e4842cec9aed2e25d5766e654eb02cc62c1769c
git -C CoACD submodule update --init --recursive
git -C ACVD checkout -q b65c74ab3f4d73e0df1936c1df00bb249e66f14d

cd CoACD
if [ ! -x build/main ]; then
  rm -rf build && mkdir -p build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release
  make -j32 main
  cd ..
fi
ls -la build/main

cd ../ACVD
export PATH=$ENV2/bin:/usr/bin:/bin
# VTK 9.2.6's exported OpenGL target uses $ENV{CONDA_BUILD_SYSROOT}/usr/lib/libGL.so and
# .../usr/include. Point that variable at a shim holding the env's own libglvnd libGL and headers.
mkdir -p $W/acvd_sysroot/usr/lib
ln -sfn $ENV2/lib/libGL.so $W/acvd_sysroot/usr/lib/libGL.so
ln -sfn $ENV2/include $W/acvd_sysroot/usr/include
export CONDA_BUILD_SYSROOT=$W/acvd_sysroot
rm -rf CMakeCache.txt CMakeFiles
cmake . -DCMAKE_BUILD_TYPE=Release -DVTK_DIR=$(ls -d $ENV2/lib/cmake/vtk-*) -DPython3_EXECUTABLE=$ENV2/bin/python -DPython3_ROOT_DIR=$ENV2
make -j32
ls -la bin/ACVD
echo BUILD_DONE
