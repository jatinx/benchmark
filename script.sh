#! /bin/sh
echo "Running Benchmark"
CDIR=$PWD
WDIR="$CDIR/tmpwdir"
rm -rf $WDIR
mkdir -p $WDIR
cd $WDIR
HIPDIR="$WDIR/HIP"
git clone https://github.com/ROCm-Developer-Tools/HIP.git $HIPDIR
cd $HIPDIR
mkdir "$HIPDIR/build"
HIP_PATH="$HIPDIR/build/install"
cd "$HIPDIR/build"
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/install
make -j$(nproc) install
HIPCCBIN=$PWD/install/bin/hipcc
cd $WDIR
cp -R ../GPUTest ./tests
cd tests
$HIPCCBIN main.cc -isystem "$CDIR/build/install/include" -L"$CDIR/build/install/lib" -lbenchmark -lpthread -o mainpre -O3
cd $HIPDIR
PATCH="https://github.com/ROCm-Developer-Tools/HIP/pull/$1.patch"
wget $PATCH
git apply "$1.patch"
cd build
make -j$(nproc) install
cd $WDIR
cd tests
$HIPCCBIN main.cc -isystem "$CDIR/build/install/include" -L"$CDIR/build/install/lib" -lbenchmark -lpthread -o mainpost -O3
$CDIR/tools/compare.py benchmarks ./mainpre ./mainpost
