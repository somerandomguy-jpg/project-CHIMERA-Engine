cd /home/deb-vik/new_engine_avx_512_fsr
mkdir -p build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
ninja -j6
