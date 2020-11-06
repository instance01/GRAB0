cd src
mkdir dist
cd dist

# Setup dependencies.
# libtorch
# In case an error happens here, mirror: https://mega.nz/file/tt9WDYDK#DVEtQ99AhtnhLmOQ3G4Ogc_HFgNoY6Dn4Ji9_qyw69E
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.5.0%2Bcpu.zip && unzip "libtorch-cxx11-abi-shared-with-deps-1.5.0+cpu.zip"
# nlohmann-json
git clone https://github.com/nlohmann/json.git
cd json
cmake .
# box2d
cd ..
git clone --branch instance https://github.com/Instance-contrib/box2d.git
cd box2d
./build.sh
cp ./src/box2dConfigVersion.cmake ./build/src/box2dConfig.cmake
cd ..

# Build the project.
cmake .
cmake --build .

echo 'Done. You may now run ./GRAB0 in src/.'
