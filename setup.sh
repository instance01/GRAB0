cd src
mkdir dist
cd dist

# Setup dependencies.
# In case an error happens here, mirror: https://mega.nz/file/tt9WDYDK#DVEtQ99AhtnhLmOQ3G4Ogc_HFgNoY6Dn4Ji9_qyw69E
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.5.0%2Bcpu.zip && unzip "libtorch-cxx11-abi-shared-with-deps-1.5.0+cpu.zip"
git clone https://github.com/nlohmann/json.git
cd json
cmake .

# Build the project.
cd ../..
cmake .
cmake --build .

echo 'Done. You may now run ./GRAB0 in src/.'
