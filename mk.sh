mkdir lib
g++ -O3 -shared -std=c++11 -fPIC -I./ -I./hnswlib src/FeatureDB.cpp -o lib/libfeaturedb.so
mkdir bin
g++ -std=c++11 -I./ -L./lib test/test.cpp -o bin/test -lfeaturedb
cp test/*fvecs bin
export LD_LIBRARY_PARH=./lib:$LD_LIBRARY_PATH
cd bin
./test
