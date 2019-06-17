mkdir lib
g++ -O3 -shared -std=c++11 -fPIC -I./ -I./hnswlib src/FeatureDB.cpp -o lib/libfeaturedb.so
mkdir bin
g++ -std=c++11 -I./ -L./lib test/test.cpp -o bin/test -lfeaturedb
cp test/*fvecs bin
cp lib/libfeaturedb.so bin
cd bin
./test
