mkdir lib
g++ -O3 -shared -std=c++11 -fPIC -I./ -I./hnswlib src/FeatureDB.cpp -o lib/FeatureDB.so
