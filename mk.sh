echo "cd HierarchicalNSW/gemv"
cd HierarchicalNSW/gemv
echo "mkdir bin"
mkdir bin
echo "g++ -std=c++11 codegen/codegen_gemv_25_10.cpp -o bin/codegen_gemv_25_10"
g++ -std=c++11 codegen/codegen_gemv_25_10.cpp -o bin/codegen_gemv_25_10
echo "g++ -std=c++11 codegen/codegen_kernel.cpp -o bin/codegen_kernel"
g++ -std=c++11 codegen/codegen_kernel.cpp -o bin/codegen_kernel
echo "./bin/codegen_gemv_25_10"
./bin/codegen_gemv_25_10
echo "./bin/codegen_kernel"
./bin/codegen_kernel
echo "mv GemvKernel.h ./include"
mv GemvKernel.h ./include
echo "mv AVXKernel.h ./include"
mv AVXKernel.h ./include
echo "mkdir obj"
mkdir ../../obj
echo "compile *.o"
g++ -std=c++11 -masm=intel -fPIC  -I./include -c GemvKernel.cc -o ../../obj/GemvKernel.o
g++ -std=c++11 -masm=intel -fPIC  -I./include -c AVXKernel.cc -o  ../../obj/AVXKernel.o
g++ -std=c++11 -fPIC -I./include -c src/cpuinfo.cc -o ../../obj/cpuinfo.o
g++ -std=c++11 -fPIC -I./include -c src/kernel.cc -o  ../../obj/kernel.o
g++ -std=c++11 -fPIC -I./include -c src/gemv.cc -o  ../../obj/gemv.o
g++ -std=c++11 -fPIC -I./include -c src/int8.cc -o  ../../obj/int8.o
g++ -std=c++11 -fPIC -I./include -c src/util.cpp -o ../../obj/util.o
echo "compile *.so"
cd ../../
g++ -O3 -shared -std=c++11 -fPIC -I./ -I./HierarchicalNSW -I./HierarchicalNSW/gemv/include src/FeatureDB.cpp \
obj/GemvKernel.o obj/AVXKernel.o obj/cpuinfo.o obj/kernel.o obj/gemv.o obj/int8.o obj/util.o -o lib/libfeaturedb.so
echo "cleaning ..."
rm -r obj
rm -r HierarchicalNSW/gemv/bin
rm HierarchicalNSW/gemv/*.cc
rm HierarchicalNSW/gemv/include/*Kernel*
mkdir bin
g++ -std=c++11 -I./ -L./lib test/test.cpp -o bin/test -lfeaturedb -fopenmp
cp test/*fvecs bin
cp lib/libfeaturedb.so bin
cd bin
./test
