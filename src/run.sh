g++ -ggdb -std=c++11 `pkg-config --cflags opencv` -o `basename detection.cpp .cpp` detection.cpp `pkg-config --libs opencv`
