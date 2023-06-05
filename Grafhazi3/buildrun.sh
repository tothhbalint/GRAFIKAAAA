#!/bin/bash
# hozzuk létre a build mappát - ha már létezne, akkor nem csinál semmit parancs
mkdir build
# lépjünk bele a build mappába
cd build
# generáljuk ki cmake-kel a build fájlokat a szülő könyvtárból
cmake ..
# buildeljük a programot
make
#futtatjuk a programot
./Skeleton
#visszalépünk a projekt gyökérmappájába - ahol eredetileg voltunk
cd ..

