#!/bin/bash
g++ main.cpp -o out -std=c++20 -O3 
time ./out
rm -f out
