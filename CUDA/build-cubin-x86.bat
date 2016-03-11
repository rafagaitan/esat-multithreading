call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" x86 
nvcc -arch=sm_30 -Xptxas="-v" --machine=32 --cubin -o %1\MatrixMult.cubin MatrixMult.cu
pause
