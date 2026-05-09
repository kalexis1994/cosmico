@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
echo RC_PATH:
where rc.exe
echo CL_PATH:
where cl.exe
echo NVCC_PATH:
where nvcc.exe
