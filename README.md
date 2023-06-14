An example of a program that solves the Euler equations of fluid dynamics using an explicit time integrator with the matrix-free framework applied to a high-order discontinuous Galerkin discretizatoin in space. 

In the file `main.cpp` we can select different test cases and running the progrma with the corresponding inputfile.
- backward facing step test case -> `inputfile_bstep.prm`
- forward facing step test case"-> `inputfile_fstep.prm`
- sod-Shock problem -> `inputfile_sod.prm`
- supersonic flow past a circular cylinder test -> `inputfile_cylinder.prm`
- double-mach reflection problem; -> `inputfile_DMR.prm`
- 2D Riemann problem -> `inputfile_2Driemann.prm`
    
In each input file there are parameters that we can change in order to simulate the testcase with different setting.
We can change the type of limiter (TVB or filtering procedures), the limiter TVB function (minmod or vanAlbada), the refine (do refine or not do refine), the refinement indicator (gradient of density or gradient of pressure), the numerical flux (Lax, HLL, HLLC, Roe, SLAU)

Other files in the directory:
- `parameters *.h *.cpp` 
- `equationData *.h *.cpp` 
- `eulerproblem *.h *.cpp` 
- `euleroperator *.h *.cpp`
- `operation.h`

In the file `refman.pdf` there is all the description of the code Eulero. 
The output results are stored in `../Eulero/build/results`
In the `build_Eulero` folder, download from `https://github.com/Felotti/Eulero.git`, there are the simulations obtained for the project.

To generate a Makefile for this code using  Cmake, type the following command into the terminal from the main directory Eulero
``` 
mkdir build 
```
``` 
cmake -S ../Eulero -B ../Eulero/build
```
``` 
cd build
```
```
cmake --build .
```
```
mkdir results
```
``` 
./main "../inputfile_***.prm"
```

For this last line, change the name of the `inputfile_***.prm`, choosing the name described before.

