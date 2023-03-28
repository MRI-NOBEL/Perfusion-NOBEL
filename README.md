# Perfusion-NOBEL

Pre-clinical DSC-MRI perfusion map generator written in Python. Optimized for the analysis of brain disease models in rats.


## Getting Started

This repository is a work in progress. Both the repository structure and source code are subjected to changes in order to make everythin more comprehensive and easy to use and edit.

An user manual, including a tutorial on how to obtain all the maps, will also be available soon.

### Prerequisites

WIP


### Installing 

Perfusion-NOBEL repository is still in its early stages of development. As of now, the best way to run the code would be through a Python interpreter (Spyder, PyCharm, ...) but in the near future we will release a more user-friendly version that doesnt need any external software to run. 


### Running the code

Perfusion-NOBEL is structured in three main files, which can be found inside [Perfusion-NOBEL](Perfusion-NOBEL). All three must be located in the same directory to avoid import errors.
  - [InitialParametersClass.py](InitialParametersClass.py): Module including all the parameters and data needed to perform the analysis. In case of needing to change any parameter outside the ones requested directly by the software, all of them can be found here, from MR and sequence-characteristic parameters (TE, time between frames, ...), tisular constants (Apparent brain density, hematocrit ratio, ...), to any internal parameter used by the project to run the calculations.
  - [PerfusionClass.py](PerfusionClass.py): Class where all the perfusion methods are defined. Everything regarding to the mathematical moddel and its implementation can be found in this file.
  - [PerfusionMain.py](PerfusionMain.py): As the name indicactes, this is the main script of the project and the one that should be executed. The deconvolution method is selected in this file by commenting/uncommenting the corresponding lines that include the keys.


## Contributors

  - See the list of [contributors](https://github.com/MRI-NOBEL/Perfusion-NOBEL/contributors) who participated in this project.

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE.md)
 - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

WIP

 
 
