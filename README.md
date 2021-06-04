# Blockchains Model Simulator
##### Folder structure
- src/
Contains the python source code files
- src/bcns
Contains the core classes of the simulator
- src/run\_eval.py
This is the entry point, you can run the simulator by chosing a configuration. For example:
python3 run\_eval.py -c "2" #for the P2P configuration or
python3 run\_eval.py -c "2-c" #for the Coordinated configuration

Configurations can be modified or added at src/evaluation/SimulationConfigs.py

- evaluation\_results/
contains the dataframes that contain all simulation results. These dataframes can be read and the results can be plotted using the provided notebook (PLOTTER.ipynb)

- Data of the simulations in the paper and the plotters can be found at: https://drive.google.com/drive/u/4/folders/10ychGyYtrcoShLG9UmcTFnzCHsF3EWUj

Thanks!

##### Contributors to the code

Mohamed Alzayat <alzayat@mpi-sws.org>

Johnnatan Messias <johnme@mpi-sws.org>

Balakrishnan Chandrasekaran <b.chandrasekaran@vu.nl>

We thank Antoine Kaufmann for his earl-on suggestions that influenced the simulator architecture
