# Ribbon-Containers
Repository of build information for Ribbon Docker/Apptainer containers.

Each container should have one primary piece of software.

# Container Naming Convention
In this repository, under `containers/`, the files to build and test each container will be under a directory of that containers name, with subdirectories for each version number.

**Container Names**:
- Each container should have the name of the *primary software* it contains. While the directory will have the proper capitalization, the name of the uploaded container will be in all lowercase.

**Version Names**
- Each container should have a version number matching the version number of the primary software in the container.
- If a container is updated before the software is, you may append another decimal to create sub-versions (e.g. 5.0.3 -> 5.0.3.1, 5.0.3.2, ...)
- If the software **does not have a version**, the  version number will be the *date* in YY.M.D format (e.g. 25.3.1 for March 1st, 2025). 

**Final Uploaded names**
- The final uploaded name shall be the lowercase software name. (e.g. `chai-1`)
- For apptainer images, the tag will be the version number.  (e.g. `0.6.1`)
- For docker images, the tag will be "docker-" with the version number. (e.g. `docker-0.6.1`)

# READMEs

Each version should have a readme containing usage instructions (with optional test data), implementation notes, and To Dos. Example:

### Usage
- Docker: `[command]`
- Apptainer: `[command]`


### Implementation Notes
- Notes here...

### To Do:
- To Dos here...