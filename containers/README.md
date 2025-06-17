# Best Practices

## Building Containers:
In this repo, each piece of software is installed into a Docker container. From this docker container, and Apptainer is built.
- The repo structure will be `containers/[Software]/[Version#]`
- Each directory will contain a dockerfile (`Dockerfile`) and an apptainer definition (`definition.def`).
- It will also contain a README with implementation notes, as well as example Docker and Apptainer instructions to run the software.
- They may optionally contain example data.

To update a container (for instance, for a new software release), create a new directory titled with the new verison.
- The container version number should match the given software number. If you need to make more container releases between software updates, an additional number can be appended.
- If the software does not have a version number, the version number should be the ~date~ of the container release in Y.M.D format (e.g. 25.3.1).

## Templates
Dockerfile and definition.def templates are in the `templates` folder.