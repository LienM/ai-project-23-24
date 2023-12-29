This folder contains the complete implementation of various recommendation models using the Recpack library.

**Files:**


1. **00 - Dataset** : This folder serves as a destination for intermediate data generated during the recommendation process, such as recommended items.
2. **00 - Recpack_old** : This directory includes the initial Recpack implementations:
    - 02 - Recpack: A preliminary test of Recpack using sample data.
    - 02 - Recpack2: An attempt to decode IDs from a Compressed Sparse Row (CSR) matrix.
    - 02 - Recpack3: The successful implementation of Recpack with custom data.

3. **Clean Versions** : The following files represent improved versions of the previous implementations:
    - 02- Recpack4: A refined version of Recpack3.
    - 02 - Recpack5: A graphical implementation using an auxiliary function script named aux_recpack.

4. **Auxiliary Functions and Libraries** :
    - aux_functions: A Python script containing auxiliary functions used in the Recpack notebooks 1, 2, and 3.
    - aux_graphs: A Python script containing auxiliary graphical functions specifically designed for Recpack5.
    - aux_recpack: A Python script equivalent to Recpack4, intended for convenience and may have minor differences from Recpack4.

5. **Modified Libraries** :
    - PipelineBuilder_modified: An altered version of the PipelineBuilder class from the Recpack library.
    - Pipeline_modified: An adapted version of the Pipeline class from the Recpack library.


**IMPORTANT NOTE:**

This folder contains the implementation of various recommendation models using the Recpack library. Most of these files represent snapshots of the development process and may contain "live" changes. As such, they may not constitute fully fledged reports in the traditional sense. It is recommended to review the entire collection of files and scripts to gain a comprehensive understanding of the development process.