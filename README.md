# Neural BRDF Reconstruction

This repository contains the code for reconstruction of a full BRDF from NBRDFs obtained by [Neural BRDF](https://github.com/asztr/Neural-BRDF/tree/main) model.
The code is fully written in pytorch based on the implementation provided [here](https://github.com/asztr/Neural-BRDF/tree/main/binary_to_nbrdf/pytorch_code). The ```common.py``` is rewritten to support pytorch.

You can first train an NBRDF given the input binary file and then, reconstruct the full BRDF in MERL format.
