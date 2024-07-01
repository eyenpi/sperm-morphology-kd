# Less-supervised learning with knowledge distillation for sperm morphology analysis

This project is the official implementation of the paper ["Less-supervised learning with knowledge distillation for sperm morphology analysis"](https://doi.org/10.1080/21681163.2024.2347978).

## Configuration

You can modify the hyperparameters and other settings in the `config.py` file.

## Project Structure

- `data/`: Directory for storing the MHSMA dataset
- `models/`: Contains the VGG and custom VGG model implementations
- `utils/`: Utility functions for data loading, loss calculation, and attacks
- `train.py`: Script for training the model
- `test.py`: Script for testing the model
- `config.py`: Configuration file with hyperparameters and settings

# Data Directory

Place the MHSMA dataset files in this directory. The expected files are:

- `x_64_train.npy`
- `x_64_valid.npy`
- `x_64_test.npy`
- `y_acrosome_train.npy`
- `y_acrosome_valid.npy`
- `y_acrosome_test.npy`

Make sure to download these files from the official MHSMA dataset source and place them in this directory before running the training or testing scripts.

## Citation

If you use this code in your research, please cite the following paper:

```bibtex
@article{doi:10.1080/21681163.2024.2347978,
        author = {Ali Nabipour, Mohammad Javad Shams Nejati, Yasaman Boreshban and Seyed Abolghasem Mirroshandel},
        title = {Less-supervised learning with knowledge distillation for sperm morphology analysis},
        journal = {Computer Methods in Biomechanics and Biomedical Engineering: Imaging \& Visualization},
        volume = {12},
        number = {1},
        pages = {2347978},
        year = {2024},
        publisher = {Taylor \& Francis},
        doi = {10.1080/21681163.2024.2347978},
        URL = {https://doi.org/10.1080/21681163.2024.2347978}
}
```