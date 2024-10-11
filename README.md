# ANaGRAM: A Natural Gradient Relative to Adapted Model for efficient PINNs learning
## Settings instructions
This code is based on the code of E-NGD available at <https://github.com/MariusZeinhofer/Natural-Gradient-PINNs-ICML23>.
For the needs of this paper, we forked it and made it available anonymously at <https://anonymous.4open.science/r/Natural-Gradient-PINNs-ICML23-3815/>.
You have to download it first in the folder Natural-Gradient-PINNs-ICML23.

## Running experiments
All experiments can be found in the “experiments” path. Scripts with the suffix “expes” are those to be used to reproduce the results of the article.
For ENGD and ANaGRAM, simply run the corresponding script directly, e.g. for the 2D Laplace equation:
- For ANaGRAM:
`python anagram_laplace_2d_expes.py`
- For ENGD:
`python engd_laplace_2d_expes.py`

For Adam, GD, L-BFGS, use the interface to pass corresponding options (see below), e.g. for the 2D Laplace equation:
- For Adam:
`python anagram_laplace_2d_expes.py --nsteps 20001 --expe_name adam_laplace_2d_expes --optimizer adam`
- For GD:
`python anagram_laplace_2d_expes.py --nsteps 20001 --expe_name sgd_laplace_2d_expes --optimizer sgd`
- For L-BFGS:
`python anagram_laplace_2d_expes.py --nsteps 2001 --expe_name lbfgs_laplace_2d_expes --optimizer lbfgs`

By default, logs and final weights will be saved in the experiments-results subfolder.

## Interface
Each script has the following interface (examplified here on laplace 2D equation):

```
usage: anagram_laplace_2d_expes.py [-h] [-ls LAYER_SIZES [LAYER_SIZES ...]] [-exp EXPE_NAME] [-p PATH] [-tbp TENSORBOARD_PATH] [-sfw] [-lw LOG_WEIGHTS] [-lsv LOG_SINGULAR_VALUES]
                                   [-tb TENSORBOARD] [-NNTK NNTK_PLOT] [-NTK NTK_PLOT] [-vb VERBOSITY] [-ns NSTEPS] [--seed SEED] [--rcond RCOND] [-rabs] [-lrk] [-lbsv]
                                   [--log_proportion_last_layer] [-opt {adam,sgd,anagram,adam-lbfgs,lbfgs,engd}]

options:
  -h, --help            show this help message and exit
  -ls LAYER_SIZES [LAYER_SIZES ...], --layer_sizes LAYER_SIZES [LAYER_SIZES ...]
                        Defines the MLP architecture as a sequence of layers sizes
  -exp EXPE_NAME, --expe_name EXPE_NAME
                        Name of the experiment
  -p PATH, --path PATH  Path into which the experiments potential outputs should be stored
  -tbp TENSORBOARD_PATH, --tensorboard_path TENSORBOARD_PATH
                        Path into which the tensorboard logs should be stored
  -sfw, --save_final_weights
                        Save final weights of the neural network
  -lw LOG_WEIGHTS, --log_weights LOG_WEIGHTS
                        Save weights of the neural network every n steps [0 means never]
  -lsv LOG_SINGULAR_VALUES, --log_singular_values LOG_SINGULAR_VALUES
                        Save singular values of the svd used for natural gradient every n steps [0 means never]
  -tb TENSORBOARD, --tensorboard TENSORBOARD
                        Store the training metrics in tensorboard every n steps [0 means never]
  -NNTK NNTK_PLOT, --NNTK_plot NNTK_PLOT
                        Plot the Natural Neural Tangent Kernel of the network every n steps [0 means never]
  -NTK NTK_PLOT, --NTK_plot NTK_PLOT
                        Plot the Neural Tangent Kernel of the network every n steps [0 means never]
  -vb VERBOSITY, --verbosity VERBOSITY
                        Plot resutls on console every n steps [0 means never]
  -ns NSTEPS, --nsteps NSTEPS
                        Number of optimization steps
  --seed SEED           Seed to be used
  --rcond RCOND         The rcond for the spectral cutoff in anagram [negative or null value means default value]
  -rabs, --rcond_absolute
                        If set, then rcond is taken as an absolute value and not relative to biggest singular value
  -lrk, --log_svd_rank  Log the rank of the svd used for natural gradient in tensorboard
  -lbsv, --log_biggest_singular_value
                        Log the biggest singular value of the svd used for natural gradient in tensorboard
  --log_proportion_last_layer
                        Log the norm proportion of the last layer update in tensorboard
  -opt {adam,sgd,anagram,lbfgs,engd}, --optimizer {adam,sgd,anagram,lbfgs,engd}
                        Specify which optimizer should be used
```
