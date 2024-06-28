# Water Strider Optimization for FIR Filter Coefficient Optimization

This repository contains an implementation of the Water Strider Optimization (WSO) algorithm for optimizing the coefficients of a linear phase Finite Impulse Response (FIR) filter. The WSO algorithm is inspired by the behavior of water striders and is used here to design low-pass FIR filters with minimal error.

## Table of Contents

- [Water Strider Optimization for FIR Filter Coefficient Optimization](#water-strider-optimization-for-fir-filter-coefficient-optimization)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Parameters](#parameters)
  - [Running the Optimization](#running-the-optimization)
  - [Results](#results)
  - [Contributing](#contributing)
  - [License](#license)

## Introduction

The Water Strider Optimization (WSO) algorithm is a nature-inspired optimization technique. This implementation aims to optimize the coefficients of a linear phase FIR filter to achieve desired frequency response characteristics. The WSO algorithm simulates the survival and mating behavior of water striders to find optimal solutions in a given search space.

## Installation

To run this code, you'll need Python 3.x and the following Python packages:

- `numpy`
- `matplotlib`
- `scipy`

You can install these packages using `pip`:

```sh
pip install numpy matplotlib scipy
```

## Usage

Clone the repository and navigate to the project directory:

```sh
git clone https://github.com/yourusername/WSO-FIR-Filter.git
cd WSO-FIR-Filter
```

Run the optimization script:

```sh
python optimize_fir_filter.py
```

## Parameters

The WSO algorithm allows customization through various parameters:

- `pop_size`: Population size (number of water striders)
- `dim`: Dimension of the problem (number of FIR filter coefficients)
- `max_iter`: Maximum number of iterations
- `inertia_weight`: Inertia weight for velocity update
- `cognitive_coeff`: Cognitive coefficient for personal best influence
- `social_coeff`: Social coefficient for global best influence
- `bounds`: Tuple representing the bounds for coefficients
- `omega_pass`: Passband edge frequency (normalized)
- `omega_stop`: Stopband edge frequency (normalized)
- `alpha`: Phase offset (unused in this implementation)
- `tau`: Group delay (unused in this implementation)
- `W`: Weighting function
- `delta_pass`: Passband ripple
- `delta_stop`: Stopband ripple
- `nte`: Number of territories
- `ar`: Attraction response probability
- `initial_population`: Initial population of FIR filter coefficients (optional)

## Running the Optimization

The optimization process can be run multiple times to visualize the results across different runs. The following example demonstrates how to run the optimization for 10 iterations:

```python
pop_size = 50
dim = 21  # Number of coefficients in the FIR filter
max_iter = 100
inertia_weight = 0.5
cognitive_coeff = 1.5
social_coeff = 1.5
bounds = (-1, 1)
omega_pass = 0.44
omega_stop = 0.55
alpha = 0
tau = 10
W = lambda omega: 1  # Uniform weighting function as an example
delta_pass = 0.01  # Example pass-band ripple
delta_stop = 0.01  # Example stop-band ripple
nte = 5  # Number of territories
ar = 0.7  # Attraction response probability

# Initial Ideal Filter Coefficients
N1 = [
    0.0388639813481993, 0.00260088729777498, -0.0302244308396995, -0.0180939407775842,
    0.0356920644947972, 0.0393786067312622, -0.0450132314893481, -0.0923298471847741,
    0.0471816860342088, 0.311919757687732, 0.448804967338731, 0.311919757687732,
    0.0471816860342088, -0.0923298471847741, -0.0450132314893481, 0.0393786067312622,
    0.0356920644947972, -0.0180939407775842, -0.0302244308396995, 0.00260088729777498,
    0.0388639813481993
]

# Convert N1 to a numpy array and initialize WSO with it
N1 = np.array(N1)
wso = WaterStriderOptimization(
    pop_size, dim, max_iter, inertia_weight, cognitive_coeff, social_coeff, bounds,
    omega_pass, omega_stop, alpha, tau, W, delta_pass, delta_stop, nte, ar,
    initial_population=N1
)
wso.run_multiple(n_runs=10)
```

## Results

The optimized FIR filter coefficients are printed, and the frequency response plots for each run are displayed in subplots. These visualizations help in understanding the performance and consistency of the WSO algorithm.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details.
