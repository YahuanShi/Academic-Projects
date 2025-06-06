# Machine Learning in Computational Mechanics

## Course Description

This course introduces modern machine learning techniques in the context of computational mechanics. It bridges numerical simulation and artificial intelligence by using neural networks to accelerate simulations, model complex material behavior, and solve inverse problems. Students gain hands-on experience with PyTorch-based neural networks applied to elasticity, plasticity, and energy-based formulations.

---

## Learning Outcomes

After completing this course, I was able to:

### Knowledge:
- Understand categories of AI and ML in computational mechanics
- Understand the structure and functioning of neural networks
- Gain familiarity with core functionality of PyTorch and autograd

### Skills:
- Implement neural networks for regression and material modeling
- Apply CANN and DEM to elasticity and hyperelasticity problems
- Optimize and accelerate machine learning pipelines in scientific computing

### Competencies:
- Choose suitable ML methods for mechanics problems
- Embed ML models within domain-specific PDE constraints
- Evaluate model accuracy and computational efficiency

---

## Assignments

| Assignment | Folder | Description |
|------------|--------|-------------|
| Assignment 1 | [`assignment-01`](./assignment-01) | **Introduction to PyTorch and Linear Regression**. Implemented 2D nonlinear displacement visualization, linear regression with PyTorch autograd, and explained regression mathematically. Practiced basic PyTorch operations, plotting, and gradient computation. |
| Assignment 2 | [`assignment-02`](./assignment-02) | **Nonlinear Regression, Helmholtz Free Energy, and CANN**. Extended regression to multivariate setting with noise and regularization. Implemented Helmholtz free energy for Neo-Hookean material using autograd. Designed and trained a CANN model for hyperelastic material modeling and evaluated its accuracy. |
| Assignment 3 | [`assignment-03`](./assignment-03) | **Deep Energy Method (DEM), Hyperparameter Optimization, and KAN**. Analyzed provided DEM implementation, evaluated loss term influences, performed hyperparameter optimization. Replaced standard material model with Neo-Hookean and compared with CANN results. Replaced FCNN with KAN architecture and optimized for computational speed (GPU, vectorization, JIT). |

---

## Tools & Libraries

- Python 3.12
- PyTorch (torch, autograd)
- NumPy
- Matplotlib
- PyTorch Lightning (optional for advanced training loops)
- JIT acceleration tools (TorchScript, Numba)
- KAN open-source libraries (pykan, fast-kan)

---

## Repository Structure

ML-in-Computational-Mechanics/
├── assignment-01/ # PyTorch basics, linear regression, visualization
├── assignment-02/ # Nonlinear regression, Helmholtz energy, CANN
├── assignment-03/ # DEM analysis, Neo-Hookean modeling, KAN experiments
└── README.md # This file

---

## Notes

- The course provided valuable hands-on experience applying neural networks to core problems in computational mechanics.
- Special emphasis was placed on understanding the physics behind the models (e.g. hyperelasticity), as well as optimizing ML pipelines for **scientific computing use cases** (DEM, CANN, KAN).
- The assignments required careful tuning and validation to ensure physically accurate results and computational efficiency.

---

## Future Directions

- Further explore advanced neural operator methods (Fourier Neural Operators, DeepONet)
- Apply KAN to 3D solid mechanics and complex boundary conditions
- Investigate hybrid FEM-ML models combining classical solvers with learned components
- Explore mixed physics-informed neural networks for elasticity + plasticity problems

---

> This project demonstrates my ability to implement and optimize **physics-informed neural networks and ML models** for real-world scientific computing problems using modern deep learning frameworks.