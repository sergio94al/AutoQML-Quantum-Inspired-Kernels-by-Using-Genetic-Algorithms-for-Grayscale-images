# AutoQML-Quantum-Inspired-Kernels-by-Using-Genetic-Algorithms-for-Grayscale-images

This is the official repository of the article *AutoQML: Automatic Generation and Training of Robust Quantum-Inspired Classifiers by Using Evolutionary Algorithms on Grayscale Images*. This article discusses the classification of grayscale images using quantum circuits. Given that images represent high-dimensional data and quantum circuits face challenges in handling them, a strategy is implemented in which dimensionality reduction methods are added.

DOI: https://arxiv.org/abs/2208.13246


### Abstract

A new hybrid system is proposed for automatically generating and training quantum-inspired classifiers on grayscale images by using multiobjective genetic algorithms. It is defined a dynamic fitness function to obtain the smallest circuit complexity and highest accuracy on unseen data, ensuring that the proposed technique is generalizable and robust. At the same time, it is minimized the
complexity of the generated circuits in terms of the number of entangling operators by penalizing their appearance and number of gates. The size of the images is reduced by using two dimensionality reduction approaches: principal component analysis (PCA), which is encoded within the individual and genetically optimized by the system, and a small convolutional autoencoder (CAE). These two
methods are compared with one another and with a classical nonlinear approach to understand their behaviors and to ensure that the classification ability is due to the quantum circuit and not the preprocessing technique used for dimensionality reduction.

## 0. Goals of the Technique

* **Maximize the accuracy on unseen data**.
* **Minimize the quantum classifier size**, in terms of quantum gates, layers and number of qubits, thus, reducing the expressivity of the quantum circuits.
* **Optimization of the circuit structure, gate types and its parameters *θ***.
* Generate an **automatic and optimized system for data encoding** of classical information into the quantum feature maps.
* Take into account the use case, generating **ad-hoc classifiers for each data set**.
* **PCA components optimization** in the genetic algorithm.
* Application of **quantum circuits on high dimensionality** data such as images.
* Search of **quantum-inspired solutions** that can be implemented on classical computers.
* Capacity to include **many variables in few qubits**.




