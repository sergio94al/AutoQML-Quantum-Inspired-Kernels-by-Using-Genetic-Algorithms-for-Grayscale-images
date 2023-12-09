# AutoQML: Automatic Quantum-Inspired Kernels by Using Genetic Algorithms for Grayscale images

This is the official repository of the article *AutoQML: Automatic Generation and Training of Robust Quantum-Inspired Classifiers by Using Evolutionary Algorithms on Grayscale Images*. This article discusses the classification of grayscale images using quantum circuits. Given that images represent high-dimensional data and quantum circuits face challenges in handling them, a strategy is implemented in which dimensionality reduction methods are added.

DOI: https://arxiv.org/abs/2208.13246


### Abstract

A new hybrid system is proposed for automatically generating and training quantum-inspired classifiers on grayscale images by using multiobjective genetic algorithms. It is defined a dynamic fitness function to obtain the smallest circuit complexity and highest accuracy on unseen data, ensuring that the proposed technique is generalizable and robust. At the same time, it is minimized the
complexity of the generated circuits in terms of the number of entangling operators by penalizing their appearance and number of gates. The size of the images is reduced by using two dimensionality reduction approaches: principal component analysis (PCA), which is encoded within the individual and genetically optimized by the system, and a small convolutional autoencoder (CAE). These two
methods are compared with one another and with a classical nonlinear approach to understand their behaviors and to ensure that the classification ability is due to the quantum circuit and not the preprocessing technique used for dimensionality reduction.

## 0. Goals of the Technique

* **Maximize the accuracy on unseen data**.
* **Minimize the quantum classifier size**, in terms of quantum gates, layers and number of qubits, thus, reducing the expressivity of the quantum circuits.
* **Generation of circuit structure, gate types and its parameters *θ***.
* Generate an **automatic and optimized system for data encoding** of classical information into the quantum feature maps.
* Take into account the use case, generating **ad-hoc classifiers for each data set**.
* **PCA components optimization** within the individuals of the genetic algorithm.
* Application of **quantum circuits on high dimensionality** data such as images.
* Search of **quantum-inspired solutions** that can be implemented on classical computers.
* Capacity to include **many variables in few qubits**.

## 1. Description

In this paper we propose a novel technique for **quantum machine learning** (QML) which allows the **automatic generation of quantum-inspired kernels for classification** by using Quantum Support Vector Machine (QSVM), based on Multi-Objective Genetic Algorithms (MO-GA) for **grayscale-image datasets**.

The goal of the technique is to achieve the quantum circuit that provides the **best accuracy in classificationon test data**, as well as the **smallest ansatz size**, without compromising precision. Since we train the models with training data and the fitness function objective is the accuracy on test data, we force the circuits-solution to be robust and to **avoid overfitting effects, being quantum classifiers with generalization power**. 

Taking into account the ansatz size, our goal is to minimize it as much as possible in order to have solutions that avoid expressivity problems. This is possible because we code identity gates, which allows the **possibility of eliminating gates, layers and even reduce the number of qubits in the circuits**.


## 2. Quantum circuits Encoding
Quantum circuits are encoded in binary strings, with each binary string composed of MxNx7 bits, where M and N represent the user-defined number of qubits and layers. The string is divided into groups of seven, where each group defines a quantum gate. The first three bits in each group specify the gate type, while the last four bits determine the rotation angles. Consequently, there are $2^3$ possible gates and $2^4$ rotation angles in this encoding scheme, improving our last encoding [1].

<p align="center">
  <img src="https://github.com/sergio94al/AutoQML-Quantum-Inspired-Kernels-by-Using-Genetic-Algorithms-for-Grayscale-images/blob/main/coding_quantum_gates.png" width="300" height="450">
</p>

In this encoding, not only gates with embedded variables are encoded and used, but also gates with fixed rotations are encoded to provide greater flexibility in classifier generation. Additionally, entanglement gates between two consecutive qubits and identity operators, which do not alter quantum states and thus allow for the reduction of circuit complexities, are also encoded.










[1] https://github.com/sergio94al/Automatic_design_of_quantum_feature_maps_Genetic_Auto-Generation










### 2.1 Quantum Circuits Optimization Algorithm

* **Step 1**: Firstly, quantum gates H, CNOT and parameterized in the X,Y,Z axes with four associated angles are pre-coded to binary code. Each gate is coded into five bits, being the first three bits for gate selection and the last two bits for angle if necessary. During the process, binary strings (individuals) are created, which will encode for a specific ansatz.
* **Step 2**: A starting population is created -Initial population.
* **Step 3**: These individuals are evaluated in the **evaluation function or *fitness***. The output of this function will determine whether the individual is accurate for the given problem or not. In the proposed technique, the **binary strings are converted into quantum circuits** which will act as feature maps into QSVM. Firstly, the classifier is fitted with training set and then we make predictions over test set (data not previously seen by the model) **-seeking generalization power-**, getting the objetive of the fitness function. At the same time, we calculate the number of gates penalizing doubly the entangling operators due to a higher computational cost. We calculate a metric -Weight Control- in order to find a **balance between both metrics**, the accuracy and the reduction of number of gates. It is important since a high weight on the reducing circuit size objetive can lead less accuracy because of information loss.
* **Step 4**: We select the best individuals. We apply **genetic operators** of crossover (Two-points) and mutation (Flipbit), generating new individuals (offspring) for the next generation. These operators are applied with a probability *Pm* and *Pc* respectively. The mutation operator allows us to reach other points in the search space since it allows us to **avoid local minima**, making the search for the best solution more efficient.
* **Step 5**: The process is repeated until convergence or when stop conditions are achieved. **The best individuals are kept in the Pareto front**.

## Multi-Objective Genetic Algorithms (MO-GA)

## 5. Files Description

* circuit.py: We create the quantum operators that will composed the quantum circuit.
* fitness.py: Evaluation fuction of the genetic algorithm (we fit 2 variables to return -the objetives)
* gsvm.py: Genetic algorithm function with the genetic operators. We call the fitness function.
* qsvm.py: We create a simulated quantum support vector machine by using sklearn.
* encoding.py: In this file we create the encoding of the quantum gates and the parameters *θ*.
* encoding2.py: This file is used to visualize and obtain the solution after the evolution.
