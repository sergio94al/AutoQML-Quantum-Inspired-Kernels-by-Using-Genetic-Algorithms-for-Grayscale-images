<p align="center">
    <img src="https://github.com/sergio94al/AutoQML-Quantum-Inspired-Kernels-by-Using-Genetic-Algorithms-for-Grayscale-images/blob/main/images/lop.png" width="700" height="200">
</p>



# AutoQML: Automatic Generation of Optimized Quantum-Inspired Classifiers by Using Evolutionary Algorithms for Grayscale Images

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

The goal of the technique is to achieve the quantum circuit that provides the **best accuracy in classification on test data**, as well as the **smallest ansatz size**, without compromising precision. Since we train the models with training data and the fitness function objective is the accuracy on test data, we force the circuits-solution to be robust and to **avoid overfitting effects, being quantum classifiers with generalization power**. 

We incorporate a dimensionality reduction method for images into the individuals' encoding, allowing the quantum circuit to be generated ad-hoc for the number of variables. This **optimization simultaneously enhances both the quantum circuit and the dimensionality reduction method**.

Taking into account the ansatz size, our goal is to minimize it as much as possible in order to have solutions that avoid expressivity problems. This is possible because we code identity gates, which allows the **possibility of eliminating gates, layers and even reduce the number of qubits in the circuits**.


## 2. Quantum Circuits Encoding
Quantum circuits are encoded in binary strings, with each binary string composed of MxNx7 bits, where M and N represent the user-defined number of qubits and layers. The string is divided into groups of seven, where each group defines a quantum gate. The first three bits in each group specify the gate type, while the last four bits determine the rotation angles. Consequently, there are $2^3$ possible gates and $2^4$ rotation angles in this encoding scheme, improving our last encoding [1,2]. In this encoding, not only gates with embedded variables are encoded and used, but also gates with fixed rotations are encoded to provide greater flexibility in classifier generation. Additionally, entanglement gates between two consecutive qubits and identity operators, which do not alter quantum states and thus allow for the reduction of circuit complexities, are also encoded.

<p align="center">
  <img src="https://github.com/sergio94al/AutoQML-Quantum-Inspired-Kernels-by-Using-Genetic-Algorithms-for-Grayscale-images/blob/main/images/codes_2.png" width="850" height="350">
</p>

In our strategy for applying this technique to grayscale images, we integrate a dimensionality reduction method directly into the individual, employing Principal Component Analysis (PCA). Thus, while the quantum circuit is constructed as a binary string of MxNx7, in this instance, we extend the string length to (MxNx7)+7 bits. Within these additional seven bits, the initial six are used to determine the number of PCA components by converting the binary string into an integer, with a range spanning from 0 to 64 features.

<p align="center">
  <img src="https://github.com/sergio94al/AutoQML-Quantum-Inspired-Kernels-by-Using-Genetic-Algorithms-for-Grayscale-images/blob/main/images/genetic code.png" width="300" height="350">
</p>

In this figure, the genetic code used for generating quantum circuits through gates and rotation angles is depicted. It is essential to take into account that **circuits are constructed sequentially, layer by layer**. The first gate is placed on the first qubit in the first layer, the second gate on the second qubit of the first layer, and so forth, until reaching the predetermined maximum number of qubits (M). At that point, the process restarts with the first qubit of the second layer. This sequence continues until the entire binary string of the individual is completed.



## 3. Multi-Objective Genetic Algorithms (MO-GA)

### 3.1. NSGA-II and Genetic Operators

A genetic algorithm is an optimization and search technique based on biological evolution. It is a stochastic process that employs genetic operators such as selection, crossover and mutation to explore and enhance solutions to a given problem. In this approach, the problem is modeled as a solution space, where each potential solution is represented as a chromosome —a string of genes.

In this methodology, we use NSGA-II [4]. Non-dominated Sorting Genetic Algorithm II, is a multi-objective genetic algorithm used to optimize problems with multiple criteria. This algorithm categorizes solutions into layers of non-dominance, seeking Pareto-optimal solutions that cannot be improved in one criterion without worsening at least one other criterion, called **non-pareto dominated**. It utilizes genetic operators such as selection, crossover, and mutation to guide evolution toward high-quality solutions and maintain diversity in the population. In terms of selection, NSGA-II often employs a method called fitness *tournament selection*. In this selection approach, several individuals are randomly chosen from the population, and their fitness values are compared. The individual with the best fitness within the group is chosen as a parent for reproduction. This process is repeated to form the population of offspring. Fitness tournament selection favors the choice of high-fitness solutions while still allowing lower-fitness individuals the chance to reproduce, contributing to the preservation of genetic diversity in the population during the algorithm's evolution.

The process begins with the random creation of an initial population of potential solutions. Each individual in the population represents a potential solution to the problem, and its quality is evaluated using a fitness function that measures how good that solution is. Selection simulates the process of natural selection, where fitter individuals are more likely to reproduce and pass on their characteristics.

Genetic operators include crossover, which combines the features of two individuals to produce new offspring. In this strategy, we use *two-point crossover*as crossover operator applied to two parent chromosomes. Two random crossover points are selected along the gene chains of both parents, and the segments between these points are exchanged to create two offspring (children). This operator simulates the idea of exchanging genetic material between two parents to generate offspring.

Mutation introduces random changes to an individual's genes. In this context, we use *flip bit mutation* as mutation operator applied to a chromosome -solution, by flipping the value of one or more randomly selected bits in the gene chain. This operator simulates the idea of randomly changing some genes in a chromosome to **explore new regions of the search space**. These operators help explore and diversify the solution space. The evolutionary process is repeated over several generations, and in each iteration, the population evolves toward more promising solutions. Elitism can be incorporated to preserve the best individuals from one generation to the next.

### 3.2. Fitness Function
The fitness function plays a crucial role in assessing how well an individual performs for the given problem. The objectives we have defined to evaluate classifiers include accuracy on unseen data (test data) during model training, aiming for greater model generalization, and circuit complexity, determined by the number and intricacy of gates.

We have defined two dimensionality reduction methods. One embeds the method directly into the individual for optimization, while the other utilizes a trained model to reduce the data's dimensionality to 64 features in the latent space, employing a small Convolutional AutoEncoder.

<p align="center">
  <img src="https://github.com/sergio94al/AutoQML-Quantum-Inspired-Kernels-by-Using-Genetic-Algorithms-for-Grayscale-images/blob/main/images/fitness_function.png" width="900" height="450">
</p>

Individuals are evaluated in the **evaluation function or *fitness***. The output of this function will determine whether the individual is accurate for the given problem or not. In the proposed technique, the **binary strings are converted into quantum circuits** which will act as feature maps into QSVM. Firstly, the classifier is fitted with training set and then we make predictions over test set (data not previously seen by the model) **-seeking generalization power-**, getting the objective of the fitness function. At the same time, we calculate the number of gates penalizing doubly the entangling operators due to a higher computational cost. We calculate a metric -Weight Control- in order to find a **balance between both metrics**, the accuracy and the reduction of number of gates. It is important since a high weight on the reducing circuit size objetive can lead less accuracy because of information loss.

#### 3.2.1 PCA Approach
One of the proposed approaches is PCA. In this case, the first six bits of the individual are taken and used to determine the number of components. This number, after apply the transformations is the number of variables to be embedded in the rotation operators of the circuit. As it has been commented, the maximum individual size is calculated as M × N × 7. In this approach, seven more bits are added to the individual to be used as the number of components in order to maintain the same chain size of the quantum circuit. In this case, the string length is calculated as M × N × 7 + 7. The first six bits are used only to ensure a fair comparison with the other approximation methods, leaving the bit number seven of the individual unused. However, more bits could be used in this dimensionality reduction method encoding. The number of features is limited to 6 bits, resulting in a maximum of 64 dimensions, as $2^6$.

##### Steps
* The circuit's maximum size is defined based on the number of qubits (M) and layers (N). The individual's encoding includes seven bits for the number of components (M×N×7+7). Since we limit components to 64, the first six bits are utilized, maintaining the structure of the quantum circuit M×N×7 for fair comparison.
* The individual enters the fitness function, separating the circuit (M×N×7) and number of components (7 bits). The first six bits are converted to an integer, representing the number of components for the PCA method.
* The dataset is split into training and test sets, standardized, and the PCA transformation is applied with the individual's specified number of components. A death penalty is applied if components are zero or one. The individual is decoded to embed PCA features in quantum rotations, and the QSVM is trained only on the training set.
* Once trained, predictions are made on the test set, and the quantum circuit complexity is calculated. These metrics are objectives of the genetic algorithm. Optimal individuals are stored in the Pareto front, and genetic operators create the next generation.
* Iterate from Step 2 until the algorithm converges or meets the genetic algorithm's stop conditions.

#### 3.2.1 CAE Approach
In the CAE approach, a small convolutional autoencoder neural network is pretrained to perform 64-dimensional extraction, which is equivalent to six bits. This method can be considered to be a form of *transfer learning*. In the CAE approach, we apply the encoding part of the network to the input data, obtaining the latent space in 64 dimensions. In this case, the whole individual corresponds to the quantum circuit, it means, M×N×7.

##### Steps
* A small Convolutional AutoEncoder (CAE) neural network is implemented to extract information. The encoding part of the network, with an output of 64 dimensions, is utilized for application.
* The maximum circuit size is determined based on the number of input qubits (M) and layers (N) with the expression M×N×7.
* The dataset is split into training and test sets, standardized, and the pretrained CAE model is applied to obtain 64 fixed dimensions as input variables for the quantum circuit. This process prevents data leakage effects.
* The individual enters the fitness function, is decoded, and generates a quantum feature map embedding CAE's output dimensions in the quantum gates. The Quantum Support Vector Machine (QSVM) is trained exclusively with the decoded quantum circuit using the training set.
* After training, the model is applied to the test set, making predictions and calculating accuracy. Simultaneously, the complexity of the quantum circuit is assessed. Both metrics are objectives of the genetic algorithm. Individuals improving both objectives or one without worsening the other, compared to those in the Pareto front, are stored in the Pareto front for that generation.
* Genetic operators are applied to create the next generation.
* Iterate the processes starting from Step 4 until the algorithm converges or the defined stop conditions are reached, obtaining the Pareto front.

## 4. Results: Pareto Front and Best Quantum Circuit
Este sistema, permite obtener circuitos-solución que se van posicionando en el frente de Pareto basado en las métricas obtenidas en la función de fitness. Se pretende obtener el mejor clasificador con la menor complejidad posible, siempore que no afecte a la métrica principal: accuracy.

<p align="center">
  <img src="https://github.com/sergio94al/AutoQML-Quantum-Inspired-Kernels-by-Using-Genetic-Algorithms-for-Grayscale-images/blob/main/images/par" width="900" height="450">
</p>

Este circuto, 

## 5. Verified Code - Code Ocean Reproducibility Badge

The code for this work has been verified and published on the Code Ocean platform, where the Reproducibility Badge has been granted, indicating that it is functional. *Sergio Altares-López, Juan José García-Ripoll, Angela Ribeiro (2023) AutoQML: Quantum Inspired Kernels by Using Genetic Algorithms for Grayscale Images. https://doi.org/10.24433/CO.3955535.v1*

<p align="center">
    <img src="https://github.com/sergio94al/AutoQML-Quantum-Inspired-Kernels-by-Using-Genetic-Algorithms-for-Grayscale-images/blob/main/images/badge_CO.png" width="275" height="50">
</p>

```xml
@misc{d8c085c8-ea11-47ff-ab90-9715ffc2b39d,
  title = {AutoQML: Quantum Inspired Kernels by Using Genetic Algorithms for Grayscale Images},
  author = {Sergio Altares-López and Juan José García-Ripoll and Angela Ribeiro},
   journal = {Expert Systems with Applications},
  doi = {10.24433/CO.3955535.v1}, 
  howpublished = {\url{https://www.codeocean.com/}},
  year = 2023,
  month = {11},
  version = {v1}
}
```

## 6. How to Use the Code
### 6.1. Files Description

* circuit.py: We create the quantum operators that will composed the quantum circuit.
* fitness.py: Evaluation fuction of the genetic algorithm (we fit 2 variables to return -the objetives)
* gsvm.py: Genetic algorithm function with the genetic operators. We call the fitness function.
* qsvm.py: We create a simulated quantum support vector machine by using sklearn.
* encoding.py: In this file we create the encoding of the quantum gates and the parameters *θ*.
* encoding2.py: This file is used to visualize and obtain the solution after the evolution.
* Visualizing.ipynb: Execute this file to generate a quantum circuit from the designated folder of pickles. Ensure that you select the one with the highest accuracy and minimal complexity, as indicated in the file name.
* QSVM_Name.ipynb: Execute this file to run the implementation of the genetic algorithm, which generates optimal quantum circuits by calling the corresponding .py files. Users can customize parameters based on their specific requirements.
## 7. References
* [1] https://github.com/sergio94al/Automatic_design_of_quantum_feature_maps_Genetic_Auto-Generation
* [2] Altares-López, S., Ribeiro, A., & García-Ripoll, J. J. (2021). Automatic design of quantum feature maps. Quantum Science and Technology, 6(4), 045015. (https://iopscience.iop.org/article/10.1088/2058-9565/ac1ab1)
* [3] Fortin, F. A., De Rainville, F. M., Gardner, M. A. G., Parizeau, M., & Gagné, C. (2012). DEAP: Evolutionary algorithms made easy. The Journal of Machine Learning Research, 13(1), 2171-2175. (https://deap.readthedocs.io/en/master/index.html)
* [4] Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE transactions on evolutionary computation, 6(2), 182-197.
* [5] Sklearn. Support Vector Machine. https://scikit-learn.org/stable/modules/svm.html
## 8. Public Datasets
* [1] N. Chakrabarty, Brain dataset, https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection/metadata (April 2019).
* [2] P. Raikote, Covid-19 dataset, https://www.kaggle.com/pranavraikokte/covid19-image-dataset/metadata (2020).
