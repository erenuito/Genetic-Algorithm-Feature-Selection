# Genetic Algorithm Optimized Feature Selection for Random Forest Classifier

This project implements a genetic algorithm to optimize feature selection for a Random Forest classifier using a diabetes dataset. The goal is to enhance the accuracy of the classifier by selecting the most relevant features.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Functions Explained](#functions-explained)
  - [calculate_fitness_v2](#calculate_fitness_v2)
  - [parallel_fitness](#parallel_fitness)
  - [genetic_algorithm_optimized](#genetic_algorithm_optimized)
  - [tournament_selection](#tournament_selection)
  - [crossover](#crossover)
  - [mutation](#mutation)
  - [fitness_based_selection](#fitness_based_selection)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Feature selection is a critical step in building a machine learning model, as it helps improve the model's performance by reducing overfitting, improving accuracy, and reducing training time. This project leverages a genetic algorithm to select the optimal subset of features for a Random Forest classifier to predict diabetes outcomes.

## Dataset

The dataset used in this project is the Diabetes dataset, which contains the following columns:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome

The target variable is `Outcome`, indicating whether a patient has diabetes.

## Installation

To run this project, you need to have Python installed along with the following libraries:

- numpy
- pandas
- scikit-learn
- joblib

You can install the required libraries using pip:

```bash
pip install numpy pandas scikit-learn joblib
Usage
Clone this repository:
```
##Usage
1.Clone this repository:

```bash

git clone https://github.com/yourusername/genetic-algorithm-feature-selection.git
cd genetic-algorithm-feature-selection

```
2.Ensure you have the diabetes.csv file in the same directory as the script.


3.Run the script:

```bash

python genetic_algorithm_feature_selection.py

```
#Project Structure

genetic_algorithm_feature_selection.py: Main script containing the implementation of the genetic algorithm and the Random Forest classifier.

#Functions Explained
calculate_fitness_v2
Calculates the fitness of an individual in the population. Fitness is measured as the accuracy of the Random Forest classifier using the selected features.


```python

def calculate_fitness_v2(individual, X_train, X_test, y_train, y_test):
    X_train_selected = X_train.iloc[:, individual == 1]
    X_test_selected = X_test.iloc[:, individual == 1]

    model = RandomForestClassifier(random_state=48)

    model.fit(X_train_selected, y_train)

    y_pred = model.predict(X_test_selected)

    accuracy = accuracy_score(y_test, y_pred)

```
    return accuracy
parallel_fitness
Evaluates the fitness of the entire population in parallel to speed up the process.

python
Kodu kopyala
def parallel_fitness(population, X_train, X_test, y_train, y_test):
    fitness_values = Parallel(n_jobs=-1)(delayed(calculate_fitness_v2)(individual, X_train, X_test, y_train, y_test) for individual in population)
    return np.array(fitness_values)
genetic_algorithm_optimized
Main function to run the genetic algorithm. It iterates through the specified number of runs, performing selection, crossover, and mutation to evolve the population.

python
Kodu kopyala
def genetic_algorithm_optimized(X_train, X_test, y_train, y_test, population_size, crossover_probability, mutation_probability, tournament_size, n_runs):
    population = np.random.randint(2, size=(population_size, X_train.shape[1]))

    for n_run in n_runs:
        for run in range(n_run):
            fitness_values = parallel_fitness(population, X_train, X_test, y_train, y_test)

            parents = tournament_selection(population, fitness_values, tournament_size)

            children = crossover(parents, crossover_probability)

            children = mutation(children, mutation_probability)

            new_population = np.vstack((population, children))

            new_fitness_values = parallel_fitness(new_population, X_train, X_test, y_train, y_test)

            best_individuals = fitness_based_selection(new_population, new_fitness_values, population_size)

            population = new_population[best_individuals]

        best_individual_index = np.argmax(fitness_values)
        best_individual = population[best_individual_index]
        best_accuracy = fitness_values[best_individual_index]

        print(f"Run {n_run}: Best individual: {best_individual}, Accuracy: {best_accuracy}")
tournament_selection
Selects parents for the next generation using tournament selection.

python
Kodu kopyala
def tournament_selection(population, fitness_values, tournament_size):
    parents = []

    for _ in range(len(population)):
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness_values = fitness_values[tournament_indices]
        best_parent_index = tournament_indices[np.argmax(tournament_fitness_values)]
        parents.append(population[best_parent_index])

    return np.array(parents)
crossover
Performs crossover between pairs of parents to produce children.

python
Kodu kopyala
def crossover(parents, crossover_probability):
    children = []

    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        parent2 = parents[i + 1]

        if np.random.rand() < crossover_probability:
            point = np.random.randint(len(parent1))
            child1 = np.hstack((parent1[:point], parent2[point:]))
            child2 = np.hstack((parent2[:point], parent1[point:]))
        else:
            child1, child2 = parent1.copy(), parent2.copy()

        children.append(child1)
        children.append(child2)

    return np.array(children)
mutation
Mutates the children based on the mutation probability.

python
Kodu kopyala
def mutation(children, mutation_probability):
    for i in range(len(children)):
        for j in range(len(children[i])):
            if np.random.rand() < mutation_probability:
                children[i][j] = 1 - children[i][j]

    return children
fitness_based_selection
Selects the best individuals from the combined population of parents and children based on their fitness values.

python
Kodu kopyala
def fitness_based_selection(population, fitness_values, new_population_size):
    best_individuals_index = np.argsort(fitness_values)[-new_population_size:]
    return best_individuals_index
Results
The output of the script provides the best individual (feature subset) and the corresponding accuracy for different numbers of runs:

mathematica
Kodu kopyala
Run 100: Best individual: [1 1 1 0 0 1 1 1], Accuracy: 0.7792207792207793
Run 500: Best individual: [1 1 1 0 1 1 1 1], Accuracy: 0.7792207792207793
Run 1000: Best individual: [1 1 1 0 1 1 1 1], Accuracy: 0.7792207792207793
Run 2000: Best individual: [1 1 1 0 0 1 1 1], Accuracy: 0.7792207792207793
Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.
