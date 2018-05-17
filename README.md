# InverseIsing
This repo accompanies the paper "Inverse Ising inference by combining Ornstein-Zernike theory with deep learning" by Soma Turi and Alpha Lee

### Prerequisites 
Codes in this repo call functions in Matlab R2015b, Python 3.6 and scikit-learn 0.19.1

### Description 
1. pseudolikelihood.m: A Matlab implementation of the pseudolikelihood algorithm for inverse Ising inference (c.f. E. Aurell and M. Ekeberg, Physical Review Letters, 108, 090201 (2012))
2. generate_Ising_data.m: A Matlab function that performs the Monte Carlo simulations to generate the training data
3. nn_production.ipynb: A Jupyter notebook with the neural network archetectures that parameterise the F and G functions reported in our paper 
4. F.sav and G.sav: Parameters of the neural network closure
5. tox21.py: A Python script to perform the benchmarking study on the Tox21 dataset. The script takes F.sav and G.sav as input
6. tox21_10k_challenge_score.smiles, tox21_10k_challenge_score.txt: the Tox21 dataset taken from the challenge website 
