#pragma once
#include "MatrixNN.h"

class GANNChromosome
{
public:
	GANNChromosome(MatrixNN::MatrixNNInit neuralNetworkLayout);
	~GANNChromosome();

	std::vector<float> GetWeights() { return m_neuralNetwork->GetWeights(); }
	std::vector<float> GetBiases() { return m_neuralNetwork->GetBiases(); }

	void SetWeights(std::vector<float> weights) { m_neuralNetwork->SetWeights(weights); }
	void SetBiases(std::vector<float> biases) { m_neuralNetwork->SetBiases(biases); }

	Matrix FeedForward(std::vector<float> inputArray);
	void AddFitness(float fitness) { m_fitness += fitness; }

	float GetFitness() const { return m_fitness; }

private:
	MatrixNN* m_neuralNetwork;
	float m_fitness;
};

class GANN
{
public:
	GANN(MatrixNN::MatrixNNInit neuralNetworkLayout);
	GANN(unsigned int populationSize, MatrixNN::MatrixNNInit neuralNetworkLayout);
	~GANN();

	MatrixNN::MatrixNNInit GetNeuralNetworkLayout() const { return m_neuralNetworkLayout; }
	void AddChromosome(GANNChromosome* chromosome) { m_population.push_back(chromosome); }

	std::vector<GANNChromosome*>* GetPopulation() { return &m_population; };
	void CreateNewGeneration();

	GANNChromosome* GetFittest();

private:
	std::vector<GANNChromosome*> CalcNewPopulationPool();
	void FindSuitableMates(std::vector<GANNChromosome*> pool, GANNChromosome** chromosome0, GANNChromosome** chromosome1);
	GANNChromosome* Mate(GANNChromosome* chromosome0, GANNChromosome* chromosome1);

	MatrixNN::MatrixNNInit m_neuralNetworkLayout;
	std::vector<GANNChromosome*> m_population;
	unsigned int m_generation;
};