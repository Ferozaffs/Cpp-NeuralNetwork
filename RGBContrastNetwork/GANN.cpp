#include "pch.h"
#include "GANN.h"
#include <algorithm>

bool compareChomosomes(GANNChromosome* a, GANNChromosome* b)
{ 
	return (a->GetFitness() > b->GetFitness());
}

GANNChromosome::GANNChromosome(MatrixNN::MatrixNNInit neuralNetworkLayout)
	: m_fitness(0.0f)
{
	m_neuralNetwork = new MatrixNN(neuralNetworkLayout);
}

GANNChromosome::~GANNChromosome()
{
	delete m_neuralNetwork;
}

Matrix GANNChromosome::FeedForward(std::vector<float> inputArray)
{
	return m_neuralNetwork->FeedForward(inputArray);
}

GANN::GANN(MatrixNN::MatrixNNInit neuralNetworkLayout)
	: m_neuralNetworkLayout(neuralNetworkLayout)
	, m_generation(0)
{

}

GANN::GANN(unsigned int populationSize, MatrixNN::MatrixNNInit neuralNetworkLayout)
	: m_neuralNetworkLayout(neuralNetworkLayout)
	, m_generation(0)
{
	for (auto i = 0U; i < populationSize; ++i)
	{
		GANNChromosome* chromosome = new GANNChromosome(m_neuralNetworkLayout);
		m_population.push_back(chromosome);
	}
}

GANN::~GANN()
{
	while (!m_population.empty())
	{
		delete m_population.back();
		m_population.pop_back();
	}
}

void GANN::CreateNewGeneration()
{
	auto newPopulationPool = CalcNewPopulationPool();

	GANNChromosome* chromosome0 = nullptr;
	GANNChromosome* chromosome1 = nullptr;
	std::vector<GANNChromosome*> newPopulation;
	for (auto i = 0U; i < m_population.size(); ++i)
	{
		FindSuitableMates(newPopulationPool, &chromosome0, &chromosome1);
		newPopulation.push_back(Mate(chromosome0, chromosome1));
	}

	while (!m_population.empty())
	{
		delete m_population.back();
		m_population.pop_back();
	}

	m_population.swap(newPopulation);

	++m_generation;
}

GANNChromosome* GANN::GetFittest()
{
	std::sort(m_population.begin(), m_population.end(), compareChomosomes);

	return m_population[0];
}

std::vector<GANNChromosome*> GANN::CalcNewPopulationPool()
{
	const unsigned int eliteThreshold = 10;
	const unsigned int survivalThreshold = 25;

	std::vector<GANNChromosome*> newPopulationPool;

	std::sort(m_population.begin(), m_population.end(), compareChomosomes);

	unsigned int numElites = eliteThreshold * m_population.size() / 100;
	for (auto i = 0U; i < numElites; ++i)
	{
		unsigned int chromsomeIndex = rand() % (eliteThreshold * m_population.size() / 100);
		newPopulationPool.push_back(m_population[chromsomeIndex]);
	}
		
	for (auto i = numElites; i < m_population.size(); ++i)
	{
		unsigned int chromsomeIndex = rand() % (survivalThreshold * m_population.size() / 100);
		newPopulationPool.push_back(m_population[chromsomeIndex]);
	}

	return newPopulationPool;
}

void GANN::FindSuitableMates(std::vector<GANNChromosome*> pool, GANNChromosome** chromosome0, GANNChromosome** chromosome1)
{
	unsigned int chromsomeIndex0 = rand() % pool.size();
	unsigned int chromsomeIndex1 = rand() % pool.size();

	(*chromosome0) = pool[chromsomeIndex0];
	(*chromosome1) = pool[chromsomeIndex1];
}

GANNChromosome* GANN::Mate(GANNChromosome* chromosome0, GANNChromosome* chromosome1)
{
	std::vector<float> chromsomeWeights0 = chromosome0->GetWeights();
	std::vector<float> chromsomeWeights1 = chromosome1->GetWeights();
	std::vector<float> chromsomeBiases0 = chromosome0->GetBiases();
	std::vector<float> chromsomeBiases1 = chromosome1->GetBiases();

	std::vector<float> newWeights;
	std::vector<float> newBiases;

	for (auto i = 0U; i < chromsomeWeights0.size(); ++i)
	{ 
		unsigned int r = rand() % 100;

		if (r < 45)
		{
			newWeights.push_back(chromsomeWeights0[i]);
		}
		else if (r < 90)
		{
			newWeights.push_back(chromsomeWeights1[i]);
		}
		else
		{
			float normalizedRand = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
			normalizedRand = normalizedRand * 2.0f - 1.0f;
			newWeights.push_back(normalizedRand);
		}
	}

	for (auto i = 0U; i < chromsomeBiases0.size(); ++i)
	{
		unsigned int r = rand() % 100;

		if (r < 45)
		{
			newBiases.push_back(chromsomeBiases0[i]);
		}
		else if (r < 90)
		{
			newBiases.push_back(chromsomeBiases1[i]);
		}
		else
		{
			float normalizedRand = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
			normalizedRand = normalizedRand * 2.0f - 1.0f;
			newBiases.push_back(normalizedRand);
		}
	}

	GANNChromosome* newChromosome = new GANNChromosome(m_neuralNetworkLayout);
	newChromosome->SetWeights(newWeights);
	newChromosome->SetBiases(newBiases);

	return newChromosome;
}
