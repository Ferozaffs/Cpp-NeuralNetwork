#include "pch.h"
#include <iostream>
#include <time.h>
#include <vector>
#include <algorithm>
#include "DumbNeuron.h"
#include "MatrixNN.h"
#include <windows.h>
#include "GANN.h"


float FillColor(float* color)
{
	for (auto i = 0U; i < 3; ++i)
	{
		color[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}

	return 0.2126f * color[0] + 0.7152f * color[1] + 0.0722f * color[2];
}

std::vector<float> FillColor()
{
	std::vector<float> colorVec;
	for (auto i = 0U; i < 3; ++i)
	{
		colorVec.push_back(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
	}

	return colorVec;
}

int main()
{
	srand(time(NULL));
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);
	LARGE_INTEGER time;
	QueryPerformanceCounter(&time);

	/*
	MatrixNN::MatrixNNInit init;
	init.numInputs = 3;
	init.numTargets = 3;
	init.numHiddenLayers = 2;
	init.numHiddenLayerNodes = 8;
	init.func = MatrixNN::ResolveFunction_Sigmoid;

	GANN* gann = new GANN(1000, init);

	std::vector<float> targets;
	targets.resize(3);
	float lastAverageCost = 1.0f;
	const unsigned int numGenerations = 100;
	for (auto generation = 0U; generation < numGenerations; ++generation)
	{
		auto population = gann->GetPopulation();
		float cost = 0.0f;
		for (auto i = 0U; i < 100; ++i)
		{
			auto color = FillColor();
	
			for (auto it = population->begin(); it != population->end(); ++it)
			{
				auto outputs = (*it)->FeedForward(color);

				if (color[0] > color[1] && color[0] > color[2])
				{
					if (outputs.m_data[0] > outputs.m_data[1] && outputs.m_data[0] > outputs.m_data[2])
					{
						(*it)->AddFitness(1.0f);
					}
				}
				else if (color[1] > color[0] && color[1] > color[2])
				{
					if (outputs.m_data[1] > outputs.m_data[0] && outputs.m_data[1] > outputs.m_data[2])
					{
						(*it)->AddFitness(1.0f);
					}
				}
				else if(color[2] > color[0] && color[2] > color[1])
				{
					if (outputs.m_data[2] > outputs.m_data[0] && outputs.m_data[2] > outputs.m_data[1])
					{
						(*it)->AddFitness(1.0f);
					}
				}
			}
		}

		if (generation < numGenerations - 1)
		{
			gann->CreateNewGeneration();
		}	
	}

	LARGE_INTEGER time2;
	QueryPerformanceCounter(&time2);

	auto tot = (time2.QuadPart - time.QuadPart) / frequency.QuadPart;
	std::cout << "Time: " << tot << "\n\n";

	auto fittest = gann->GetFittest();
	std::cout << "Fitness: " << fittest->GetFitness() <<"\n\n";

	while (1)
	{

		std::cout << "Red: ";
		std::cin >> targets[0];
		std::cout << "\nGreen: ";
		std::cin >> targets[1];
		std::cout << "\nBlue: ";
		std::cin >> targets[2];

		auto outputs = fittest->FeedForward(targets);

		std::cout << "\n\n Chance color is Red: " << (unsigned int)(outputs.m_data[0] * 100.0f) << "\n";
		std::cout << "Chance color is Green: " << (unsigned int)(outputs.m_data[1] * 100.0f) << "\n";
		std::cout << "Chance color is Blue: " << (unsigned int)(outputs.m_data[2] * 100.0f) << "\n\n";
	}*/

	

	//Matrix Neural network
	MatrixNN neuralNetwork(3, 3, MatrixNN::ResolveFunction_Sigmoid);
	neuralNetwork.AppendHiddenLayer(8);
	neuralNetwork.AppendHiddenLayer(8);
	neuralNetwork.Compile();
	
	std::vector<float> targets;
	targets.resize(3);	
	float lastAverageCost = 1.0f;
	for (auto pass = 0U; pass < 500000; ++pass)
	{
		float cost = 0.0f;
		for (auto i = 0U; i < 100; ++i)
		{
			auto color = FillColor();
			auto outputs = neuralNetwork.FeedForward(color);

			
			if (color[0] > color[1] && color[0] > color[2])
			{
				targets[0] = 1.0f;
				targets[1] = 0.0f;
				targets[2] = 0.0f;
			}
			else if (color[1] > color[0] && color[1] > color[2])
			{
				targets[0] = 0.0f;
				targets[1] = 1.0f;
				targets[2] = 0.0f;
			}
			else
			{
				targets[0] = 0.0f;
				targets[1] = 0.0f;
				targets[2] = 1.0f;
			}

			cost += neuralNetwork.CalculateCost(outputs, targets);
		}

		if (pass % 100 == 0)
		{
			cost /= 100.0f;
			//std::cout << "AverageCost: " << cost << std::endl;
			lastAverageCost = cost;
		}

		static const float learningRateMax = 0.1f;
		static const float learningRateMin = 0.01f;
		neuralNetwork.Train(learningRateMin + lastAverageCost * (learningRateMax - learningRateMin));
	}

	LARGE_INTEGER time2;
	QueryPerformanceCounter(&time2);

	auto tot = (time2.QuadPart - time.QuadPart) / frequency.QuadPart;
	std::cout << tot;

	while (1)
	{
		
		std::cout << "\n\nRed: ";
		std::cin >> targets[0];
		std::cout << "\nGreen: ";
		std::cin >> targets[1];
		std::cout << "\nBlue: ";
		std::cin >> targets[2];

		auto outputs = neuralNetwork.FeedForward(targets);

		std::cout << "\n\n Chance color is Red: " << (unsigned int)(outputs.m_data[0] * 100.0f) << "\n";
		std::cout << "Chance color is Green: " << (unsigned int)(outputs.m_data[1] * 100.0f) << "\n";
		std::cout << "Chance color is Blue: " << (unsigned int)(outputs.m_data[2] * 100.0f) << "\n";
	}

	//Dumb neuron
	/*
	DumbNeuronCluster cluster(3,2,8,3);

	float correctEngagement[3];
	float rgb[3];
	unsigned int numCorrectGuesses = 0;
	for (auto pass = 0U; pass < 500000; ++pass)
	{
		float cost = 0.0f;
		correctEngagement[0] = 0.0f;
		correctEngagement[1] = 0.0f;
		correctEngagement[2] = 0.0f;
		for (auto i = 0U; i < 100; ++i)
		{
			FillColor(rgb);
			auto nnRGB = cluster.Calculate(rgb);

			if (rgb[0] > rgb[1] && rgb[0] > rgb[2])
			{
				if (pass % 100 == 0)
				{
					cost += pow(nnRGB[0] - 1.0f, 2.0f);
					cost += pow(nnRGB[1] - 0.0f, 2.0f);
					cost += pow(nnRGB[2] - 0.0f, 2.0f);
				}

				correctEngagement[0] += 1.0f;
			}
			else if (rgb[1] > rgb[0] && rgb[1] > rgb[2])
			{
				if (pass % 100 == 0)
				{
					cost += pow(nnRGB[0] - 0.0f, 2.0f);
					cost += pow(nnRGB[1] - 1.0f, 2.0f);
					cost += pow(nnRGB[2] - 0.0f, 2.0f);
				}

				correctEngagement[1] += 1.0f;
			}
			else
			{
				if (pass % 100 == 0)
				{
					cost += pow(nnRGB[0] - 0.0f, 2.0f);
					cost += pow(nnRGB[1] - 0.0f, 2.0f);
					cost += pow(nnRGB[2] - 1.0f, 2.0f);
				}

				correctEngagement[2] += 1.0f;
			}
		}

		
		correctEngagement[0] /= 100.0f;
		correctEngagement[1] /= 100.0f;	
		correctEngagement[2] /= 100.0f;
		cluster.Train(correctEngagement);

		if (pass % 100 == 0)
		{
			cost /= 100.0f;
			std::cout << "AverageCost: " << cost << std::endl;
		}
	}

	while (1)
	{
		std::cout << "\n\nRed: ";
		std::cin >> rgb[0];
		std::cout << "\nGreen: ";
		std::cin >> rgb[1];
		std::cout << "\nBlue: ";
		std::cin >> rgb[2];

		auto nnRGB = cluster.Calculate(rgb);

		std::cout << "\n\n Chance color is Red: " << (unsigned int)(nnRGB[0] * 100.0f) << "\n";
		std::cout << "Chance color is Green: " << (unsigned int)(nnRGB[1] * 100.0f) << "\n";
		std::cout << "Chance color is Blue: " << (unsigned int)(nnRGB[2] * 100.0f) << "\n";
	}*/
}




