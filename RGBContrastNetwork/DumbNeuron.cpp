#include "pch.h"
#include "DumbNeuron.h"

DumbNeuronNode::DumbNeuronNode(unsigned int numInputs)
{
	m_numInputs = numInputs;
	m_weights = new float[m_numInputs];
	m_inputSum = new float[m_numInputs];
	memset(m_inputSum, 0, sizeof(float) * m_numInputs);
}

DumbNeuronNode::~DumbNeuronNode()
{
	delete[] m_weights;
	delete[] m_inputSum;
}

void DumbNeuronNode::SetWeights()
{
	for (auto i = 0U; i < m_numInputs; ++i)
	{
		m_weights[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		m_weights[i] = m_weights[i] * 2.0f - 1.0f;
	}

	m_bias = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	m_bias = m_bias * 2.0f - 1.0f;

	// 	m_weights[0] = 0.2126f;
	// 	m_weights[1] = 0.7152f;
	// 	m_weights[2] = 0.0722f;
	// 
	// 	m_bias = 0.0f;
}

float DumbNeuronNode::Calculate(std::vector<float> inputs)
{
	float result = 0;
	for (auto i = 0U; i < m_numInputs; ++i)
	{
		m_inputSum[i] += inputs[i];
		result += m_weights[i] * inputs[i];
	}

	result += m_bias;
	float sigmoid = (1.0f / (1.0f + exp(-result)));
	//float reLu = std::max(0.0f, result);

	m_outputSum += sigmoid;
	++m_numCalcs;

	return sigmoid;
}

std::vector<float> DumbNeuronNode::Train(float error)
{
	float avgOutput = m_outputSum / m_numCalcs;
	//float derivedReLu = avgOutput > 0 ? 1.0f : 0.0f;
	float derivedSigmoid = avgOutput * (1 - avgOutput);

	m_bias -= derivedSigmoid * error * 0.5f;

	std::vector<float> parentRatios;
	for (auto i = 0U; i < m_numInputs; ++i)
	{
		float derivedInput = m_inputSum[i] / m_numCalcs;
		parentRatios.push_back(m_weights[i] * derivedSigmoid * error);
		m_weights[i] -= derivedInput * derivedSigmoid * error * 0.5f;
	}

	return parentRatios;
}

void DumbNeuronNode::ClearSums()
{
	m_outputSum = 0.0f;
	m_numCalcs = 0;
	for (auto i = 0U; i < m_numInputs; ++i)
	{
		m_inputSum[i] = 0.0f;
	}
}

DumbNeuronCluster::DumbNeuronCluster(unsigned int numInputs, unsigned int numLayers, unsigned int numNodes, unsigned int numOutputs)
	: m_numInputs(numInputs)
{
	std::vector<DumbNeuronNode*> layerNodes;
	unsigned int tempInputs = numInputs;
	for (auto i = 0U; i < numLayers; ++i)
	{
		for (auto j = 0U; j < numNodes; ++j)
		{
			layerNodes.push_back(new DumbNeuronNode(tempInputs));
		}

		m_layers.push_back(layerNodes);
		tempInputs = layerNodes.size();
		layerNodes.clear();
	}

	for (auto j = 0U; j < numOutputs; ++j)
	{
		layerNodes.push_back(new DumbNeuronNode(tempInputs));
	}
	m_layers.push_back(layerNodes);

	SetWeights();
}

DumbNeuronCluster::~DumbNeuronCluster()
{
	for (auto it = m_layers.begin(); it != m_layers.end(); ++it)
	{
		while (it->empty())
		{
			delete it->back();
			it->pop_back();
		}
	}
}

void DumbNeuronCluster::SetWeights()
{
	for (auto it = m_layers.begin(); it != m_layers.end(); ++it)
	{
		for (auto it2 = it->begin(); it2 != it->end(); ++it2)
		{
			(*it2)->SetWeights();
		}
	}
}

std::vector<float> DumbNeuronCluster::Calculate(float* inputs)
{
	std::vector<float> tempInputs;
	tempInputs.resize(m_numInputs);
	std::vector<float> tempOutputs;
	memcpy(&tempInputs[0], inputs, sizeof(float) * m_numInputs);

	for (auto it = m_layers.begin(); it != m_layers.end(); ++it)
	{
		tempOutputs.clear();

		for (auto it2 = it->begin(); it2 != it->end(); ++it2)
		{
			tempOutputs.push_back((*it2)->Calculate(tempInputs));
		}

		tempInputs = tempOutputs;
	}

	return tempOutputs;
}

void DumbNeuronCluster::Train(float* targets)
{
	bool firstLayer = true;
	std::vector<std::vector<float>> ratioVec;
	std::vector<std::vector<float>> prevRatioVec;
	for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it)
	{
		prevRatioVec = ratioVec;
		ratioVec.clear();
		if (firstLayer)
		{
			unsigned int counter = 0;
			for (auto it2 = it->begin(); it2 != it->end(); ++it2)
			{
				float error = 2.0f * ((*it2)->GetAvgOutput() - targets[counter]);
				ratioVec.push_back((*it2)->Train(error));
				++counter;
			}

			firstLayer = false;
		}
		else
		{
			unsigned int nodeCounter = 0;
			for (auto it2 = it->begin(); it2 != it->end(); ++it2)
			{
				float error = 0.0f;
				for (auto it3 = prevRatioVec.begin(); it3 != prevRatioVec.end(); ++it3)
				{
					error += it3->at(nodeCounter);
				}
				++nodeCounter;

				ratioVec.push_back((*it2)->Train(error));
			}
		}
	}

	for (auto it = m_layers.begin(); it != m_layers.end(); ++it)
	{
		for (auto it2 = it->begin(); it2 != it->end(); ++it2)
		{
			(*it2)->ClearSums();
		}
	}

}