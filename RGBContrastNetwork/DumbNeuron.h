#pragma once
#include <vector>

class DumbNeuronNode
{
public:
	DumbNeuronNode(unsigned int numInputs);
	~DumbNeuronNode();

	void SetWeights();
	float Calculate(std::vector<float> inputs);
	std::vector<float> Train(float error);
	void ClearSums();
	float GetAvgOutput() { return m_outputSum / m_numCalcs; }

	float* m_weights;
	float m_bias;
	unsigned int m_numInputs;

	unsigned int m_numCalcs;
	float m_outputSum;
	float* m_inputSum;
};

class DumbNeuronCluster
{
public:
	DumbNeuronCluster(unsigned int numInputs, unsigned int numLayers, unsigned int numNodes, unsigned int numOutputs);
	~DumbNeuronCluster();

	void SetWeights();
	std::vector<float> Calculate(float* inputs);
	void Train(float* errors);

	std::vector<std::vector<DumbNeuronNode*>> m_layers;
	unsigned int m_numInputs;

};

