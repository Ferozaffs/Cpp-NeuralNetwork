#pragma once
#include <vector>
#include "Matrix.h"

class NNLayer
{
public:
	NNLayer(unsigned int numInputs, unsigned int numOutputs)
	{
		layerWeights = Matrix(numOutputs, numInputs);
		layerBias = Matrix(numOutputs, 1);	

		layerInputs = Matrix(numInputs, 1);
		layerOutputs = Matrix(numOutputs, 1);
	}
	~NNLayer(){}

	void RandomizeNormalized()
	{
		layerWeights.RandomizeNormalized();
		layerBias.RandomizeNormalized();
	}

	Matrix layerWeights;
	Matrix layerBias;

	Matrix layerInputs;
	Matrix layerOutputs;
};

class MatrixNN
{
public:
	enum ResolveFunction
	{
		ResolveFunction_ReLu,
		ResolveFunction_Sigmoid,
	};

	struct MatrixNNInit
	{
		unsigned int numInputs;
		unsigned int numTargets;
		unsigned int numHiddenLayers;
		unsigned int numHiddenLayerNodes;
		ResolveFunction func;
	};

	MatrixNN(MatrixNNInit init);
	MatrixNN(unsigned int numInputs, unsigned int numTargets, ResolveFunction func);
	~MatrixNN();

	void Compile();

	Matrix FeedForward(std::vector<float> inputArray);
	float CalculateCost(Matrix outputs, std::vector<float> targetArray);
	void Train(float learningRate);

	void AppendHiddenLayer(unsigned int numNodes);

	std::vector<float> GetWeights();
	std::vector<float> GetBiases();
	void SetWeights(std::vector<float> weights);
	void SetBiases(std::vector<float> biases);

private:
	static const float Sigmoid(float val);
	static const float DiSigmoid(float val);

	void CalculateErrors(float learningRate);

	bool m_compiled;

	unsigned int m_numInputs;
	unsigned int m_numTargets;
	ResolveFunction m_resolveFunction;

	std::vector<NNLayer> m_layerData;
	std::vector<unsigned int> m_hiddenLayers;

	float m_accumelatedCost;
	Matrix m_accumelatedTargets;
	unsigned int m_numCalculations;
};