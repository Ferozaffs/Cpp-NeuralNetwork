#include "pch.h"
#include "MatrixNN.h"

const float MatrixNN::Sigmoid(float val)
{
	return (1.0f / (1.0f + exp(-val)));
}

const float MatrixNN::DiSigmoid(float val)
{
	return val * (1 - val);
}

MatrixNN::MatrixNN(MatrixNNInit init)
	: m_numInputs(init.numInputs)
	, m_numTargets(init.numTargets)
	, m_resolveFunction(init.func)
	, m_compiled(false)
{
	m_accumelatedTargets = Matrix(init.numTargets, 1);
	for (auto i = 0U; i < init.numHiddenLayers; ++i)
	{
		AppendHiddenLayer(init.numHiddenLayerNodes);
		Compile();
	}
}

MatrixNN::MatrixNN(unsigned int numInputs, unsigned int numTargets, ResolveFunction func)
	: m_numInputs(numInputs)
	, m_numTargets(numTargets)
	, m_resolveFunction(func)
	, m_compiled(false)
{
	m_accumelatedTargets = Matrix(numTargets, 1);
}

MatrixNN::~MatrixNN()
{

}

void MatrixNN::Compile()
{
	unsigned int nInputs = m_numInputs;
	for (auto it = m_hiddenLayers.begin(); it != m_hiddenLayers.end(); ++it)
	{
		NNLayer layer(nInputs, (*it));
		layer.RandomizeNormalized();

		m_layerData.push_back(layer);
		nInputs = (*it);
	}

	NNLayer outputLayer(nInputs, m_numTargets);
	outputLayer.RandomizeNormalized();

	m_layerData.push_back(outputLayer);

	m_compiled = true;
}

Matrix MatrixNN::FeedForward(std::vector<float> inputArray)
{
	Matrix inputs = Matrix::FromVector(inputArray);
	Matrix outputs;
	for (auto it = m_layerData.begin(); it != m_layerData.end(); ++it)
	{
		it->layerInputs.Add(inputs);

		outputs = it->layerWeights.Multiply(inputs);
		outputs.Add(it->layerBias);
		outputs.Map(Sigmoid);

		it->layerOutputs.Add(outputs);
		inputs = outputs;
	}

	return outputs;
}

void MatrixNN::Train(float learningRate)
{
	CalculateErrors(learningRate);
}

void MatrixNN::AppendHiddenLayer(unsigned int numNodes)
{
	m_hiddenLayers.push_back(numNodes);
}

float MatrixNN::CalculateCost(Matrix outputs, std::vector<float> targetArray)
{
	++m_numCalculations;
	
	Matrix targets = Matrix::FromVector(targetArray);
	outputs.Subtract(targets);
	float cost = 0;
	for (auto i = 0U; i < outputs.m_numRows; ++i)
	{
		cost += pow(outputs.m_data[i], 2.0f);
	}
	
	m_accumelatedCost += cost;
	m_accumelatedTargets.Add(targets);

	return cost;
}

void MatrixNN::CalculateErrors(float learningRate)
{
	float averageCost = m_accumelatedCost / m_numCalculations;
	Matrix averageTargets = m_accumelatedTargets.GetScaled(1.0f / m_numCalculations);
	Matrix averageOutputs = m_layerData.back().layerOutputs.GetScaled(1.0f / m_numCalculations);

	Matrix errors = averageOutputs.GetSubtracted(averageTargets);
	errors.Scale(2.0f);

	for (auto it = m_layerData.rbegin(); it != m_layerData.rend(); ++it)
	{
		averageOutputs = it->layerOutputs.GetScaled(1.0f / m_numCalculations);
		averageOutputs.Map(DiSigmoid);
		Matrix gradients = averageOutputs.GetScaled(errors);
		
		Matrix averageInput = it->layerInputs.GetScaled(1.0f / m_numCalculations);
		Matrix transposedInputs = averageInput.Transpose();
		Matrix deltaWeights = gradients.Multiply(transposedInputs);

		Matrix transposedWeights = it->layerWeights.Transpose();
		errors = transposedWeights.Multiply(gradients);

		gradients.Scale(learningRate);
		it->layerWeights.Subtract(deltaWeights);
		it->layerBias.Subtract(gradients);

		it->layerInputs.Zero();
		it->layerOutputs.Zero();
	}

	m_accumelatedTargets.Zero();
	m_accumelatedCost = 0.0f;
	m_numCalculations = 0;
}

std::vector<float> MatrixNN::GetWeights()
{
	std::vector<float> weights;
	for (auto it = m_layerData.begin(); it != m_layerData.end(); ++it)
	{
		for (auto it2 = it->layerWeights.m_data.begin(); it2 != it->layerWeights.m_data.end(); ++it2)
		{
			weights.push_back((*it2));
		}
	}

	return weights;
}

std::vector<float> MatrixNN::GetBiases()
{
	std::vector<float> biases;
	for (auto it = m_layerData.begin(); it != m_layerData.end(); ++it)
	{
		for (auto it2 = it->layerBias.m_data.begin(); it2 != it->layerBias.m_data.end(); ++it2)
		{
			biases.push_back((*it2));
		}
	}

	return biases;
}

void MatrixNN::SetWeights(std::vector<float> weights)
{
	unsigned int counter = 0;
	for (auto it = m_layerData.begin(); it != m_layerData.end(); ++it)
	{
		unsigned int dataSize = it->layerWeights.m_data.size();
		for (auto i = 0U; i < dataSize; ++i)
		{
			it->layerWeights.m_data[i] = weights[counter++];
		}
	}
}

void MatrixNN::SetBiases(std::vector<float> biases)
{
	unsigned int counter = 0;
	for (auto it = m_layerData.begin(); it != m_layerData.end(); ++it)
	{
		unsigned int dataSize = it->layerBias.m_data.size();
		for (auto i = 0U; i < dataSize; ++i)
		{
			it->layerBias.m_data[i] = biases[counter++];
		}
	}
}

