#pragma once
#include <vector>
#include <assert.h>
#include <functional>

class Matrix
{
public:
	Matrix()
	{
		m_numRows = 0;
		m_numColumns = 0;
	}
	Matrix(unsigned int rows, unsigned int cols)
	{
		m_numRows = rows;
		m_numColumns = cols;

		m_data.resize(rows * cols);
	}
	~Matrix() 
	{
	}

	void Zero()
	{
		for (auto i = 0U; i < m_numRows; ++i)
		{
			for (auto j = 0U; j < m_numColumns; ++j)
			{
				m_data[i * m_numColumns + j] = 0.0f;
			}
		}
	}

	void SetRow(unsigned int index, std::vector<float> data)
	{
		for (auto i = 0U; i < m_numColumns; ++i)
		{
			m_data[index * m_numColumns + i] = data[i];
		}
	}
	void SetValue(unsigned int row, unsigned int column, float data)
	{
		m_data[row * m_numColumns + column] = data;
	}
	void RandomizeNormalized()
	{
		for (auto i = 0U; i < m_numRows; ++i)
		{
			for (auto j = 0U; j < m_numColumns; ++j)
			{
				m_data[i * m_numColumns + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
				m_data[i * m_numColumns + j] = m_data[i * m_numColumns + j] * 2.0f - 1.0f;
			}
		}
	}

	void Scale(float scale)
	{
		for (auto i = 0U; i < m_numRows; ++i)
		{
			for (auto j = 0U; j < m_numColumns; ++j)
			{
				m_data[i * m_numColumns + j] *= scale;
			}
		}
	}
	void Scale(Matrix m)
	{
		assert(m_numRows == m.m_numRows && m_numColumns == m.m_numColumns && "Matrix doesn't share layout");

		for (auto i = 0U; i < m_numRows; ++i)
		{
			for (auto j = 0U; j < m_numColumns; ++j)
			{
				m_data[i * m_numColumns + j] *= m.m_data[i * m.m_numColumns + j];
			}
		}
	}
	Matrix GetScaled(float scale)
	{
		Matrix result = Matrix(m_numRows, m_numColumns);
		for (auto i = 0U; i < m_numRows; ++i)
		{
			for (auto j = 0U; j < m_numColumns; ++j)
			{
				result.m_data[i * result.m_numColumns + j] = m_data[i * m_numColumns + j] * scale;
			}
		}

		return result;
	}
	Matrix GetScaled(Matrix m)
	{
		Matrix result = Matrix(m_numRows, m_numColumns);
		for (auto i = 0U; i < m_numRows; ++i)
		{
			for (auto j = 0U; j < m_numColumns; ++j)
			{
				result.m_data[i * result.m_numColumns + j] = m_data[i * m_numColumns + j] * m.m_data[i * m.m_numColumns + j];
			}
		}

		return result;
	}

	void Add(Matrix m)
	{
		assert(m_numRows == m.m_numRows && m_numColumns == m.m_numColumns && "Matrix doesn't share layout");

		for (auto i = 0U; i < m_numRows; ++i)
		{
			for (auto j = 0U; j < m_numColumns; ++j)
			{
				m_data[i * m_numColumns + j] += m.m_data[i * m.m_numColumns + j];
			}
		}
	}
	void Subtract(Matrix m)
	{
		assert(m_numRows == m.m_numRows && m_numColumns == m.m_numColumns && "Matrix doesn't share layout");

		for (auto i = 0U; i < m_numRows; ++i)
		{
			for (auto j = 0U; j < m_numColumns; ++j)
			{
				m_data[i * m_numColumns + j] -= m.m_data[i * m.m_numColumns + j];
			}
		}
	}
	Matrix GetSubtracted(Matrix m)
	{
		assert(m_numRows == m.m_numRows && m_numColumns == m.m_numColumns && "Matrix doesn't share layout");
		Matrix result = Matrix(m_numRows, m_numColumns);
		for (auto i = 0U; i < m_numRows; ++i)
		{
			for (auto j = 0U; j < m_numColumns; ++j)
			{
				result.m_data[i * m_numColumns + j] = m_data[i * m_numColumns + j] - m.m_data[i * m.m_numColumns + j];
			}
		}

		return result;
	}
	Matrix Multiply(Matrix m)
	{
		assert(m_numColumns == m.m_numRows && "Matrix row/column mismatch");

		Matrix result = Matrix(m_numRows, m.m_numColumns);
		for (auto i = 0U; i < result.m_numRows; ++i)
		{
			for (auto j = 0U; j < result.m_numColumns; ++j)
			{
				float sum = 0.0f;
				for (auto k = 0U; k < m_numColumns; ++k)
				{
					sum += m_data[i * m_numColumns + k] * m.m_data[k * m.m_numColumns + j];
				}

				result.m_data[i * result.m_numColumns + j] = sum;
			}
		}

		return result;
	}
	Matrix Transpose()
	{
		Matrix result = Matrix(m_numColumns, m_numRows);
		for (auto i = 0U; i < result.m_numRows; ++i)
		{
			for (auto j = 0U; j < result.m_numColumns; ++j)
			{
				result.m_data[i * result.m_numColumns + j] = m_data[j * m_numColumns + i];
			}
		}

		return result;
	}

	void Map(std::function<float(float)> callback)
	{
		for (auto i = 0U; i < m_numRows; ++i)
		{
			for (auto j = 0U; j < m_numColumns; ++j)
			{
				m_data[i * m_numColumns + j] = callback(m_data[i * m_numColumns + j]);
			}
		}
	}
	Matrix GetMapped(std::function<float(float)> callback)
	{
		Matrix result = Matrix(m_numColumns, m_numRows);
		for (auto i = 0U; i < m_numRows; ++i)
		{
			for (auto j = 0U; j < m_numColumns; ++j)
			{
				result.m_data[i * result.m_numColumns + j] = callback(m_data[i * m_numColumns + j]);
			}
		}

		return result;
	}

	static Matrix FromVector(std::vector<float> vec)
	{
		Matrix result(vec.size(), 1);
		for (auto i = 0U; i < vec.size(); ++i)
		{
			result.m_data[i] = vec[i];
		}

		return result;
	}

	unsigned int m_numRows;
	unsigned int m_numColumns;
	std::vector<float> m_data;
};
