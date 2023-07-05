#include "Neuron.hpp"

const int MINIMUM = -0.5;
const int MAXIMUM = 0.5;

// activation function
int stepFunction(int s)
{
	return s >= 0 ? 1 : 0;
}

double sigmoidFunction(double x)
{
	return 1 / (1 + std::exp(-x));
}

double sigmoidDerivative(double x)
{
	return sigmoidFunction(x) * (1 - sigmoidFunction(x));
}

double relu(double x)
{
	return x >= 0 ? x : 0;
}

double Neuron::getBias() const
{
	return this->bias;
}

std::vector<double> Neuron::getWeight() const
{
	return this->weight;
}

void Neuron::setBias(double& bias)
{
	this->bias = bias;
}

void Neuron::setWeight(std::vector<double>& weight)
{
	this->weight = weight;
}

Neuron::Neuron(size_t input_size)
{
	this->weight.resize(input_size);
	initialized();
}

double Neuron::compute(const std::vector<double>& x) const
{
	if (x.size() != weight.size())
	{
		std::cerr << "입력층의 개수가 부정확합니다" << std::endl;
		return -1;
	}
	
	// 초기값
	double wx = 0.0;
	
	for (std::size_t i = 0; i < weight.size(); i++)
	{
		wx += weight.at(i) * x[i];
	}

	lastV = wx + this->bias;
	lastX = x;
	return sigmoidFunction(lastV);
	
}

void Neuron::train(double a, const std::vector<std::pair<std::vector<double>, double>>
	& train_data)
{
	std::size_t input_size = train_data[0].first.size();

	if (input_size != this->weight.size())
	{
		std::cerr << "입력층 사이즈가 정확하지 않습니다." << std::endl;
		return;
	}


	for (std::size_t i = 0; i < train_data.size(); i++)
	{
		double o = compute(train_data[i].first);
		double t = train_data[i].second;

		for (std::size_t j = 0; j < input_size; j++)
		{
			this->weight[j] += a * sigmoidDerivative(lastV) * (t - o) * train_data[i].first[j];
		}
		this->bias += a * sigmoidDerivative(lastV) * (t - o);
		
		this->lastD = sigmoidDerivative(lastV) * (t - o);
	
	}

}

void Neuron::train(double a, double e, const std::vector<double>& train_data)
{
	std::size_t input_size = train_data.size();
	
	if (input_size != this->weight.size())
	{
		std::cerr << "Input size != weight size";
	}
	
	for (std::size_t j = 0; j < input_size; j++)
	{
		this->weight[j] += a * sigmoidDerivative(lastV) * e * train_data[j];
	}
	
	this->bias += a * sigmoidDerivative(lastV) * e;
	this->lastD = sigmoidDerivative(lastV) * e;
	
}

void Neuron::initialized()
{
	for (std::size_t i = 0; i < weight.size(); i++)
	{
		this->weight.at(i) = rand() % (MAXIMUM - MINIMUM + 1) + MINIMUM;
	}

	this->bias = rand() % (MAXIMUM - MINIMUM + 1) + MINIMUM;
}



std::size_t Neuron::inputSize() const
{
	return this->weight.size();
}


double Neuron::getLastV() const
{
	return this->lastV;
}

double Neuron::getLastD() const
{
	return this->lastD;
}

const std::vector<double>& Neuron::getLastX() const
{
	return this->lastX;
}

