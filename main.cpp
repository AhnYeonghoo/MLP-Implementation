#include "Neuron.hpp"

int main()
{
	Neuron and_neuron(2);

	for (int i = 0; i < 100000; ++i)
	{
		and_neuron.train(0.1,
			{
				{ { 0, 0 }, 0 },
				{ { 1, 0 }, 0 },
				{ { 0, 1 }, 0 },
				{ { 1, 1 }, 1 },
			});
	}

	std::cout << "0 and 0 = " << and_neuron.compute({ 0, 0 }) << '\n';
	std::cout << "1 and 0 = " << and_neuron.compute({ 1, 0 }) << '\n';
	std::cout << "0 and 1 = " << and_neuron.compute({ 0, 1 }) << '\n';
	std::cout << "1 and 1 = " << and_neuron.compute({ 1, 1 }) << '\n';

	Neuron or_neuron(2);

	for (int i = 0; i < 100000; ++i)
	{
		or_neuron.train(0.1,
			{
				{ { 0, 0 }, 0 },
				{ { 1, 0 }, 1 },
				{ { 0, 1 }, 1 },
				{ { 1, 1 }, 1 },
			});
	}

	std::cout << "0 or 0 = " << or_neuron.compute({ 0, 0 }) << '\n';
	std::cout << "1 or 0 = " << or_neuron.compute({ 1, 0 }) << '\n';
	std::cout << "0 or 1 = " << or_neuron.compute({ 0, 1 }) << '\n';
	std::cout << "1 or 1 = " << or_neuron.compute({ 1, 1 }) << '\n';

	return 0;
}