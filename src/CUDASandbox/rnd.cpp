#include "rnd.h"
#include <random>

thread_local std::mt19937 gen{ std::random_device{}() };

template<typename T>
T rnd::random(T min, T max)
{
	using dist = std::conditional_t<
		std::is_integral<T>::value,
		std::uniform_int_distribution<T>,
		std::uniform_real_distribution<T>
	>;
	
	return dist{ min, max }(gen);
}