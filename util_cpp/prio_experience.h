#ifndef SHARE_PRIO_EXPERIENCE_H
#define SHARE_PRIO_EXPERIENCE_H

#include "sum_tree_base.h"
#include <random>
#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


namespace py = pybind11;


/// https://github.com/takoika/PrioritizedExperienceReplay
class PrioExperienceBase{
public:
	PrioExperienceBase(int capacity, int batch);

	int add(double priority);

	void sample(py::array_t<int> & indices, py::array_t<double> & priorities_a);

	void update_priority(const py::array_t<int> & indices, const py::array_t<double> & priorities);

	inline int capacity() const {return tree_.capacity(); }
	inline int get_size()const{return tree_.get_size();}
	inline int batch_size()const{return batch_;}
protected:
	SumTreeBase tree_;
	const int batch_;

private:
 	std::mt19937 engine_;
	std::uniform_real_distribution<double> dist_;

};

#endif
