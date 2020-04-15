

#include "prio_experience.h"
#include "inc/cnarray.hpp"


typedef cndarray<int, 1> int_1darray;
typedef cndarray<double, 1> double_1darray;

PrioExperienceBase::PrioExperienceBase(int capacity, int batch)
	:tree_(capacity)
	, batch_(batch)

	, engine_()
	, dist_(0.0, 1.0)
	{}

int PrioExperienceBase::add(double priority){
	return tree_.add(priority);
}


void PrioExperienceBase::sample(py::array_t<int> & indices_a, py::array_t<double> & priorities_a){
	if ( ! (1 == indices_a.ndim() &&  1 == priorities_a.ndim() ) ) {
        throw std::runtime_error("Incorrect number of dimensions: indices, priorities.");
    }
    if( ! (batch_ == indices_a.shape(0) && batch_ == priorities_a.shape(0)) ){
        throw std::runtime_error("Incorrect number of shape: indices, priorities.");
    }
	const double num_els = tree_.get_size();
	if( num_els < batch_){
		throw std::runtime_error("number elements < batch_size!");	
	}

    int_1darray indices(indices_a);
	double_1darray priorities(priorities_a);
	
///	const double capacity = tree_.capacity();

///	double min_p = 10e10;
	for(int i = 0; i < batch_; ++i){
		const double r = dist_(engine_);
		const std::pair<int, double> & res = tree_.find(r);
		//// index, priority = res.first, res.second
		if(res.first >= num_els){
			indices.ix(i) = -1;
			continue;
		}

		/// const double w = res.second; /// > 1e-10 ? std::pow(res.second * capacity, -beta) : 1e-10;		
///		if(res.second < min_p) min_p = res.second;
		
		indices.ix(i) = res.first;
		priorities.ix(i) = res.second;

		tree_.update_value(res.first, 0);	///# To avoid duplicating
	}

///	if(min_p <=0) min_p=1e-10;
	for(int i = 0; i < batch_; ++i){
		if(indices.ix(i) >= 0){
			tree_.update_value(indices.ix(i), priorities.ix(i));
		}
	}
///	return min_p;
}



void PrioExperienceBase::update_priority(const py::array_t<int> & indices_a, const py::array_t<double> & priorities_a){
	if ( ! (1 == indices_a.ndim() &&  1 == priorities_a.ndim() ) ) {
        throw std::runtime_error("Incorrect number of dimensions: indices, priorities");
    }

    if( ! (indices_a.shape(0) == priorities_a.shape(0)) ){
        throw std::runtime_error("Incorrect number of shape: indices, priorities");
    }

    int_1darray indices(indices_a);
    double_1darray priorities(priorities_a);

    for(int i = 0; i < indices.shape(0); ++i){
    	tree_.update_value(indices.ix(i), priorities.ix(i));
    }
}



