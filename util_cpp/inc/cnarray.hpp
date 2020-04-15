#ifndef KLEIN_CPP_INC_CNARRAY_HPP 
#define KLEIN_CPP_INC_CNARRAY_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "cnarray_c.hpp"


/*
/// @brief construct np.1darray from a raw data pointer
///        user should make sure the data pointer is valid
///        during the lifetime of the constructed np.1darray
template<typename T>
py::array to_ndarray(T * data, int n){
    return np::from_data(data,
							np::dtype::get_builtin<T>(),
							boost::python::make_tuple(n),
							boost::python::make_tuple(sizeof(T)),
							boost::python::object()
						);
}
*/


/// @brief internally used for destroy the pointer malloced in the method cndarray::init
template<typename T>
inline void internal_destroy_cndarray_object(PyObject* obj) {

    T * b = reinterpret_cast<T*>( PyCapsule_GetPointer(obj, NULL) );
    if(b){
        ///boost::alignment::aligned_free(b);
        std::free(b);
    }
}


/**
 * @class template<> cndarray
 * @brief provide a convenient way of transferring between c++ data array and numpy.ndarray
 *
 * @tparam T: data type (e.g. double, int, ...)
 * @tparam N: array dimension (e.g. 1, 2, 3, ...)
 */
template<typename T, int N>
class cndarray : public cndarray_c<T, N>{
public:
    cndarray()
		:cndarray_c<T, N>() {}

    cndarray(T* data, bool is_raw=false)
		:cndarray_c<T, N>(data ,is_raw) {}

    /// @brief from np::ndarray
    cndarray(const py::array & x)
		:cndarray_c<T, N>(){
        from_ndarray(x);
    }

	/// @brief init data with np::ndarray
    void from_ndarray( const py::array_t<T> & x ){
		assert((NULL == this->data_ && "You cannot reinitialize cndarray!"));
		this->data_ = reinterpret_cast<T*>((void*) x.data());
		for(int i = 0; i < N; ++i ){
			this->shape_[i] = x.shape(i);
			this->strides_[i] = x.strides(i) / sizeof(T);
		}
    }

    //// @brief to np::ndarray
    py::array to_ndarray() const {
        ssize_t v[N];
        for(int i = 0; i < N; ++i) v[i] = this->strides_[i] * sizeof(T);
        const auto info = py::buffer_info(
                this->data_,                           /* data as contiguous array  */
                sizeof(T),                            /* size of one scalar        */
                py::format_descriptor<T>::format(),   /* data type                 */
                N,                                    /* number of dimensions      */
                this->shape_,                                   /* shape of the matrix       */
                v                                  /* strides for each axis     */
        );
        /*
        if(this->is_raw_){
            py::handle h(::PyCapsule_New((void *)info.ptr, NULL, (PyCapsule_Destructor)&internal_destroy_cndarray_object<int>));
            return py::array(pybind11::dtype(info), info.shape, info.strides, info.ptr, py::object(h, false));
        }
            */
        const bool is_borrowed = !this->is_raw_;
        py::handle h(::PyCapsule_New((void *)info.ptr, NULL, is_borrowed ? (PyCapsule_Destructor)&internal_destroy_cndarray_object<int> : NULL));
        return py::array(pybind11::dtype(info), info.shape, info.strides, info.ptr, py::object(h, is_borrowed));

    }


	/// @brief return sub-cndarray with axis index is j
	///        return_type is cndarray<T, N-1> if N > 1
	///        return_type is T                if N = 1
    typename if_< is_greater< std::integral_constant<int, N>, std::integral_constant<int, 1> >, cndarray<T, N-1>, T >::type
		view(int j, int axis=0){
            typedef typename is_greater< std::integral_constant<int, N>, std::integral_constant<int, 1> >::type mpl_bool_type;
			return _view(j,  axis,  mpl_bool_type());
	}
    
    /// @brief return sub-cndarray with axis index is j
	///        return_type is cndarray<T, N-1> if N > 1
	///        return_type is T                if N = 1
    typename if_< is_greater< std::integral_constant<int, N>, std::integral_constant<int, 1> >, cndarray<T, N-1>, T >::type const
		view(int j, int axis=0)const{
            typedef typename is_greater< std::integral_constant<int, N>, std::integral_constant<int, 1> >::type mpl_bool_type;
			return _view(j,  axis,  mpl_bool_type());
	}
protected:
	cndarray<T, N-1> _view(int j, int axis, const std::true_type &)const{
		/// always make sure N > 1
		cndarray<T, N-1> sub_cndarr (this->data_ + j * this->strides_[axis]);
		int k = 0;
		for(int i = 0; i < N; ++i){
			if(i == axis) continue;
			sub_cndarr.shape(k) = this->shape_[i];
			sub_cndarr.strides(k) = this->strides_[i];
			++k;
		}
		return sub_cndarr;
	}
	T _view(int i, int axis, const std::false_type &)const{
		/// always make sure N == 1
		return this->ix(i);
	}
};


#endif // GAUSS_PY_INC_CNARRAY_HPP
