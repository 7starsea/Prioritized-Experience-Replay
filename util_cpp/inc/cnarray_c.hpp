#ifndef KLEIN_CPP_C_INC_CNARRAY_HPP
#define KLEIN_CPP_C_INC_CNARRAY_HPP

#include <cstdlib>  ///malloc
#include <cstring>  ///memset
#include <assert.h> 

#include <type_traits>
#include "mpl.hpp"

#include <iostream>


/**
 * @class template<> cndarray_c
 * @brief provide a convenient way of transferring between c++ data array and numpy.ndarray
 *
 * @tparam T: data type (e.g. double, int, ...)
 * @tparam N: array dimension (e.g. 1, 2, 3, ...)
 */
template<typename T, int N>
class cndarray_c{

protected:
	/// @brief raw data pointer
    T * data_;
	/// @brief indicating whether the cndarray_c owns the data pointer.
    bool is_raw_;
	/// @brief equivalent shape to np.ndarray
    int shape_[N];
	/// @brief equivalent strides to np.ndarray
    int strides_[N];   /// for columns-continuous memory layout: strides[0] == shape[1] when N=2
public:
    inline const int shape(int i) const{
        return shape_[i];
    }
    inline int & shape(int i){
        return shape_[i];
    }
    inline const int strides(int i) const{
        return strides_[i];
    }
    inline int & strides(int i){
        return strides_[i];
    }
	T * data(){
		return data_;
	}
	const T * data()const{
		return data_;
	}
public:
    cndarray_c()
    :data_(NULL), is_raw_(false) {
        _reset_shape_strides();
    };

    cndarray_c(T* data, bool is_raw=false)
    :data_(data), is_raw_(is_raw) {
        _reset_shape_strides();
    };

	/// @brief return sub-cndarray_c with axis index is j
	///        return_type is cndarray_c<T, N-1> if N > 1
	///        return_type is T                if N = 1
	typename if_< is_greater< std::integral_constant<int, N>, std::integral_constant<int, 1> >, cndarray_c<T, N-1>, T >::type
	view(int j, int axis=0){
		typedef typename is_greater< std::integral_constant<int, N>, std::integral_constant<int, 1> >::type mpl_bool_type;
		return _view(j,  axis,  mpl_bool_type());
	}
    
	/// @brief return sub-cndarray_c with axis index is j
	///        return_type is cndarray_c<T, N-1> if N > 1
	///        return_type is T                if N = 1
	typename if_< is_greater< std::integral_constant<int, N>, std::integral_constant<int, 1> >, cndarray_c<T, N-1>, T >::type const
	view(int j, int axis=0) const {
		typedef typename is_greater< std::integral_constant<int, N>, std::integral_constant<int, 1> >::type mpl_bool_type;
		return _view(j,  axis,  mpl_bool_type());
	}
	/// @brief compare the shape with other cndarray_c
    bool shape_equal(const cndarray_c<T, N> & other) const {
        bool flag=true;
        for(int i = 0; i < N && flag; ++i)
            flag &= shape_[i] == other.shape_[i];
        return flag;
    }

	/// @brief return the internal layout index for index [i, j=0, k=0]
    inline int id(int i, int j = 0, int k=0) const {
        return _id(i, j, k, std::integral_constant<int, N>() );
    }

	/// @brief return the value (const reference) at index [i, j=0, k=0]
    const T& ix(int i, int j = 0, int k=0) const {
        return * (data_ + this->id(i, j, k));
    }

	/// @brief return the value (reference) at index [i, j=0, k=0]
    T& ix(int i, int j = 0, int k = 0) {
        return * (data_ + this->id(i, j, k));
    }

	/// @brief number of elements
	int size(){
		int my_size = 1;
		for(int i = N-1; i >=0; --i){
			my_size *= shape_[i];
		}
		return my_size;
	}

	/// @brief initialize the c1darray, c2darray, and c3darray
	void init(int i, int j=0, int k=0){
		shape_[0] = i;
		if(N >= 2){ shape_[1] = j; }
		if(N >= 3){ shape_[2] = k; }
		init();
	}

	/// @brief reinterpret_view
	template<typename VT>
	cndarray_c<VT, N-1> reinterpret_view(int axis){
		assert(sizeof(VT) == shape_[axis] && "You can not use reinterpret_view size shape[axis]!=sizeof(ViewDataStruct)!");
		cndarray_c<VT, N-1> sub_cndarr ( reinterpret_cast<VT*>(data_) );
		int k = 0;
		for(int i = 0; i < N; ++i){
			if(i == axis) continue;
			sub_cndarr.shape(k) = shape_[i];
			if(i < axis)
				sub_cndarr.strides(k) = strides_[i] / sizeof(VT);
			else /// i > axis
				sub_cndarr.strides(k) = strides_[i];
			++k;
		}
		return sub_cndarr;
	}

    /// @brief initialize the data array by assuming shape have been assigned
    void init(){
		assert((NULL == data_ && "You cannot reinitialize cndarray_c!"));

        int my_size = 1;
        for(int i = N-1; i >=0; --i){
            strides_[i] = my_size;
            my_size *= shape_[i];
        }

        my_size *= sizeof(T);
        
        ///data_  = (T*) std::aligned_alloc(std::alignment_of<T>::value, my_size);
        data_ = (T*) std::malloc(my_size);

        is_raw_ = true;

        std::memset(data_, 0, my_size);
    }

	/// @brief clear the data
	void clear(){
		if(data_ && is_raw_){
   //         std::aligned_free(data_);
			std::free(data_);
			is_raw_ = false;
			_reset_shape_strides();
		}
	}

	/// @brief fill the array with zero
    void fill_zero(){
        if(data_) std::memset(data_, 0, size() * sizeof(T) );
    }

protected:
    int _id(int i, int, int, const std::integral_constant<int, 1> &) const {
        return i * strides_[0];
    }
    int _id(int i, int j, int, const std::integral_constant<int, 2> &) const {
        return i * strides_[0] + j * strides_[1];
    }
    int _id(int i, int j, int k, const std::integral_constant<int, 3> &) const {
        return i * strides_[0] + j * strides_[1] + k * strides_[2];
    }

	void _reset_shape_strides(){
		for(int i = 0; i < N; ++i){
			shape_[i] = 0; strides_[i] = 0;
		}
	}

	cndarray_c<T, N-1> _view(int j, int axis, const std::true_type &) const {
		/// always make sure N > 1
		cndarray_c<T, N-1> sub_cndarr (data_ + j * strides_[axis]);
		int k = 0;
		for(int i = 0; i < N; ++i){
			if(i == axis) continue;
			sub_cndarr.shape(k) = shape_[i];
			sub_cndarr.strides(k) = strides_[i];
			++k;
		}
		return sub_cndarr;
	}
	T _view(int i, int axis, const std::false_type &) const {
		/// always make sure N == 1
		return ix(i);
	}
};


#endif // GAUSS_C_INC_CNARRAY_HPP
