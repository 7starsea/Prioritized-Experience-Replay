#ifndef KLEIN_CPP_INC_MPL_HPP
#define KLEIN_CPP_INC_MPL_HPP

#include <type_traits>

template<typename T1, typename T2>
struct equal_to{
    typedef std::integral_constant<bool, (T1::value == T2::value)> type;
    using value_type = bool;
    static constexpr bool value = type::value;
    constexpr operator value_type() const noexcept { return value; }
};

template<typename T1, typename T2>
struct not_equal_to{
    typedef std::integral_constant<bool, (T1::value != T2::value)> type;
    using value_type = bool;
    static constexpr bool value = type::value;
    constexpr operator value_type() const noexcept { return value; }

};

template<typename T1, typename T2>
struct is_greater{
    typedef std::integral_constant<bool, (T1::value > T2::value)> type;
    using value_type = bool;
    static constexpr bool value = type::value;
    constexpr operator value_type() const noexcept { return value; }

};

template<typename T1, typename T2>
struct is_less{
    typedef std::integral_constant<bool, (T1::value < T2::value)> type;
    using value_type = bool;
    static constexpr bool value = type::value;
    constexpr operator value_type() const noexcept { return value; }

};

template<typename T1, typename T2>
struct is_greater_equal{
    typedef std::integral_constant<bool, (T1::value >= T2::value)> type;
    using value_type = bool;
    static constexpr bool value = type::value;
    constexpr operator value_type() const noexcept { return value; }

};

template<typename T1, typename T2>
struct is_less_equal{
    typedef std::integral_constant<bool, (T1::value <= T2::value)> type;
    using value_type = bool;
    static constexpr bool value = type::value;
    constexpr operator value_type() const noexcept { return value; }

};

template<typename C, typename T1, typename T2>
struct if_{
    typedef typename std::conditional<C::value, T1, T2>::type type;
};


#endif
