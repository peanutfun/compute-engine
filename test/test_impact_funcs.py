"""Test impact functions"""

import pytest
import numpy as np
import numpy.testing as npt

from climadace.impact_funcs import (
    ImpactFunction,
    InterpolatedImpactFunction,
    impact_function,
    FunctionMap,
    ImpactFunctionMap
)


def default_func_1(x):
    return x + 1


def default_func_2(x):
    return x + 2

@pytest.mark.skip
def test_impact_function():
    imp_func = ImpactFunction(func=default_func_1)
    assert imp_func.func is default_func_1
    npt.assert_array_equal(imp_func(np.array([0, 1, 2])), np.array([1, 2, 3]))

    # Modify stored function
    imp_func.func = default_func_2
    assert imp_func.func is default_func_2
    npt.assert_array_equal(imp_func(np.array([0, 1, 2])), np.array([2, 3, 4]))


def test_interpolated_impact_function():
    imp_func = InterpolatedImpactFunction(xp=[0, 1, 2], fp=[1, 2, 3])
    npt.assert_array_equal(
        imp_func(np.array([-1, 0, 1, 2, 3])), np.array([1, 1, 2, 3, 3])
    )

    # Modify
    imp_func = InterpolatedImpactFunction.from_func(
        xp=[0, 1, 2], impf=default_func_2
    )
    npt.assert_array_equal(
        imp_func(np.array([-1, 0, 1, 2, 3])), np.array([2, 2, 3, 4, 4])
    )


def test_impact_func_decorator():

    @impact_function
    def my_func(x):
        return default_func_1(x)

    # assert isinstance(my_func, ImpactFunction)
    npt.assert_array_equal(my_func(np.array([0, 1, 2])), np.array([1, 2, 3]))

    # Modify stored function
    # my_func.func = default_func_2
    # assert my_func.func is default_func_2
    # npt.assert_array_equal(my_func(np.array([0, 1, 2])), np.array([2, 3, 4]))

    # Interpolated function
    @impact_function(interp_at=[0, 1])
    def my_func_2(x):
        return default_func_1(x)

    # Modify stored function
    assert isinstance(my_func_2, InterpolatedImpactFunction)
    npt.assert_array_equal(my_func_2(np.array([-1, 0, 1, 2])), np.array([1, 1, 2, 2]))


def test_func_map():
    fm = FunctionMap()
    fm.assert_required_types()

    ifm = ImpactFunctionMap(leaf="foo")
    ifm.assert_required_types()

    with pytest.raises(ValueError) as err:
        ImpactFunctionMap()
    assert "missing required FuncType" in str(err)
