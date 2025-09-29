"""Impact Functions"""

from abc import ABC, ABCMeta, abstractmethod
from collections.abc import MutableMapping
from enum import StrEnum, auto
from functools import partial
from typing import Any, Callable, Final

import numpy as np
import numpy.typing as npt


# # TODO: Should it store a name? -> NO: Names only for registered impact functions!
class ImpactFunctionBase(ABC):
    @abstractmethod
    def __call__(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """Return the value of the impact function for a given hazard intensity"""
        ...


# class ImpactFunction(ImpactFunctionBase):

#     def __init__(self, func: Callable[[npt.ArrayLike], npt.ArrayLike]):
#         super().__init__()
#         self.func = func

#     def __call__(self, x: npt.ArrayLike) -> npt.ArrayLike:
#         return self.func(x)

ImpactFunction = Callable[[npt.ArrayLike], npt.ArrayLike]


class InterpolatedImpactFunction(ImpactFunctionBase):
    def __init__(self, xp: npt.ArrayLike, fp: npt.ArrayLike):
        super().__init__()
        self.xp = xp
        self.fp = fp

    def __call__(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return np.interp(x, self.xp, self.fp)

    @classmethod
    def from_func(cls, xp: npt.ArrayLike, impf: ImpactFunction):
        """Create from an impact function and intensity values to interpolate at"""
        xp = np.asanyarray(xp)
        return cls(xp=xp, fp=impf(xp))


# NOTE: Must inherit from ABCM
class ABCSingleton(ABCMeta):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(ABCSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ImpactFunctionRegistry(MutableMapping, metaclass=ABCSingleton):
    def __init__(self):
        self.map = {}
        self.allow_overwrite = False

    def reset(self):
        """Reset to the default impact functions"""
        self._set_default_functions()

    def _set_default_functions(self):
        """Set the default impact functions"""
        self.map = {}

    def _raise_not_registered(self, key, err):
        raise KeyError(f"No function registered for: {key}") from err

    def __getitem__(self, key):
        try:
            return self.map[key]
        except KeyError as err:
            self._raise_not_registered(key=key, err=err)

    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self.map and not self.allow_overwrite:
            raise RuntimeError(
                "Impact function '{}' is already registered and 'allow_overwrite' is "
                "set to False"
            )
        self.map[key] = value

    def __delitem__(self, key: Any) -> None:
        try:
            del self.map[key]
        except KeyError as err:
            self._raise_not_registered(key=key, err=err)

    def __iter__(self):
        return iter(self.map)

    def __len__(self) -> int:
        return len(self.map)


REGISTRY = ImpactFunctionRegistry()


# class impact_function:
#     def __init__(self, *, interpolate=None):
#         self.interpolate = interpolate

#     def __call__(self, func):
#         if self.interpolate is not None:
#             return InterpolatedImpactFunction(func=func, xp=self.interpolate)
#         return ImpactFunction(func=func)

# Define decorator
def impact_function(
    func=None, *, interp_at: npt.ArrayLike | None = None, name: str | None = None
):
    if func is None:
        return partial(impact_function, interp_at=interp_at, name=name)

    if interp_at is not None:
        # xp = np.asanyarray(interp_at)
        imp_func = InterpolatedImpactFunction.from_func(xp=interp_at, impf=func)
    else:
        imp_func = func

    if name is not None:
        REGISTRY[name] = imp_func

    return imp_func


class FuncType(StrEnum):
    default = auto()
    leaf = auto()


FuncDefault: Final = FuncType.default
FuncLeaf: Final = FuncType.leaf


class FunctionMap(dict, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            self.assert_required_types()
        except AssertionError as err:
            raise ValueError("Function map is missing required FuncType") from err

    def assert_required_types(self):
        pass


class ImpactFunctionMap(FunctionMap):
    def assert_required_types(self):
        assert FuncType.leaf in self


# TODO: Maybe have a base FunctionMap that does not require a default? For aggregates
# @dataclass
# class ImpactFunctionMap(UserDict):

#     KeyType = str | FuncType

#     def __init__(self, data: dict[KeyType, ImpactFunction]):
#         super().__init__(self._prune_data(data))
#         print(self._prune_data(data))
#         self._assert_default()

#     @staticmethod
#     def _prune_data(data: dict):
#         def prune(key_from, key_to):
#             if key_from in data:
#                 if key_to in data:
#                     raise RuntimeError(
#                         "Giving both {} and {} in function map is not allowed"
#                     )
#                 data[key_to] = data.pop(key_from)

#         prune("_default", FuncDefault)
#         prune("_leaf", FuncLeaf)
#         return data

#     def _assert_default(self):
#         if FuncDefault not in self.data:
#             raise RuntimeError("Default impact function must be specified!")

# def __getitem__(self, key: KeyType) -> ImpactFunctionBase:
#     return super().__getitem__(self.key_to_str(key))

# def __setitem__(self, key: KeyType, item: KeyType) -> None:
#     return super().__setitem__(self.key_to_str(key), item)

# def __delitem__(self, key: KeyType) -> None:
#     return super().__delitem__(self.key_to_str(key))

# def __contains__(self, key: KeyType) -> bool:
#     return super().__contains__(self.key_to_str(key))

# @staticmethod
# def key_to_str(key: KeyType) -> str | FuncType:
#     if not isinstance(key, FuncType):
#         return str(key)
#     return key

# NOTE: 'str' value for name of impact function in registry (TODO!)
# impf_map: Dict[str | PurePath | FuncType, ImpactFunctionBase]

# def get(self, path: str | PurePath) -> ImpactFunctionBase:
#     return self.impf_map.get(path,)
