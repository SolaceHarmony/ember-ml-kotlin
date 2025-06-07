"""
Statistical operations interface.

This module defines the abstract interface for statistical operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Sequence

class StatsOps(ABC):
    """Abstract interface for statistical operations."""
    
    # === Descriptive Statistics ===
    @abstractmethod
    def mean(self, x: Any, axis: Optional[Union[int, Sequence[int]]] = None,
            keepdims: bool = False) -> Any:
        """
        Compute the mean of a tensor along specified axes.
        
        Args:
            x: Input tensor
            axis: Axis or axes along which to compute the mean
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Mean of the tensor
        """
        pass
    
    @abstractmethod
    def var(self, x: Any, axis: Optional[Union[int, Sequence[int]]] = None,
           keepdims: bool = False, ddof: int = 0) -> Any:
        """
        Compute the variance along the specified axis.
        
        Args:
            x: Input tensor
            axis: Axis or axes along which to compute the variance
            keepdims: Whether to keep the reduced dimensions
            ddof: Delta degrees of freedom
            
        Returns:
            Variance of the tensor
        """
        pass
    
    @abstractmethod
    def median(self, x: Any, axis: Optional[Union[int, Sequence[int]]] = None,
              keepdims: bool = False) -> Any:
        """
        Compute the median along the specified axis.
        
        Args:
            x: Input tensor
            axis: Axis or axes along which to compute the median
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Median of the tensor
        """
        pass
    
    @abstractmethod
    def std(self, x: Any, axis: Optional[Union[int, Sequence[int]]] = None,
           keepdims: bool = False, ddof: int = 0) -> Any:
        """
        Compute the standard deviation along the specified axis.
        
        Args:
            x: Input tensor
            axis: Axis or axes along which to compute the standard deviation
            keepdims: Whether to keep the reduced dimensions
            ddof: Delta degrees of freedom
            
        Returns:
            Standard deviation of the tensor
        """
        pass
    
    @abstractmethod
    def percentile(self, x: Any, q: Union[float, Any],
                  axis: Optional[Union[int, Sequence[int]]] = None,
                  keepdims: bool = False) -> Any:
        """
        Compute the q-th percentile along the specified axis.
        
        Args:
            x: Input tensor
            q: Percentile(s) to compute, in range [0, 100]
            axis: Axis or axes along which to compute the percentile
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            q-th percentile of the tensor
        """
        pass
    
    @abstractmethod
    def max(self, x: Any, axis: Optional[Union[int, Sequence[int]]] = None,
           keepdims: bool = False) -> Any:
        """
        Compute the maximum value along the specified axis.
        
        Args:
            x: Input tensor
            axis: Axis or axes along which to compute the maximum
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Maximum value of the tensor
        """
        pass
    
    @abstractmethod
    def min(self, x: Any, axis: Optional[Union[int, Sequence[int]]] = None,
           keepdims: bool = False) -> Any:
        """
        Compute the minimum value along the specified axis.
        
        Args:
            x: Input tensor
            axis: Axis or axes along which to compute the minimum
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Minimum value of the tensor
        """
        pass
    
    @abstractmethod
    def sum(self, x: Any, axis: Optional[Union[int, Sequence[int]]] = None,
           keepdims: bool = False) -> Any:
        """
        Compute the sum along the specified axis.
        
        Args:
            x: Input tensor
            axis: Axis or axes along which to compute the sum
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Sum of the tensor
        """
        pass
    
    @abstractmethod
    def cumsum(self, x: Any, axis: Optional[int] = None) -> Any:
        """
        Compute the cumulative sum along the specified axis.
        
        Args:
            x: Input tensor
            axis: Axis along which to compute the cumulative sum
            
        Returns:
            Cumulative sum of the tensor
        """
        pass
    
    @abstractmethod
    def argmax(self, x: Any, axis: Optional[int] = None,
              keepdims: bool = False) -> Any:
        """
        Returns the indices of the maximum values along an axis.
        
        Args:
            x: Input tensor
            axis: Axis along which to compute the argmax
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Indices of the maximum values
        """
        pass
    
    @abstractmethod
    def sort(self, x: Any, axis: int = -1, descending: bool = False) -> Any:
        """
        Sort a tensor along the specified axis.
        
        Args:
            x: Input tensor
            axis: Axis along which to sort
            descending: Whether to sort in descending order
            
        Returns:
            Sorted tensor
        """
        pass
    
    @abstractmethod
    def argsort(self, x: Any, axis: int = -1, descending: bool = False) -> Any:
        """
        Returns the indices that would sort a tensor along the specified axis.
        
        Args:
            x: Input tensor
            axis: Axis along which to sort
            descending: Whether to sort in descending order
            
        Returns:
            Indices that would sort the tensor
        """
        pass

    # === Probability Distributions ===
    @abstractmethod
    def gaussian(self, input_value: Any, mu: Any = 0.0, sigma: Any = 1.0) -> Any:
        """
        Compute the value of the Gaussian (normal distribution) function.

        Formula: (1 / (sigma * sqrt(2 * pi))) * exp(-0.5 * ((x - mu) / sigma)^2)

        Args:
            input_value: The input value(s).
            mu: The mean (center) of the distribution. Defaults to 0.0.
            sigma: The standard deviation (spread) of the distribution. Defaults to 1.0.

        Returns:
            The Gaussian function evaluated at the input value(s).
        """
        pass
