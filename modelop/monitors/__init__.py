"""
Statistical and volumetric monitors.

Note:
    These monitors return summaries in dictionary form so that they can be directly
    yielded from a ModelOp runtime and parsed by a Model Life Cycle (MLC).

Example:
    >>> # modelop.metrics
    >>> def metrics(data: pandas.DataFrame):
    ...     from modelop.monitors.volumetrics import VolumetricMonitor
    ...     result = VolumetricMonitor(data).count()
    ...     yield result
    >>> import pandas
    >>> df = pandas.DataFrame({"values": [1, 2, 3]})
    >>> result = next(metrics(df))
    >>> print(result)
    {"record_count": 3}

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    bias
    drift
    performance
    stability
    volumetrics
"""
