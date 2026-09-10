SDK
===

The user-facing layer. Application code depends on these classes and nothing
else, which is what makes a NetQMPI program backend-agnostic.

Environment
-----------

.. automodule:: netqmpi.sdk.environment
   :members:
   :undoc-members:
   :show-inheritance:

Communicator
------------

.. automodule:: netqmpi.sdk.communicator
   :members:
   :undoc-members:
   :show-inheritance:

Circuit
-------

.. automodule:: netqmpi.sdk.circuit
   :members:
   :undoc-members:
   :show-inheritance:

Resources
---------

.. automodule:: netqmpi.sdk.resources
   :members:
   :undoc-members:
   :show-inheritance:

Operations
----------

The operation model recorded by a circuit and consumed by a backend adapter.
Every class can be imported directly from :mod:`netqmpi.sdk.operations`.

Base operation
^^^^^^^^^^^^^^

.. automodule:: netqmpi.sdk.operations.operation
   :members:
   :undoc-members:
   :show-inheritance:

Container
^^^^^^^^^

.. automodule:: netqmpi.sdk.operations.container
   :members:
   :undoc-members:
   :show-inheritance:

Gates
^^^^^

.. automodule:: netqmpi.sdk.operations.gate
   :members:
   :undoc-members:
   :show-inheritance:

Non-unitary operations
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: netqmpi.sdk.operations.non_unitary
   :members:
   :undoc-members:
   :show-inheritance:

Communication operations
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: netqmpi.sdk.operations.qmpi
   :members:
   :undoc-members:
   :show-inheritance:
