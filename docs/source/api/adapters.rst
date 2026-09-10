Backend adapters
================

Four worked implementations of the :class:`~netqmpi.runtime.executor.Executor`
contract. Application code never imports these; the CLI selects one from its
flag.

.. note::

   ``netqasm`` and ``cunqa`` are imported at module level by their adapters and
   are mocked when these docs are built, since neither can be installed in a
   public CI runner. Signatures and docstrings are accurate; the types they
   borrow from those packages are not resolved into links.

.. _cunqa:

CUNQA
-----

The reference backend: HPC emulation through virtual QPUs, and the only adapter
implementing every communication primitive. See :doc:`../backends/cunqa`.

Executor and configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: netqmpi.runtime.adapters.cunqa.cunqa_executor
   :members:
   :undoc-members:
   :show-inheritance:

Circuit adapter
^^^^^^^^^^^^^^^

.. automodule:: netqmpi.runtime.adapters.cunqa.cunqa_circuit
   :members:
   :undoc-members:
   :show-inheritance:

Communicator
^^^^^^^^^^^^

.. automodule:: netqmpi.runtime.adapters.cunqa.cunqa_communicator
   :members:
   :undoc-members:
   :show-inheritance:

NetQASM / SquidASM
------------------

Low-level quantum-network simulation. See :doc:`../backends/netqasm`.

Executor and configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: netqmpi.runtime.adapters.netqasm.netqasm_executor
   :members:
   :undoc-members:
   :show-inheritance:

Circuit adapter
^^^^^^^^^^^^^^^

.. automodule:: netqmpi.runtime.adapters.netqasm.netqasm_circuit
   :members:
   :undoc-members:
   :show-inheritance:

Communicator
^^^^^^^^^^^^

.. automodule:: netqmpi.runtime.adapters.netqasm.netqasm_communicator
   :members:
   :undoc-members:
   :show-inheritance:

Qiskit Aer
----------

Shot-based circuit simulation on one monolithic circuit. See
:doc:`../backends/aer`.

Executor
^^^^^^^^

.. automodule:: netqmpi.runtime.adapters.aer.aer_executor
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
^^^^^^^^^^^^^

.. automodule:: netqmpi.runtime.adapters.aer.aer_run_config
   :members:
   :undoc-members:
   :show-inheritance:

Circuit adapter
^^^^^^^^^^^^^^^

.. automodule:: netqmpi.runtime.adapters.aer.aer_circuit
   :members:
   :undoc-members:
   :show-inheritance:

Communicator
^^^^^^^^^^^^

.. automodule:: netqmpi.runtime.adapters.aer.aer_communicator
   :members:
   :undoc-members:
   :show-inheritance:

Qoala
-----

Quantum-internet node execution environment, simulation only. See
:doc:`../backends/qoala`.

Executor and configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: netqmpi.runtime.adapters.qoala.qoala_executor
   :members:
   :undoc-members:
   :show-inheritance:

Circuit adapter
^^^^^^^^^^^^^^^

.. automodule:: netqmpi.runtime.adapters.qoala.qoala_circuit
   :members:
   :undoc-members:
   :show-inheritance:

Communicator
^^^^^^^^^^^^

.. automodule:: netqmpi.runtime.adapters.qoala.qoala_communicator
   :members:
   :undoc-members:
   :show-inheritance:
