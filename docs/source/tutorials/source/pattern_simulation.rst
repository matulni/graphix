.. _simulation-tutorial:

Pattern simulation
==================

In the previous tutorials, we saw that an MBQC pattern defines a map from an input qubit register to an output qubit register. Graphix provides a modular framework to simulate this map numerically. The entry point is a single call to :meth:`~.Simulable.simulate`. Behind this simple interface sit several options that control how the simulation runs. This tutorial walks through them one by one.

Simulation backend
------------------

To run a pattern, the commands need a concrete meaning on an actual quantum state. The **backend** implements the :ref:`command semantics <mbqc-commands-table>` on a concrete quantum state representation.

You select the backend with the ``backend`` parameter of :meth:`~.Simulable.simulate`. Graphix ships with two built-in options:

.. list-table::
   :header-rows: 1
   :widths: 22 20 25 25 25

   * - Backend
     - Value
     - Class
     - Use case
     - Memory scaling (:math:`n` qubits)
   * - Statevector (default)
     - ``"statevector"``
     - :class:`~.StatevectorBackend`
     - Ideal, noiseless simulation
     - :math:`2^n`
   * - Density matrix
     - ``"densitymatrix"``
     - :class:`~.DensityMatrixBackend`
     - :ref:`Noisy simulation <noisy-simulation>`
     - :math:`2^n \times 2^n`

.. note::

    A third backend (``"tensornetwork"``) exists but is currently unmaintained and pending a major revision. 

The two built-in backends cover most needs, but they are not the only option. Graphix is designed to be extensible: you can implement a custom backend by subclassing the abstract class :class:`~.Backend` and passing it directly as the ``backend`` argument. This lets you plug in alternative state representations or simulation techniques without modifying the rest of your workflow. For a complete, practical illustration, see the :ref:`Stim backend <stim-backend>` that supports efficient simulation of Clifford patterns.

Input state
-----------

Graphix lets you choose the input state of the computation represented by an MBQC pattern. It is set with the ``input_state`` parameter of :meth:`~.Simulable.simulate`, which accepts several formats.

Tensor product of planar states
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Planar states are represented by the class :class:`~.PlanarState`, which bundles a plane (:math:`\XYplane`, :math:`\XZplane` or :math:`\YZplane`) and an angle. The most common ones are grouped in the namespace :class:`~.BasicStates`:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - State
     - Definition
   * - :attr:`.BasicStates.ZERO`
     - :math:`\ket{0}`
   * - :attr:`.BasicStates.ONE`
     - :math:`\ket{1}`
   * - :attr:`.BasicStates.PLUS`
     - :math:`\ketplus`
   * - :attr:`.BasicStates.MINUS`
     - :math:`\ketminus`
   * - :attr:`.BasicStates.PLUS_I`
     - :math:`\frac{1}{\sqrt{2}}\left(\ket{0} + i\ket{1}\right)`
   * - :attr:`.BasicStates.MINUS_I`
     - :math:`\frac{1}{\sqrt{2}}\left(\ket{0} - i\ket{1}\right)`

If a single state is passed, the backend takes its tensor product with itself to match the number of input qubits. Alternatively, you can pass a sequence of :class:`~.PlanarState` objects, one per input qubit, to prepare a product state with different factors.

.. attention::

   By default, ``input_state=BasicStates.PLUS``, i.e. :math:`\bigotimes_{i \in I} \ketplus_i`, where :math:`I` is the set of input nodes. This differs from circuit-based SDKs, where the default input is typically the all-zero state. The :math:`\ketplus` default ensures that, when no input state is provided, the result is equivalent to running the computation on a graph state without input nodes.

Simulate a small pattern with a single ``BasicStates`` value, then with a sequence of different :class:`~.PlanarState` objects (including one built from an explicit plane and angle).

.. jupyter-execute::

   # Write example

Array of amplitudes
~~~~~~~~~~~~~~~~~~~

You can pass a :class:`numpy.ndarray` containing the :math:`2^n` complex amplitudes of the state vector, or a :math:`2^n \times 2^n` matrix for the ``"densitymatrix"`` backend. The state must be *normalized*.

.. jupyter-execute::

   # Write example

Statevector or density matrix objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, you can pass a :class:`~.Statevector` or a :class:`~.DensityMatrix`, which are also the output types of :meth:`~.Simulable.simulate`. This makes it easy to chain simulations by feeding the output of one pattern into the next.


.. jupyter-execute::

   # Write example


.. _noisy-simulation:

Noise model
-----------

Branch selector
---------------



