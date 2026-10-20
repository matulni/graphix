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

The two built-in backends cover most needs, but they are not the only option. Graphix is designed to be extensible: you can implement a custom backend by subclassing the abstract class :class:`~.Backend` and pass it directly as the ``backend`` argument. This lets you plug in alternative state representations or simulation techniques without modifying the rest of your workflow. For a practical illustration, see the :ref:`Stim backend <stim-backend>` which supports efficient simulation of Clifford patterns.

Input state
-----------

When simulating an MBQC pattern, you can choose the input state for the computation. It is set with the ``input_state`` parameter of :meth:`~.Simulable.simulate`, which accepts several formats that we enumerate next.

Tensor product of planar states
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The input state is very often specified as a tensor product of planar states. Planar states are represented by the class :class:`~.PlanarState`, which bundles a plane (:math:`\XYplane`, :math:`\XZplane` or :math:`\YZplane`) and an angle. The most common ones are grouped in the namespace :class:`~.BasicStates`:

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

.. jupyter-execute::

  from graphix import Circuit, BasicStates, PlanarState, Plane
  import numpy as np

  rng = np.random.default_rng(42)

  qc = Circuit(2)
  qc.h(0)
  qc.cnot(0, 1)
  pattern = qc.transpile().pattern

  # Input state 1: |00>
  state_in_1 = BasicStates.ZERO

  # Input state 2: sqrt(2)/2(|01> + e^(i*pi/4)|11>)
  state_in_2 = [PlanarState(Plane.XY, 0.25), BasicStates.ONE]

  for i, state_in in enumerate([state_in_1, state_in_2]):
      state_out = pattern.simulate(input_state=state_in, rng=rng)
      print(f"Output state {i+1}: {state_out}")


Array of amplitudes
~~~~~~~~~~~~~~~~~~~

For arbitrary inputs, including entangled states, you can pass a :class:`numpy.ndarray` of :math:`2^n` complex amplitudes (or a :math:`2^n \times 2^n` matrix for the ``"densitymatrix"`` backend). The state must be *normalized*. Amplitudes are indexed by the computational basis states in binary order, with the first input qubit as the most significant bit: for two qubits, the entries correspond to :math:`\ket{00}`, :math:`\ket{01}`, :math:`\ket{10}` and :math:`\ket{11}`, in that order:

.. jupyter-execute::

  # Input state 3: sqrt(2)/2(|01> - |10>)
  state_in_3 = np.array([0, 1, -1, 0])/np.sqrt(2)
  state_out_3 = pattern.simulate(input_state=state_in_3, rng=rng)
  print(f"Output state 3: {state_out_3}")

Statevector or density matrix objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, you can pass a :class:`~.Statevector` or a :class:`~.DensityMatrix`, which are also the output types of :meth:`~.Simulable.simulate`. This makes it easy to chain simulations by feeding the output of one pattern into the next. As an illustration, the following example performs a Trotterized time evolution of a simple Ising Hamiltonian: each pattern implements one Trotter step, and the output state of a step is the input state of the next.

.. jupyter-execute::

  from graphix import Circuit, BasicStates, Statevector
  import numpy as np

  rng = np.random.default_rng(42)
  # Rotation gates take angles in units of pi, so we normalize the fields by pi.
  n, dt, J, h = 3, 0.2, 1.0/np.pi, 0.8/np.pi

  # One Trotter step of H = -J ΣZ_i Z_{i+1} - h ΣX_i
  qc = Circuit(n)
  for i in range(n - 1):
      qc.rzz(i+1, i,  2 * J * dt)
  for i in range(n):
      qc.rx(i, 2 * h * dt)
  pattern = qc.transpile().pattern

  def magnetization(state):
      """Magnetization on first site"""
      probs = state.to_prob_dict()
      return sum(p if bits[0] == '0' else -p for bits, p in probs.items())

  # Initial state |000>
  state = Statevector(nqubit=n, data=BasicStates.ZERO)
  for t in range(8):
      print(f"t = {t * dt:.1f}  <Z_0> = {magnetization(state):+.3f}")
      state = pattern.simulate(input_state=state, rng=rng)

.. tip::

   The :class:`~.Statevector` returned by :meth:`~.Simulable.simulate` provides several useful methods:

   - :meth:`~.Statevector.fidelity` computes the fidelity with another statevector;
   - :meth:`~.Statevector.isclose` checks whether two states are equal up to a global phase;
   - :meth:`~.Statevector.to_dict` converts the state to a dictionary of amplitudes;
   - :meth:`~.Statevector.to_prob_dict` converts it to a dictionary of measurement probabilities.

.. _noisy-simulation:

Noise model
-----------

Branch selector
---------------



