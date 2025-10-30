"""Class for open graph states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from graphix.flow._find_cflow import find_cflow
from graphix.flow._find_gpflow import AlgebraicOpenGraph, PlanarAlgebraicOpenGraph, compute_correction_matrix
from graphix.flow.core import CausalFlow, GFlow, PauliFlow
from graphix.fundamentals import AbstractMeasurement, AbstractPlanarMeasurement
from graphix.measurements import Measurement

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    import networkx as nx

    from graphix.pattern import Pattern

# TODO: Maybe move these definitions to graphix.fundamentals and graphix.measurements ? Now they are redefined in graphix.flow._find_gpflow, not very elegant.
_M_co = TypeVar("_M_co", bound=AbstractMeasurement, covariant=True)
_PM_co = TypeVar("_PM_co", bound=AbstractPlanarMeasurement, covariant=True)


@dataclass(frozen=True)
class OpenGraph(Generic[_M_co]):
    """An unmutable dataclass providing a representation of open graph states.

    Attributes
    ----------
    graph : networkx.Graph[int]
        The underlying resource-state graph. Nodes represent qubits and edges represent the application of :math:`CZ` gate on the linked nodes.
    input_nodes : Sequence[int]
        An ordered sequence of node labels corresponding to the open graph inputs.
    output_nodes : Sequence[int]
        An ordered sequence of node labels corresponding to the open graph outputs.
    measurements : Mapping[int, _M_co]
        A mapping between the non-output nodes of the open graph (`key`) and their corresponding measurement label (`value`). Measurement labels can be specified as `Measurement` or `Plane|Axis` instances.

    Notes
    -----
    The inputs and outputs of `OpenGraph` instances in Graphix are defined as ordered sequences of node labels. This contrasts the usual definition of open graphs in the literature, where inputs and outputs are unordered sets of nodes labels. This restriction facilitates the interplay with `Pattern` objects, where the order of input and output nodes represents a choice of Hilbert space basis.

    Example
    -------
    >>> import networkx as nx
    >>> from graphix.fundamentals import Plane
    >>> from graphix.opengraph import OpenGraph, Measurement
    >>>
    >>> graph = nx.Graph([(0, 1), (1, 2)])
    >>> measurements = {i: Measurement(0.5 * i, Plane.XY) for i in range(2)}
    >>> input_nodes = [0]
    >>> output_nodes = [2]
    >>> og = OpenGraph(graph, input_nodes, output_nodes, measurements)
    """

    graph: nx.Graph[int]
    input_nodes: Sequence[int]
    output_nodes: Sequence[int]
    measurements: Mapping[int, _M_co]

    def __post_init__(self) -> None:
        """Validate the correctness of the open graph."""
        all_nodes = set(self.graph.nodes)
        inputs = set(self.input_nodes)
        outputs = set(self.output_nodes)

        if not set(self.measurements).issubset(all_nodes):
            raise ValueError("All measured nodes must be part of the graph's nodes.")
        if not inputs.issubset(all_nodes):
            raise ValueError("All input nodes must be part of the graph's nodes.")
        if not outputs.issubset(all_nodes):
            raise ValueError("All output nodes must be part of the graph's nodes.")
        if outputs & self.measurements.keys():
            raise ValueError("Output nodes cannot be measured.")
        if all_nodes - outputs != self.measurements.keys():
            raise ValueError("All non-ouptut nodes must be measured.")
        if len(inputs) != len(self.input_nodes):
            raise ValueError("Input nodes contain duplicates.")
        if len(outputs) != len(self.output_nodes):
            raise ValueError("Output nodes contain duplicates.")

    @staticmethod
    def from_pattern(pattern: Pattern) -> OpenGraph[Measurement]:
        """Initialise an `OpenGraph[Measurement]` object from the underlying resource-state graph of the input measurement pattern.

        Parameters
        ----------
        pattern : Pattern
            The input pattern.

        Returns
        -------
        OpenGraph[Measurement]
        """
        graph = pattern.extract_graph()

        input_nodes = pattern.input_nodes
        output_nodes = pattern.output_nodes

        meas_planes = pattern.get_meas_plane()
        meas_angles = pattern.get_angles()
        measurements: Mapping[int, Measurement] = {
            node: Measurement(meas_angles[node], meas_planes[node]) for node in meas_angles
        }

        return OpenGraph(graph, input_nodes, output_nodes, measurements)

    def to_pattern(self: OpenGraph[Measurement]) -> Pattern | None:
        """Extract a deterministic pattern from an `OpenGraph[Measurement]` if it exists.

        Returns
        -------
        Pattern | None
            A deterministic pattern on the open graph. If it does not exist, it returns `None`.

        Notes
        -----
        - The open graph instance must be of parametric type `Measurement` to allow for a pattern extraction, otherwise it does not contain information about the measurement angles.

        - This method proceeds by searching a flow on the open graph and converting it into a pattern as prescripted in Ref. [1].
        It first attempts to find a causal flow because the corresponding flow-finding algorithm has lower complexity. If it fails, it attemps to find a Pauli flow because this property is more general than a generalised flow, and the corresponding flow-finding algorithms have the same complexity in the current implementation.

        References
        ----------
        [1] Browne et al., NJP 9, 250 (2007)
        """
        cflow = self.find_causal_flow()
        if cflow is not None:
            return cflow.to_corrections().to_pattern()

        pflow = self.find_pauli_flow()
        if pflow is not None:
            return pflow.to_corrections().to_pattern()

        return None

    def neighbors(self, nodes: Collection[int]) -> set[int]:
        """Return the set containing the neighborhood of a set of nodes in the open graph.

        Parameters
        ----------
        nodes : Collection[int]
            Set of nodes whose neighborhood is to be found

        Returns
        -------
        neighbors_set : set[int]
            Neighborhood of set `nodes`.
        """
        neighbors_set: set[int] = set()
        for node in nodes:
            neighbors_set |= set(self.graph.neighbors(node))
        return neighbors_set

    def odd_neighbors(self, nodes: Collection[int]) -> set[int]:
        """Return the set containing the odd neighborhood of a set of nodes in the open graph.

        Parameters
        ----------
        nodes : Collection[int]
            Set of nodes whose odd neighborhood is to be found

        Returns
        -------
        odd_neighbors_set : set[int]
            Odd neighborhood of set `nodes`.
        """
        odd_neighbors_set: set[int] = set()
        for node in nodes:
            odd_neighbors_set ^= self.neighbors([node])
        return odd_neighbors_set

    def find_causal_flow(self: OpenGraph[_PM_co]) -> CausalFlow[_PM_co] | None:
        """Return a causal flow on the open graph if it exists.

        Returns
        -------
        CausalFlow | None
            A causal flow object if the open graph has causal flow, `None` otherwise.

        Notes
        -----
        - The open graph instance must be of parametric type `Measurement` or `Plane` since the causal flow is only defined on open graphs with :math:`XY` measurements.
        - This function implements the algorithm presented in Ref. [1] with polynomial complexity on the number of nodes, :math:`O(N^2)`.

        References
        ----------
        [1] Mhalla and Perdrix, (2008), Finding Optimal Flows Efficiently, doi.org/10.1007/978-3-540-70575-8_70
        """
        return find_cflow(self)

    def find_gflow(self: OpenGraph[_PM_co]) -> GFlow[_PM_co] | None:
        r"""Return a maximally delayed generalised flow (gflow) on the open graph if it exists.

        Returns
        -------
        GFlow | None
            A gflow object if the open graph has gflow, `None` otherwise.

        Notes
        -----
        - The open graph instance must be of parametric type `Measurement` or `Plane` since the gflow is only defined on open graphs with planar measurements. Measurement instances with a Pauli angle (integer multiple of :math:`\pi/2`) are interpreted as `Plane` instances, in contrast with :func:`OpenGraph.find_pauli_flow`.
        - This function implements the algorithm presented in Ref. [1] with polynomial complexity on the number of nodes, :math:`O(N^3)`.

        References
        ----------
        [1] Mitosek and Backens, 2024 (arXiv:2410.23439).
        """
        aog = PlanarAlgebraicOpenGraph(self)
        correction_matrix = compute_correction_matrix(aog)
        if correction_matrix is None:
            return None
        return GFlow.from_correction_matrix(
            correction_matrix
        )  # The constructor can return `None` if the correction matrix is not compatible with any partial order on the open graph.

    def find_pauli_flow(self: OpenGraph[_M_co]) -> PauliFlow[_M_co] | None:
        r"""Return a maximally delayed generalised flow (gflow) on the open graph if it exists.

        Returns
        -------
        PauliFlow | None
            A Pauli flow object if the open graph has Pauli flow, `None` otherwise.

        Notes
        -----
        - Measurement instances with a Pauli angle (integer multiple of :math:`\pi/2`) are interpreted as `Axis` instances, in contrast with :func:`OpenGraph.find_gflow`.
        - This function implements the algorithm presented in Ref. [1] with polynomial complexity on the number of nodes, :math:`O(N^3)`.

        References
        ----------
        [1] Mitosek and Backens, 2024 (arXiv:2410.23439).
        """
        aog = AlgebraicOpenGraph(self)
        correction_matrix = compute_correction_matrix(aog)
        if correction_matrix is None:
            return None
        return PauliFlow.from_correction_matrix(
            correction_matrix
        )  # The constructor can return `None` if the correction matrix is not compatible with any partial order on the open graph.
