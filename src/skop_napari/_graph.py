"""The graph, drawn: roots and steps as boxes, reads as arrows.

The Record panel's table says what each step is. This says how they fit
together, and answers the question a table cannot: *where did this come from?*
Pointing at a node lights the path through it -- everything it was made from
in blue, everything made from it in purple -- and dims the rest.

Where a node goes is ``skop_visual.layout``'s decision, and which nodes are on
the path is ``skop_visual.lineage``'s; neither knows about Qt. This module
only paints what they say, so it holds no idea of the graph of its own.
"""

from __future__ import annotations

from typing import Any

from qtpy.QtCore import QPointF, QSize, Qt, Signal
from qtpy.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from qtpy.QtWidgets import (
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QSizePolicy,
)

from skop_visual import (
    COLOURS,
    ROOT_COLOURS,
    SIGNALS,
    Layout,
    Lineage,
    Placed,
)

#: The size of a node, and the gap between neighbours.
_WIDTH, _HEIGHT = 128, 46
_GAP_X, _GAP_Y = 20, 34
#: What the two halves of a path are drawn in: made from, and makes.
_UPSTREAM = QColor("#1a73e8")
_DOWNSTREAM = QColor("#8430ce")
_PLAIN = QColor("#80868b")
#: How much of a node outside the path is left.
_DIMMED = 0.28


def _emoji_free(signal: str) -> str:
    """A badge's words without its emoji, which a small node has no room for."""
    return signal.split(" ", 1)[-1]


class _Node(QGraphicsRectItem):
    """One box. It remembers which node of the layout it is."""

    def __init__(self, placed: Placed, selected: bool) -> None:
        super().__init__(0, 0, _WIDTH, _HEIGHT)
        self.key = placed.key
        fill, stroke = ROOT_COLOURS if placed.badge is None else COLOURS[placed.badge]
        self.setBrush(QBrush(QColor(fill)))
        self.setPen(QPen(QColor(stroke), 3 if selected else 1.4))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        title = QGraphicsSimpleTextItem(_clip(placed.label, 17), self)
        bold = QFont(title.font())
        bold.setBold(True)
        title.setFont(bold)
        title.setBrush(QBrush(QColor(stroke)))
        title.setPos(8, 5)
        sub = "input" if placed.badge is None else _emoji_free(SIGNALS[placed.badge])
        detail = QGraphicsSimpleTextItem(sub, self)
        detail.setBrush(QBrush(QColor(stroke)))
        detail.setPos(8, 24)
        self.setToolTip(placed.label)


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


class GraphCanvas(QGraphicsView):
    """Draws a ``Layout``, and reports which node was clicked."""

    #: The key of the node clicked, or '' for the background.
    picked = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        #: Which nodes were drawn, by key, for a test or a caller to ask.
        self.nodes: dict[str, _Node] = {}
        #: The edges drawn on the path, as ``(source, target)`` keys.
        self.lit: set[tuple[str, str]] = set()
        self._layout: Layout | None = None

    # -- drawing ---------------------------------------------------------

    def show_graph(
        self,
        layout: Layout,
        lineage: Lineage | None = None,
        selected: str | None = None,
    ) -> None:
        """Draw the layout, with the path through *selected* lit.

        Rebuilt from scratch each time. A handful of boxes is cheaper to paint
        again than to keep in step, and there is no state in the scene that
        the layout does not already hold.
        """
        self._layout = layout
        self._scene.clear()
        self.nodes = {}
        self.lit = set()

        up = (
            set(lineage.upstream) | {f"root:{r}" for r in lineage.roots}
            if lineage
            else set()
        )
        down = set(lineage.downstream) if lineage else set()
        here = {lineage.step} if lineage else set()

        # Top to bottom, each level centred on the widest: a dock is tall and
        # narrow, and a graph drawn left to right in one is a thin strip.
        in_level: dict[int, int] = {}
        for placed in layout.nodes:
            in_level[placed.col] = in_level.get(placed.col, 0) + 1
        widest = max(in_level.values(), default=1)

        def at(placed: Placed) -> QPointF:
            slack = (widest - in_level[placed.col]) / 2
            return QPointF(
                (placed.row + slack) * (_WIDTH + _GAP_X),
                placed.col * (_HEIGHT + _GAP_Y),
            )

        where = {n.key: at(n) for n in layout.nodes}

        for source, target, param in layout.edges:
            colour, width, lit = _PLAIN, 1.3, False
            if lineage is not None:
                if {source, target} <= up | here:
                    colour, width, lit = _UPSTREAM, 2.8, True
                elif {source, target} <= down | here:
                    colour, width, lit = _DOWNSTREAM, 2.8, True
            if lit:
                self.lit.add((source, target))
            self._scene.addItem(
                self._edge(
                    where[source],
                    where[target],
                    colour,
                    width,
                    param,
                    lineage is None or lit,
                )
            )

        for placed in layout.nodes:
            node = _Node(placed, selected=placed.key == selected)
            node.setPos(where[placed.key])
            if lineage is not None and not lineage.touches(placed.key):
                node.setOpacity(_DIMMED)
            self._scene.addItem(node)
            self.nodes[placed.key] = node

        self._scene.setSceneRect(self._scene.itemsBoundingRect().adjusted(-8, -8, 8, 8))
        self._fit()

    def _edge(
        self,
        start: QPointF,
        end: QPointF,
        colour: QColor,
        width: float,
        param: str,
        bright: bool,
    ) -> QGraphicsPathItem:
        """A curve from the bottom of one box to the top of the next."""
        a = QPointF(start.x() + _WIDTH / 2, start.y() + _HEIGHT)
        b = QPointF(end.x() + _WIDTH / 2, end.y())
        bend = max(14.0, (b.y() - a.y()) / 2)
        path = QPainterPath(a)
        path.cubicTo(QPointF(a.x(), a.y() + bend), QPointF(b.x(), b.y() - bend), b)
        # The head is part of the same item, so it dims and lights with the line.
        head = QPolygonF(
            [b, QPointF(b.x() - 4.5, b.y() - 9), QPointF(b.x() + 4.5, b.y() - 9)]
        )
        path.addPolygon(head)
        item = QGraphicsPathItem(path)
        pen = QPen(colour, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        item.setPen(pen)
        item.setBrush(QBrush(colour))
        item.setZValue(-1)
        item.setOpacity(1.0 if bright else _DIMMED)
        item.setToolTip(param)
        return item

    # -- sizing ----------------------------------------------------------

    def _fit(self) -> None:
        """Fit the whole graph in view, but never blow a small one up."""
        rect = self._scene.sceneRect()
        if rect.isEmpty():
            return
        self.resetTransform()
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        if self.transform().m11() > 1.0:
            self.resetTransform()
            self.centerOn(rect.center())

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._fit()

    # -- picking ---------------------------------------------------------

    def mousePressEvent(self, event: Any) -> None:
        item = self.itemAt(event.pos())
        while item is not None and not isinstance(item, _Node):
            item = item.parentItem()
        self.picked.emit(item.key if isinstance(item, _Node) else "")
        super().mousePressEvent(event)

    def sizeHint(self) -> Any:
        return QSize(360, 260)
