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
        # Drag to pan; Ctrl+wheel zooms about the cursor. A plain wheel is left
        # to the panel, which is what scrolls when the dock is taller than the
        # screen -- a graph that swallowed it would trap the pointer.
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        #: Which nodes were drawn, by key, for a test or a caller to ask.
        self.nodes: dict[str, _Node] = {}
        #: The edges drawn on the path, as ``(source, target)`` keys.
        self.lit: set[tuple[str, str]] = set()
        self._layout: Layout | None = None
        #: Whether the user has zoomed or panned. Until then every redraw and
        #: resize fits the whole graph; after, their view is kept, because a
        #: refresh that threw it away would undo what they just did.
        self._placed_by_hand = False
        self._pressed_at: Any = None

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

        self._grow_scene()
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
        """Fit the whole graph in view, but never blow a small one up.

        Left alone once the user has zoomed or panned; a double-click on the
        background asks for the fit back.
        """
        if self._placed_by_hand:
            return
        rect = self._scene.itemsBoundingRect().adjusted(-8, -8, 8, 8)
        if rect.isEmpty():
            return
        self.resetTransform()
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        if self.transform().m11() > 1.0:
            self.resetTransform()
            self.centerOn(rect.center())

    def fit(self) -> None:
        """Forget the user's zoom and pan, and show the whole graph again."""
        self._placed_by_hand = False
        self._fit()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._fit()

    # -- zooming and panning ---------------------------------------------

    #: How far the user may zoom, as a scale of the scene.
    _ZOOM_RANGE = (0.15, 4.0)

    def zoom(self, factor: float) -> None:
        """Scale the view by *factor*, within the allowed range."""
        low, high = self._ZOOM_RANGE
        current = self.transform().m11()
        factor = max(low / current, min(high / current, factor))
        self.scale(factor, factor)
        self._placed_by_hand = True

    def wheelEvent(self, event: Any) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # One notch of a mouse wheel is 120 units: about 20% a notch, and
            # smooth on a trackpad, which sends many small ones.
            self.zoom(1.0015 ** event.angleDelta().y())
            event.accept()
            return
        # Not ours: pass it up, so the panel around the graph scrolls.
        event.ignore()

    def _grow_scene(self) -> None:
        """Give the scene room around the graph, so a zoomed view can pan.

        Without slack the scene is exactly the graph, and a view showing all
        of it has nowhere to move to.
        """
        rect = self._scene.itemsBoundingRect()
        margin = max(rect.width(), rect.height())
        self._scene.setSceneRect(rect.adjusted(-margin, -margin, margin, margin))

    # -- picking ---------------------------------------------------------

    def mousePressEvent(self, event: Any) -> None:
        self._pressed_at = event.pos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        super().mouseReleaseEvent(event)
        pressed, self._pressed_at = self._pressed_at, None
        if pressed is None:
            return
        moved = (event.pos() - pressed).manhattanLength()
        if moved > 4:
            # A drag was a pan, not a click: it changes the view, not what
            # is selected.
            self._placed_by_hand = True
            return
        self.picked.emit(self._key_at(event.pos()))

    def mouseDoubleClickEvent(self, event: Any) -> None:
        if not self._key_at(event.pos()):
            self.fit()
            return
        super().mouseDoubleClickEvent(event)

    def _key_at(self, pos: Any) -> str:
        """The node under a point of the view, or '' for the background."""
        item = self.itemAt(pos)
        while item is not None and not isinstance(item, _Node):
            item = item.parentItem()
        return item.key if isinstance(item, _Node) else ""

    def sizeHint(self) -> Any:
        return QSize(360, 260)
