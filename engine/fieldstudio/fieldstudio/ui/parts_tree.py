"""The interior parts tree (fs-shell, 2026-11-11; design 4.17 "Interior: solid by default, described inside": "the Regions
panel lists inside parts as a tree; hide or isolate a part to reach the ones beneath"; agreed with fs-contact 11-11 ~23:5x).

Read only on fs-contact's parts: every object carrying a part record o.part {source, pid, name, pairs[, parent]} is listed
under its source, nested by the record's 'parent' pid when there is one, else under part 0 (her body) / the source;
each part's regions are its children. A click selects the part's object; the eye column hides / shows it and 'Isolate
part' isolates it, VIEW state only (scene.view_hidden: a hidden part is still written back and still collides with its
touch / slide partners: the contact rule); the pairs show on hover (touch / slide / bond / pass). The tree never renames,
reorders or renumbers parts and never changes a pair."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QHBoxLayout, QHeaderView, QLabel, QPushButton, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)


def read_parts(scene):
    """on the engine thread: [{oid, name, source, pid, parent, pairs, hidden, regions {id: name}}] of the part objects"""
    out = []
    hid = getattr(scene, 'view_hidden', set())
    for o in scene.objects:
        p = getattr(o, 'part', None)
        if not p:
            continue
        out.append({'oid': o.id, 'name': p.get('name') or o.name, 'source': str(p.get('source', '')),
                    'pid': p.get('pid'), 'parent': p.get('parent'), 'pairs': dict(p.get('pairs') or {}),
                    'hidden': o.id in hid,
                    'regions': {int(k): (v.get('name') or 'Region %s' % k) for k, v in (getattr(o, 'regions', None) or {}).items()}})
    return out


def nest(parts):
    """-> {source: [(part, [children ...]) ...]}: each part under its parent pid when given (and present), else under
    part 0 of its source (part 0 itself at the top)"""
    by_src = {}
    for p in parts:
        by_src.setdefault(p['source'], []).append(p)
    out = {}
    for src, ps in by_src.items():
        pid_of = {p['pid']: p for p in ps}
        kids = {id(p): [] for p in ps}
        roots = []
        for p in sorted(ps, key=lambda q: (q['pid'] is None, q['pid'] if q['pid'] is not None else 0)):
            par = p.get('parent')
            if par is not None and par in pid_of and pid_of[par] is not p:
                kids[id(pid_of[par])].append(p)
            elif p['pid'] not in (0, None) and 0 in pid_of:
                kids[id(pid_of[0])].append(p)
            else:
                roots.append(p)

        def build(p):
            return (p, [build(c) for c in kids[id(p)]])
        out[src] = [build(r) for r in roots]
    return out


class PartsTree(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.eng = win.engine
        L = QVBoxLayout(self)
        L.setContentsMargins(0, 0, 0, 0)
        L.addWidget(QLabel('<b>Parts</b> (inside parts; the eye hides one to reach those beneath)'))
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Part', 'Shown'])
        self.tree.setMinimumHeight(120)
        hv = self.tree.header()
        hv.setStretchLastSection(False)
        hv.setSectionResizeMode(0, QHeaderView.Stretch)            # the names in full (the part, the source)
        hv.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.itemClicked.connect(self._clicked)
        self.tree.itemChanged.connect(self._toggled)
        L.addWidget(self.tree)
        h = QHBoxLayout()
        b = QPushButton('Isolate part')
        b.setToolTip('Only the chosen part shown (again: all shown); view only')
        b.clicked.connect(self._isolate)
        h.addWidget(b)
        b2 = QPushButton('Show all')
        b2.clicked.connect(lambda: self.eng.post('view', 'show_all', []))
        h.addWidget(b2)
        L.addLayout(h)
        self._filling = False
        self.parts = []
        self._tok = None
        self.eng.callDone.connect(self._called)

    def refresh(self):
        tok = object()
        self._tok = tok
        self.eng.post('call', lambda e: read_parts(e.hist.scene), tok)

    def _called(self, rr):
        if rr[0] is not self._tok:
            return
        self.show_parts(rr[1] or [])

    def show_parts(self, parts):
        self.parts = parts
        self._filling = True
        self.tree.clear()
        self.setVisible(bool(parts))
        for src, roots in nest(parts).items():
            top = QTreeWidgetItem([src.split('/')[-1].split('\\')[-1] or 'parts', ''])
            top.setForeground(0, QColor(170, 176, 190))
            self.tree.addTopLevelItem(top)

            def add(parent, node):
                p, kids = node
                it = QTreeWidgetItem([p['name'], ''])
                it.setData(0, Qt.UserRole, int(p['oid']))
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(1, Qt.Unchecked if p['hidden'] else Qt.Checked)
                if p['pairs']:
                    it.setToolTip(0, 'pid %s; interfaces: %s' % (p['pid'], ', '.join('%s %s' % (m, k) for k, m in
                                                                                      sorted(p['pairs'].items()))))
                parent.addChild(it)
                for rid, nm in sorted(p['regions'].items()):
                    r = QTreeWidgetItem(['region: %s' % nm, ''])
                    r.setData(0, Qt.UserRole, int(p['oid']))
                    r.setForeground(0, QColor(150, 160, 175))
                    it.addChild(r)
                for k in kids:
                    add(it, k)
            for r in roots:
                add(top, r)
        self.tree.expandAll()
        self._filling = False

    def _clicked(self, it, col):
        oid = it.data(0, Qt.UserRole)
        if oid is None or col == 1:
            return
        self.win.viewport.selected = [int(oid)]
        self.win.props.set_selection([int(oid)])

    def _toggled(self, it, col):
        if self._filling or col != 1:
            return
        oid = it.data(0, Qt.UserRole)
        if oid is None:
            return
        p = next((q for q in self.parts if int(q['oid']) == int(oid)), None)
        want_shown = it.checkState(1) == Qt.Checked
        if want_shown and (p is None or p['hidden']):
            self.eng.post('view', 'toggle', [int(oid)])     # hidden -> shown (the view job has no plain 'show')
        elif not want_shown:
            self.eng.post('view', 'hide', [int(oid)])
        if p is not None:
            p['hidden'] = not want_shown

    def _isolate(self):
        it = self.tree.currentItem()
        oid = None if it is None else it.data(0, Qt.UserRole)
        if oid is not None:
            self.eng.post('view', 'isolate', [int(oid)])
