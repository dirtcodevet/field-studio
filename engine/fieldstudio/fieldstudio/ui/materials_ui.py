"""Materials & Generators (fs-materials lane, 2026-11-14; INTERFACE_DESIGN_2026-11-13 5.1: the material and engine
work for seeds). A dock beside Properties:

  Apply to       the selected object
  Library        family chips (Stone, Brick & clay, Wood, ...), the family's materials with their measured colour, the
                 chosen material's card: plain words ("heavy . grippy . brittle . porous"), finish, colour variety, the
                 physics that travel with it (each number M measured / D derived / E estimated, sources on hover)
                 -> Make it this material (one undo step: look, colour, generators, a lattice fine enough for them)
  This object    its size, volume and weight from the material's density
  Generators     Stone surface (seed + New, roughness), Edge wear (chips per metre), Cracks (depth, width, seed, Add),
                 Vary across copies (count, spacing, axis -> Make copies: each copy its own seed)

The engine behind it is materials/ (library, commands, generate); the heavy work (sampling the form for chips and
cracks, the volume) runs on the engine thread through its 'call' job, and each edit is pushed as one history command.
Installed from main_window.py's plug-in block (a guarded install(win)), like the other *_ui modules.
"""
import math

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDockWidget, QDoubleSpinBox, QFormLayout, QFrame, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QScrollArea,
                               QSpinBox, QToolButton, QVBoxLayout, QWidget)


def _srgb(c):
    c = max(0.0, min(1.0, float(c)))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def _chip(rgb, size=22):
    pm = QPixmap(size, size)
    pm.fill(QColor(*[int(round(_srgb(v) * 255)) for v in rgb]))
    return QIcon(pm)


def _num(x, nd):
    x = round(float(x), nd)
    return str(int(x)) if nd == 0 or x == int(x) else str(x)


def _rng(v, unit='', k=1.0, nd=2):
    """a value or a range as text"""
    if v is None:
        return '-'
    if isinstance(v, dict):
        if 'dry' in v:
            return _rng(v['dry'], unit, k, nd) + ' dry'
        if 'typical' in v:
            return _rng(v['typical'], unit, k, nd)
        return '-'
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return '%s-%s%s' % (_num(v[0] * k, nd), _num(v[1] * k, nd), unit)
    if isinstance(v, (int, float)):
        return '%s%s' % (_num(v * k, nd), unit)
    return str(v)


class MaterialsPanel(QObject):
    def __init__(self, win):
        super().__init__(win)
        from ..materials import library as LIB, commands as MC
        self.LIB, self.MC = LIB, MC
        self.win = win
        self.oid = None
        self.mid = None
        self._pending = {}
        self.w = QWidget()
        self.w.setObjectName('materials_panel')
        v = QVBoxLayout(self.w)
        v.setContentsMargins(8, 8, 8, 8)
        self.apply_to = QLabel('Apply to: (select an object)')
        self.apply_to.setObjectName('mat_apply_to')
        v.addWidget(self.apply_to)
        # ---- library ----
        g = QGroupBox('Library')
        gl = QVBoxLayout(g)
        chips = QGridLayout()
        self.fam_group = QButtonGroup(self.w)
        self.fam_group.setExclusive(True)
        self.families = LIB.families()
        for i, (name, ms) in enumerate(self.families):
            b = QToolButton()
            b.setText(name.replace('&', '&&'))   # fs-shell-6: a button reads '&' as a key letter ('Brick _clay')
            b.setCheckable(True)
            b.setEnabled(bool(ms))
            b.setObjectName('mat_family_' + name.split()[0].lower())
            self.fam_group.addButton(b, i)
            chips.addWidget(b, i // 3, i % 3)
        gl.addLayout(chips)
        self.list = QListWidget()
        self.list.setObjectName('mat_list')
        self.list.setMaximumHeight(150)
        gl.addWidget(self.list)
        self.card = QLabel('')
        self.card.setObjectName('mat_card')
        self.card.setWordWrap(True)
        self.card.setTextFormat(Qt.RichText)
        gl.addWidget(self.card)
        f = QFormLayout()
        self.finish = QComboBox()
        self.finish.setObjectName('mat_finish')
        self.variety = QComboBox()
        self.variety.setObjectName('mat_variety')
        f.addRow('Finish', self.finish)
        f.addRow('Colour', self.variety)
        gl.addLayout(f)
        self.apply_btn = QPushButton('Make it this material')
        self.apply_btn.setObjectName('mat_apply')
        gl.addWidget(self.apply_btn)
        v.addWidget(g)
        # ---- this object ----
        g2 = QGroupBox('This object')
        g2l = QVBoxLayout(g2)
        self.props = QLabel('-')
        self.props.setObjectName('mat_props')
        self.props.setTextFormat(Qt.RichText)
        self.props.setWordWrap(True)
        g2l.addWidget(self.props)
        v.addWidget(g2)
        # ---- generators ----
        g3 = QGroupBox('Generators (same seed, same stone)')
        g3l = QFormLayout(g3)
        self.seed = QSpinBox()
        self.seed.setObjectName('gen_seed')
        self.seed.setRange(1, 99999999)
        self.seed.setValue(1)
        nb = QPushButton('New')
        nb.setObjectName('gen_seed_new')
        nb.clicked.connect(lambda: self.seed.setValue(self.seed.value() * 7 % 999983 + 3))
        sh = QHBoxLayout()
        sh.addWidget(self.seed)
        sh.addWidget(nb)
        g3l.addRow('Surface seed', sh)
        self.rough = QDoubleSpinBox()
        self.rough.setObjectName('gen_rough')
        self.rough.setRange(0.5, 4.0)
        self.rough.setSingleStep(0.25)
        self.rough.setValue(1.0)
        self.rough.setPrefix('x ')
        self.rough.setToolTip('the relief and roughness of the finish, times this')
        g3l.addRow('Roughness', self.rough)
        self.chips = QDoubleSpinBox()
        self.chips.setObjectName('gen_chips')
        self.chips.setRange(0.0, 60.0)
        self.chips.setValue(12.0)
        self.chips.setSuffix(' chips / m')
        g3l.addRow('Edge wear', self.chips)
        self.weather = QSpinBox()
        self.weather.setObjectName('gen_weather')
        self.weather.setRange(0, 100)
        self.weather.setSingleStep(10)
        self.weather.setValue(0)
        self.weather.setSuffix(' %')
        self.weather.setToolTip('weathering: 0 as made, 100 an old exposed piece (the measured weathered colour, '
                                'arrises worn round, the grains standing out)')
        g3l.addRow('Weathering', self.weather)
        self.bedding = QComboBox()
        self.bedding.setObjectName('gen_bedding')
        for k, lab in (('natural', 'Natural bed'), ('face', 'Face-bedded'), ('edge', 'Edge-bedded')):
            self.bedding.addItem(lab, k)
        self.bedding.setToolTip('how the block was cut from its bed: natural bed (beds horizontal, as quarried), '
                                'face-bedded (beds parallel to the front face), edge-bedded (beds vertical, across it)')
        g3l.addRow('Bedding', self.bedding)
        self.deposit = QSpinBox()
        self.deposit.setObjectName('gen_deposit')
        self.deposit.setRange(0, 100)
        self.deposit.setSingleStep(10)
        self.deposit.setSuffix(' %')
        self.deposit.setToolTip('a deposit lying on the surface: settles on tops and in pores, wiped off the edges')
        self.deposit_kind = QComboBox()
        self.deposit_kind.setObjectName('gen_deposit_kind')
        for k, lab in (('dust', 'Dust'), ('lime', 'Lime / salts'), ('soot', 'Soot')):
            self.deposit_kind.addItem(lab, k)
        self.exposure = QComboBox()
        self.exposure.setObjectName('gen_exposure')
        for ev, lab in ((0.0, 'Sheltered'), (0.5, 'Part exposed'), (1.0, 'In the rain')):
            self.exposure.addItem(lab, ev)
        self.exposure.setToolTip('sheltered: dust settles on tops; in the rain: tops washed, streaks down the sides, '
                                 'gathered at the base')
        dh = QHBoxLayout()
        dh.addWidget(self.deposit)
        dh.addWidget(self.deposit_kind)
        dh.addWidget(self.exposure)
        g3l.addRow('Deposit', dh)
        self.gen_apply = QPushButton('Apply surface')
        self.gen_apply.setObjectName('gen_apply')
        g3l.addRow('', self.gen_apply)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        g3l.addRow(line)
        self.crack_depth = QSpinBox()
        self.crack_depth.setObjectName('crack_depth')
        self.crack_depth.setRange(5, 100)
        self.crack_depth.setValue(50)
        self.crack_depth.setSuffix(' % through')
        self.crack_width = QDoubleSpinBox()
        self.crack_width.setObjectName('crack_width')
        self.crack_width.setRange(0.2, 5.0)
        self.crack_width.setSingleStep(0.1)
        self.crack_width.setValue(0.7)
        self.crack_width.setSuffix(' mm')
        self.crack_width.setToolTip('the opening at the mouth; it closes toward the tip')
        self.crack_seed = QSpinBox()
        self.crack_seed.setObjectName('crack_seed')
        self.crack_seed.setRange(1, 999999)
        self.crack_seed.setValue(1)
        g3l.addRow('Crack depth', self.crack_depth)
        g3l.addRow('Crack width', self.crack_width)
        g3l.addRow('Crack seed', self.crack_seed)
        self.crack_btn = QPushButton('Add crack')
        self.crack_btn.setObjectName('crack_add')
        g3l.addRow('', self.crack_btn)
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        g3l.addRow(line2)
        self.vary_n = QSpinBox()
        self.vary_n.setObjectName('vary_count')
        self.vary_n.setRange(1, 50)
        self.vary_n.setValue(4)
        self.vary_sp = QDoubleSpinBox()
        self.vary_sp.setObjectName('vary_spacing')
        self.vary_sp.setRange(1.0, 5000.0)
        self.vary_sp.setValue(240.0)
        self.vary_sp.setSuffix(' mm')
        self.vary_ax = QComboBox()
        self.vary_ax.setObjectName('vary_axis')
        self.vary_ax.addItems(['X', 'Y', 'Z'])
        g3l.addRow('Vary: copies', self.vary_n)
        g3l.addRow('Spacing', self.vary_sp)
        g3l.addRow('Along', self.vary_ax)
        self.vary_btn = QPushButton('Make copies (Vary)')
        self.vary_btn.setObjectName('vary_make')
        g3l.addRow('', self.vary_btn)
        v.addWidget(g3)
        self.status = QLabel('')
        self.status.setObjectName('mat_status')
        self.status.setWordWrap(True)
        v.addWidget(self.status)
        v.addStretch(1)
        # ---- wiring ----
        self.fam_group.idClicked.connect(self._family)
        self.list.currentItemChanged.connect(self._pick)
        self.apply_btn.clicked.connect(self.apply)
        self.gen_apply.clicked.connect(self.apply)
        self.crack_btn.clicked.connect(self.add_crack)
        self.vary_btn.clicked.connect(self.vary)
        win.engine.callDone.connect(self._done)
        win.props.selectionChanged.connect(self._selection)
        # 11-14 (measured: brushes 31 -> 3 fps): sceneChanged comes with every stroke batch; it only marks the
        # properties stale, and they refresh 0.6 s after the scene stops changing, never while a stroke is held, only
        # while this panel is shown (no engine-thread work during a stroke)
        self._props_dirty = False
        self._props_timer = QTimer(self)
        self._props_timer.setSingleShot(True)
        self._props_timer.setInterval(600)
        self._props_timer.timeout.connect(self._props_idle)
        win.engine.sceneChanged.connect(self._scene_changed)
        self.w.installEventFilter(self)              # shown again: a stale reading is refreshed once
        b0 = self.fam_group.button(0)
        if b0 is not None and b0.isEnabled():
            b0.setChecked(True)
            self._family(0)
        self._enable()

    # ---- library ------------------------------------------------------------------------------------------------
    def _family(self, i):
        self.list.clear()
        for mid in self.families[i][1]:
            it = QListWidgetItem(self.LIB.display_name(mid))
            it.setData(Qt.UserRole, mid)
            vs = self.LIB.varieties(mid)
            if vs:
                first = next(iter(vs.values()))
                it.setIcon(_chip(first['albedo_lin']))
            if not self.LIB.look_ready(mid):
                it.setToolTip('Physics from cited measurements; its look (colour as numbers) is not measured yet')
                it.setForeground(QColor(150, 150, 150))
            self.list.addItem(it)
        if self.list.count():
            self.list.setCurrentRow(0)

    def _pick(self, cur, _prev=None):
        if cur is None:
            return
        mid = cur.data(Qt.UserRole)
        self.mid = mid
        c = self.LIB.card(mid)
        ph = c['physics']
        q = lambda k: ' <span style="color:#888">(%s)</span>' % ph.get(k) if ph.get(k) else ''
        rows = [
            '<b>%s</b> &nbsp; <i>%s</i>' % (c['name'], c['words']),
            'Density %s kg/m&sup3;%s' % (_rng(ph['density_kg_m3'], nd=0), q('density_q')),
            'Friction (stone on stone) %s%s' % (_rng(ph['friction_static']), q('friction_q')),
            'Porosity %s &nbsp; water absorption %s' % (_rng(ph['porosity'], ' %', 100, 0),
                                                       _rng(ph['water_absorption_pct'], ' %', 1, 1)),
            'Strength: compressive %s MPa, tensile %s MPa; toughness %s MPa&middot;m<sup>0.5</sup>' % (
                _rng(ph['compressive_MPa'], nd=0), _rng(ph['tensile_MPa'], nd=0), _rng(ph['toughness_MPa_m05'])),
            'Hardness %s &nbsp; feels %s &deg;C at 20 &deg;C' % (_rng(ph['hardness'], nd=1), _rng(ph['feels_C_at_20'], nd=1)),
        ]
        if not c['look_ready']:
            rows.append('<span style="color:#b70">Look not measured yet: physics only.</span>')
        self.card.setText('<br>'.join(rows))
        self.card.setToolTip('\n'.join('%s: %s [%s]' % (k, s.get('cite'), s.get('lic')) for k, s in c['sources'].items()))
        self.finish.clear()
        from ..materials.library import FINISH_WORDS
        for fn in c['finishes']:
            self.finish.addItem(FINISH_WORDS.get(fn, fn), fn)
        self.variety.clear()
        for name, vv in self.LIB.varieties(mid).items():
            self.variety.addItem(_chip(vv['albedo_lin']), name.replace('_', '-').capitalize(), name)
            self.variety.setItemData(self.variety.count() - 1, vv.get('sample', ''), Qt.ToolTipRole)
        self._enable()

    # ---- selection and properties ----------------------------------------------------------------------------------
    def _selection(self, oids):
        # 11-14: selectionChanged also comes with the SAME object (a click on it, a stroke's press): nothing to do then
        # (measured: one 170-350 ms engine job per stroke, panel closed or open); a new object's properties follow the
        # same rule as a scene change (0.6 s, shown panel only)
        oid = oids[0] if oids else None
        if oid == self.oid:
            return
        self.oid = oid
        self._enable()
        self._props_dirty = True
        if oid is None:
            self.props.setText('-')
        elif self.w.isVisible():
            self._props_timer.start()
        self._load_from_object()

    def _load_from_object(self):
        """the selected object's own material and generator settings into the controls (its seed stays its own)"""
        oid = self.oid
        if oid is None:
            return

        def fn(eng, oid=oid):
            sc = eng.hist.scene
            if sc.index_of(oid) < 0:
                return None
            o = sc.get(oid)
            m = dict(o.material or {})
            m.pop('look', None)
            gens = {}
            for n in o.base.walk():
                if n.kind == 'matgen':
                    sp = n.params.get('spec') or {}
                    gens.setdefault(sp.get('gen'), sp)
            return {'material': m, 'chips': (gens.get('chips') or {}).get('per_m')}
        self._call(fn, self._apply_loaded)

    def _apply_loaded(self, d):
        if not d or not d['material'].get('lib'):
            return
        m = d['material']
        mid = m['lib']
        for i, (name, ms) in enumerate(self.families):
            if mid in ms:
                b = self.fam_group.button(i)
                if b is not None and not b.isChecked():
                    b.setChecked(True)
                    self._family(i)
                break
        for r in range(self.list.count()):
            if self.list.item(r).data(Qt.UserRole) == mid:
                if self.list.currentRow() != r:
                    self.list.setCurrentRow(r)
                break
        k = self.finish.findData(m.get('finish'))
        if k >= 0:
            self.finish.setCurrentIndex(k)
        k = self.variety.findData(m.get('variety'))
        if k >= 0:
            self.variety.setCurrentIndex(k)
        if m.get('seed') is not None:
            self.seed.setValue(int(m['seed']))
        if m.get('rough') is not None:
            self.rough.setValue(float(m['rough']))
        if d.get('chips') is not None:
            self.chips.setValue(float(d['chips']))
        self.weather.setValue(int(round(100.0 * float(m.get('weather', 0.0) or 0.0))))
        self.deposit.setValue(int(round(100.0 * float(m.get('deposit', 0.0) or 0.0))))
        k = self.deposit_kind.findData(m.get('deposit_kind', 'dust'))
        if k >= 0:
            self.deposit_kind.setCurrentIndex(k)
        k = self.exposure.findData(float(m.get('exposure', 0.0) or 0.0))
        if k >= 0:
            self.exposure.setCurrentIndex(k)
        k = self.bedding.findData(m.get('bedding', 'natural'))
        if k >= 0:
            self.bedding.setCurrentIndex(k)

    def _enable(self):
        has = self.oid is not None
        ready = bool(self.mid) and self.LIB.look_ready(self.mid)
        self.apply_btn.setEnabled(has and ready)
        for b in (self.gen_apply, self.crack_btn, self.vary_btn):
            b.setEnabled(has)
        name = None
        if has:
            try:
                name = self.win.engine.hist.scene.get(self.oid).name
            except Exception:
                name = None
        self.apply_to.setText('Apply to: %s' % (name or '(select an object)'))

    def _call(self, fn, then):
        token = object()
        self._pending[id(token)] = (token, then)
        self.win.engine.post('call', fn, token)

    def _done(self, res):
        token, value = res
        ent = self._pending.pop(id(token), None)
        if ent is None or ent[0] is not token:
            return
        if type(value).__name__ == 'CallFailed':
            self.status.setText('Failed: %s' % str(getattr(value, 'error', value))[:300])
            return
        ent[1](value)

    def _scene_changed(self, *_):
        self._props_dirty = True
        if self.oid is not None and self.w.isVisible():
            self._props_timer.start()                # restarted by each change: fires once the changes stop

    def _props_idle(self):
        if not self._props_dirty or self.oid is None or not self.w.isVisible():
            return
        eng = self.win.engine                       # 11-14 (some cost remained on Grab / Flatten):
        from PySide6.QtWidgets import QApplication  # wait while ANY button is held, the engine is busy or interactive
        import time as _time
        if (getattr(eng, '_stroke', None) is not None or QApplication.mouseButtons() != Qt.NoButton
                or _time.perf_counter() < float(getattr(eng, '_interactive_until', 0.0) or 0.0)
                or (hasattr(eng, 'idle') and not eng.idle())):
            self._props_timer.start()
            return
        self._refresh_props()

    def eventFilter(self, obj, ev):
        if obj is self.w and ev.type() == QEvent.Show and self._props_dirty and self.oid is not None:
            self._props_timer.start()
        return False

    def _refresh_props(self):
        self._props_dirty = False
        oid = self.oid
        if oid is None:
            self.props.setText('-')
            return
        MC = self.MC

        def fn(eng, oid=oid):
            sc = eng.hist.scene
            if sc.index_of(oid) < 0:
                return None
            o_ = sc.get(oid)
            if not (o_.material or {}).get('lib') and MC.fast_props(sc, oid) is None:
                return {'volume_L': None}            # no library material on a sculpted form: nothing to weigh, no job
            d = MC.properties(sc, oid, n=48)         # a sculpted form: sampled at 48^3 (error ~(h / size)^2)
            dm = d.pop('dims_m', None)
            if dm is None:                           # not a plain box / sphere: measured (its own size however turned)
                from ..tools import measure as MS
                lo, hi = MS.extents(sc, oid, n=32, axes=MS.own_axes(sc.get(oid)))
                dm = [float(b - a) for a, b in zip(lo, hi)]
            d['dims_mm'] = [round(float(x) * 1e3, 1) for x in dm]
            return d
        self._call(fn, self._show_props)

    def _show_props(self, d):
        if not d:
            self.props.setText('-')
            return
        if d.get('volume_L') is None:
            self.props.setText('No library material yet: its weight shows once it has one')
            return
        dims = ' x '.join('%.1f' % x for x in d.get('dims_mm', []))
        rows = ['Size %s mm' % dims, 'Volume %.4f L' % d['volume_L']]
        if d.get('mass_kg') is not None:
            rows.append('<b>Weight %.2f kg</b> (%s, %s kg/m&sup3;)' % (d['mass_kg'], self.LIB.display_name(d['material']),
                                                                    _rng(d['density_kg_m3'], nd=0)))
            rows.append('%s &nbsp; finish %s, seed %s' % (d.get('words', ''), d.get('finish'), d.get('seed')))
        self.props.setText('<br>'.join(rows))

    # ---- edits ------------------------------------------------------------------------------------------------------
    def apply(self):
        if self.oid is None or not self.mid:
            return
        oid, mid = self.oid, self.mid
        fin = self.finish.currentData()
        var = self.variety.currentData()
        seed, rough, chips = self.seed.value(), self.rough.value(), self.chips.value()
        wthr = self.weather.value() / 100.0
        dpo, dpk, dpx = self.deposit.value() / 100.0, self.deposit_kind.currentData(), self.exposure.currentData()
        bed = self.bedding.currentData()
        MC = self.MC
        self.status.setText('Making it %s ...' % self.LIB.display_name(mid))

        def fn(eng):
            return MC.assign_args(eng.hist.scene, oid, mid, fin, seed, rough, chips, variety=var, weather=wthr,
                                  deposit=dpo, deposit_kind=dpk, exposure=dpx, bedding=bed)

        def then(args):
            self.win.engine.post('cmd', MC.AssignMaterial(**args), None)
            self.status.setText('%s: %s, %s, seed %d' % (self.LIB.display_name(mid), args['material']['finish'],
                                                         args['material'].get('variety'), seed))
        self._call(fn, then)

    def add_crack(self):
        if self.oid is None:
            return
        oid = self.oid
        depth, width, seed = self.crack_depth.value() / 100.0, self.crack_width.value(), self.crack_seed.value()
        MC = self.MC
        self.status.setText('Cracking ...')

        def fn(eng):
            return MC.crack_args(eng.hist.scene, oid, seed, depth, width)

        def then(a):
            from ..core.history import SetBase
            self.win.engine.post('cmd', SetBase(a['oid'], a['tree'], a['next_id']), None)
            sp = [n for n in _walk(a['tree']) if n.get('kind') == 'matgen'
                  and (n['params'].get('spec') or {}).get('gen') == 'crack']
            info = (sp[-1]['params']['spec'].get('info') or {}) if sp else {}
            self.status.setText('Crack: %s mm deep, %s mm wide at the mouth' % (info.get('depth_mm'),
                                                                                  info.get('width_mm')))
        self._call(fn, then)

    def vary(self):
        """Vary across copies = fs-tools' Array (tools.array: the app's ONE Array, decided 11-14): the copies and
        their spacing are its; each copy's own stone comes from materials.commands.reseed, which ArrayCopies calls for a
        seeded Vary (f4b665cc)"""
        if self.oid is None:
            return
        jobs = getattr(self.win.engine, '_job_handlers', None) or {}
        if 'tools.array' not in jobs:
            self.status.setText('Array is not available (fs-tools\' tools.array job is not registered)')
            return
        n, sp, ax = self.vary_n.value(), self.vary_sp.value() * 1e-3, self.vary_ax.currentIndex()
        axis = [1.0 if k == ax else 0.0 for k in range(3)]
        self.win.engine.post('tools.array', self.oid, {'mode': 'line', 'count': int(n),
                                                       'step': [float(a) * sp for a in axis],
                                                       'vary': {'seed': int(self.seed.value())}})
        self.status.setText('%d copies (Array), each its own seed' % n)


def _walk(t):
    yield t
    for c in t.get('children', []):
        yield from _walk(c)


def make_panel(win):
    """the panel's widget for ui/panels2.py (its Materials & Generators panel mounts it; pin, pop out and dock are the
    panel's own), scrolling inside a ~360 px dock"""
    P = MaterialsPanel(win)
    win.materials_panel = P
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.NoFrame)
    sa.setWidget(P.w)
    sa.setMinimumWidth(300)
    return sa


def install(win):
    P = MaterialsPanel(win)
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.NoFrame)
    sa.setWidget(P.w)
    d = QDockWidget('Materials')
    d.setObjectName('dock_materials')
    d.setWidget(sa)
    d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable)
    d.setMinimumWidth(320)
    win.addDockWidget(Qt.RightDockWidgetArea, d)
    docks = getattr(win, 'docks', {})
    if 'Properties' in docks:
        win.tabifyDockWidget(docks['Properties'], d)
    docks['Materials'] = d
    win.materials_panel = P
    tm = getattr(win, 'menu_tools', None)
    if tm is not None:
        a = tm.addAction('Materials && Generators...')
        a.triggered.connect(lambda: (d.show(), d.raise_()))
    return P
