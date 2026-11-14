"""Marks: circle a spot, or mark the exact point, inside Field Studio, read by other tools and sessions (fs-shell,
2026-11-11; design 4.14, chunk 9: instead of drawing on a photo of the screen, the user circles or points on the
model itself and names the mark).

A mark is a numbered pin (a point on a surface) or a region (a drawn loop, projected onto the surface), with its name
and note, the object it sits on, its place in that object's own frame (so it follows the object), and the camera it was
marked from. Marks save in the project (extra['marks']) and, as they are made, in the shared view folder, where any
session reads them without opening the project:

    from fieldstudio.tools.marks import read_shared
    v = read_shared()                       # {'marks': [...], 'camera': {...}, 'project': ..., 'time': ...}
    for m in v['marks']:
        print(m['n'], m['name'], m['note'], m['object'], m['centre_mm'], m['still'])

The folder: FIELDSTUDIO_SHARE_DIR, else the app setting 'share_dir', else ~/FieldStudio/shared_view. Stills are the
view as it was seen with the marks drawn, named mark_<n>.png. No Qt here."""
import json
import os
import time

KEEP = ('n', 'kind', 'oid', 'object', 'name', 'note', 'q', 'n_local', 'outline_q', 'samples_q', 'camera', 'time',
        'still', 'centre_mm', 'normal', 'size_mm', 'pixels')


def share_dir(setting=None):
    d = os.environ.get('FIELDSTUDIO_SHARE_DIR') or setting or os.path.join(os.path.expanduser('~'), 'FieldStudio',
                                                                           'shared_view')
    return os.path.abspath(d)


def clean(m):
    """a mark as saved: plain lists, no world-space caches"""
    out = {}
    for k in KEEP:
        if k in m and m[k] is not None:
            v = m[k]
            out[k] = v.tolist() if hasattr(v, 'tolist') else v
    return out


def write_shared(marks, camera=None, project=None, directory=None):
    """marks.json (the whole set, newest view) in the shared folder; returns its path"""
    d = share_dir(directory)
    os.makedirs(d, exist_ok=True)
    doc = {'app': 'Field Studio', 'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'project': project, 'camera': camera,
           'marks': [clean(m) for m in marks],
           'read_me': 'Marks made in Field Studio: n (its number), kind (pin / region), name, note, object, '
                      'centre_mm (world), size_mm (a region\'s extent), still (the view with the marks drawn), camera '
                      '(where it was marked from), q / outline_q (the object\'s own frame, metres).'}
    p = os.path.join(d, 'marks.json')
    tmp = p + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, p)
    return p


def read_shared(directory=None):
    """-> the shared view (marks.json) or {'marks': []} when none have been made"""
    p = os.path.join(share_dir(directory), 'marks.json')
    if not os.path.exists(p):
        return {'marks': []}
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def project_marks(fsproj):
    """-> the marks saved in a project folder (its extra), or []"""
    for name in ('extra.json', 'project.json', 'meta.json'):
        p = os.path.join(fsproj, name)
        if os.path.exists(p):
            try:
                with open(p, encoding='utf-8') as f:
                    d = json.load(f)
            except Exception:
                continue
            ex = d.get('extra', d) if isinstance(d, dict) else {}
            if isinstance(ex, dict) and ex.get('marks'):
                return ex['marks']
    return []
