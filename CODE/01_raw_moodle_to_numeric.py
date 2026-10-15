
# ============================================================
# 0. Mount Drive & configuration
# ============================================================
from pathlib import Path
from contextlib import contextmanager
from collections import Counter, defaultdict
import io, re, json, gzip, bz2, lzma, zipfile, sqlite3, math, hashlib, html, os
import numpy as np
import pandas as pd

def preview(obj):
    if os.environ.get('CAREKM_VERBOSE','0') == '1':
        try: print(obj.head().to_string(index=False))
        except Exception: print(obj)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PACKAGE_ROOT / 'DATA' / 'MAIN_LMS' / 'SOURCE'
OUT_DIR = PACKAGE_ROOT / 'REPRODUCED' / 'RAW_PREPROCESSING'
WORK_DIR = OUT_DIR / '_work' 

OUT_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)

DUMP_BASENAME = 'mysql-export'
EXCLUDE_COURSE_IDS = {'0', '1'}
COURSE_END_QUANTILE = 0.95
PROGRESS_EVERY = 1_000_000
FORCE_REBUILD = False
USE_OFFICIAL_COURSE_DATES = True
MAX_REASONABLE_OFFICIAL_DURATION_DAYS = 730

# Output numeric convention.
NUMERIC_DECIMALS = 6
CSV_FLOAT_FORMAT = '%.6f'
pd.set_option('display.float_format', lambda x: f'{x:.6f}')

def save_csv6(df, path, index=False):
    # Persist floating-point values with exactly 6 digits after the decimal.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        index=index,
        float_format=CSV_FLOAT_FORMAT,
        na_rep=''
    )

def file_ready(path):
    # A cached output is valid only if the file exists and is non-empty.
    # This prevents SKIP-SUDAH from accepting interrupted/zero-byte files.
    p = Path(path)
    return p.exists() and p.is_file() and p.stat().st_size > 0

def safe_read_csv(path, *args, empty_columns=None, **kwargs):
    # Robustly read cached CSVs. A zero-byte/headerless CSV from an interrupted
    # previous run is treated as empty rather than crashing with EmptyDataError.
    p = Path(path)
    if not file_ready(p):
        return pd.DataFrame(columns=list(empty_columns or []))
    try:
        return pd.read_csv(p, *args, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=list(empty_columns or []))

def stage_cached(stage_name, *paths):
    # Completed persistent stages are skipped only when every required output
    # is present and non-empty.
    ready = (not FORCE_REBUILD) and bool(paths) and all(file_ready(p) for p in paths)
    if ready:
        print(f'[cached] {stage_name} | valid output found; stage skipped.')
    return ready


# Sessionized dwell: gap above threshold is a session break (0 attributed dwell).
DWELL_POLICY = 'SESSION_BREAK_ZERO'
GRAF_CRITICAL_SECONDS = {
    'content_stay': 10 * 60,
    'forum_stay': 10 * 60,
    'selfass_stay_proxy': 10 * 60,
    'quiz_stay_results_proxy': 10 * 60,
    'exercise_stay_proxy': 20 * 60,
    'outline_stay_proxy': 3 * 60,
    'navigation_overview_stay': 3 * 60,
}

CONTENT_MODULES = {
    'resource','url','page','book','lesson','folder','glossary','wiki','scorm','imscp'
}
OTHER_INTERACTIVE_MODULES = {
    'choice','questionnaire','workshop','data','oublog','pcast'
}
GENERIC_FAMILIES = sorted(
    CONTENT_MODULES | {'assign','quiz','forum'} | OTHER_INTERACTIVE_MODULES
)
BROAD_GROUPS = ['overview','content','assignment','quiz','forum','other_interactive']

# Course-selection policy is intentionally semantic-first.
SEMANTIC_STRENGTH_WEIGHT = {
    'VERY_STRONG': 3.0,
    'STRONG': 2.5,
    'MODERATE': 1.5,
    'WEAK': 0.75,
}

print('INPUT :', DATASET_DIR / DUMP_BASENAME)
print('OUTPUT:', OUT_DIR)
print('Feature-bank families:', len(GENERIC_FAMILIES))

# ============================================================
# 1. MySQL dump reader, parser, and schema detection
# ============================================================
def locate_dump(data_dir: Path, basename=DUMP_BASENAME) -> Path:
    exact = data_dir / basename
    if exact.exists() and exact.is_file():
        return exact
    candidates = sorted(p for p in data_dir.glob(basename + '*') if p.is_file())
    if not candidates:
        raise FileNotFoundError(f'No file beginning with {basename!r} in {data_dir}')
    if len(candidates) > 1:
        raise RuntimeError('Multiple mysql-export candidates found: ' + ', '.join(p.name for p in candidates))
    return candidates[0]

def _magic(path: Path, n=8):
    with open(path, 'rb') as f:
        return f.read(n)

@contextmanager
def open_dump_text(path):
    path = Path(path)
    magic = _magic(path)
    if magic.startswith(b'\x1f\x8b'):
        raw = gzip.open(path, 'rb')
        text = io.TextIOWrapper(raw, encoding='utf-8', errors='replace', newline='')
        try:
            yield text
        finally:
            text.close()
        return
    if magic.startswith(b'BZh'):
        raw = bz2.open(path, 'rb')
        text = io.TextIOWrapper(raw, encoding='utf-8', errors='replace', newline='')
        try:
            yield text
        finally:
            text.close()
        return
    if magic.startswith(b'\xfd7zXZ\x00'):
        raw = lzma.open(path, 'rb')
        text = io.TextIOWrapper(raw, encoding='utf-8', errors='replace', newline='')
        try:
            yield text
        finally:
            text.close()
        return
    if magic.startswith(b'PK\x03\x04'):
        zf = zipfile.ZipFile(path)
        members = [n for n in zf.namelist() if not n.endswith('/')]
        sqlish = [n for n in members if n.lower().endswith(('.sql','.dump','.txt')) or 'mysql' in Path(n).name.lower()]
        if len(sqlish) == 1:
            member = sqlish[0]
        elif len(members) == 1:
            member = members[0]
        else:
            zf.close()
            raise RuntimeError('ZIP contains multiple files; SQL member cannot be selected safely.')
        raw = zf.open(member, 'r')
        text = io.TextIOWrapper(raw, encoding='utf-8', errors='replace', newline='')
        try:
            yield text
        finally:
            text.close()
            zf.close()
        return
    text = open(path, 'r', encoding='utf-8', errors='replace', newline='')
    try:
        yield text
    finally:
        text.close()

_CREATE_RE = re.compile(
    r'^\s*CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+(?:`([^`]+)`|([^\s(]+))\s*\(',
    re.I
)
_INSERT_RE = re.compile(r'^\s*INSERT\s+INTO\s+(?:`([^`]+)`|([^\s(]+))', re.I)
_COLUMN_RE = re.compile(r'^\s*`([^`]+)`\s+(.+?)(?:,\s*)?$')
_PARTITION_RE = re.compile(r'(?:^|_)log_course_(\d+)$', re.I)

def scan_schema(dump_path):
    schema_rows, insert_counts = [], Counter()
    current_table, in_create, ordinal = None, False, 0
    with open_dump_text(dump_path) as f:
        for line in f:
            if not in_create:
                m = _CREATE_RE.match(line)
                if m:
                    current_table = m.group(1) or m.group(2)
                    in_create, ordinal = True, 0
                    continue
                m = _INSERT_RE.match(line)
                if m:
                    insert_counts[m.group(1) or m.group(2)] += 1
                continue
            if line.strip().startswith(')'):
                in_create, current_table = False, None
                continue
            m = _COLUMN_RE.match(line)
            if m:
                ordinal += 1
                schema_rows.append({
                    'table': current_table,
                    'ordinal': ordinal,
                    'column': m.group(1),
                    'definition': m.group(2).strip()
                })
    schema = pd.DataFrame(schema_rows)
    if schema.empty:
        raise RuntimeError('No CREATE TABLE schema could be parsed.')

    table_rows = []
    for table, g in schema.groupby('table', sort=False):
        table_rows.append({
            'table': table,
            'n_columns': len(g),
            'insert_statements': int(insert_counts.get(table, 0)),
            'columns': '|'.join(g.sort_values('ordinal')['column'].tolist())
        })
    return schema, pd.DataFrame(table_rows)

def _table_columns(schema):
    return {
        t: g.sort_values('ordinal')['column'].tolist()
        for t, g in schema.groupby('table', sort=False)
    }

def _partition_course_id(table):
    m = _PARTITION_RE.search(str(table))
    return m.group(1) if m else None

def detect_structure(schema):
    cols = _table_columns(schema)
    standard = [
        t for t in cols
        if t == 'logstore_standard_log' or t.endswith('_logstore_standard_log')
    ]
    if len(standard) == 1:
        t = standard[0]
        low = {c.lower() for c in cols[t]}
        return {
            'source_layout':'standard_log_table',
            'log_tables':[t],
            'course_id_source':'courseid_column',
            'can_auto_build': {'userid','courseid'} <= low
        }

    parts = []
    for t, c in cols.items():
        cid = _partition_course_id(t)
        if cid is None:
            continue
        low = {x.lower() for x in c}
        if 'userid' in low and ('action' in low or 'module' in low):
            parts.append(t)
    if parts:
        return {
            'source_layout':'partitioned_course_log_tables',
            'log_tables':parts,
            'course_id_source':'table_name_log_course_<courseid>',
            'can_auto_build':True
        }
    return {'source_layout':'unknown','log_tables':[],'can_auto_build':False}

def _mysql_unescape(text):
    mp = {
        '0':'\x00','b':'\b','n':'\n','r':'\r','t':'\t',
        'Z':'\x1a','\\':'\\',"'":"'",'"':'"'
    }
    out, i = [], 0
    while i < len(text):
        if text[i] == '\\' and i + 1 < len(text):
            out.append(mp.get(text[i+1], text[i+1]))
            i += 2
        else:
            out.append(text[i])
            i += 1
    return ''.join(out)

def _parse_scalar(token):
    token = token.strip()
    if token.upper() == 'NULL':
        return None
    if len(token) >= 2 and token[0] in {"'", '"'} and token[-1] == token[0]:
        return _mysql_unescape(token[1:-1])
    return token

def _parse_tuple_text(t):
    s = t[1:-1]
    fields, start, quote, escaped = [], 0, None, False
    for i, ch in enumerate(s):
        if escaped:
            escaped = False
            continue
        if ch == '\\' and quote is not None:
            escaped = True
            continue
        if quote is not None:
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            continue
        if ch == ',':
            fields.append(_parse_scalar(s[start:i]))
            start = i + 1
    fields.append(_parse_scalar(s[start:]))
    return fields

def _iter_tuple_strings(text):
    depth, quote, escaped, start = 0, None, False, None
    for i, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == '\\' and quote is not None:
            escaped = True
            continue
        if quote is not None:
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            continue
        if ch == '(':
            if depth == 0:
                start = i
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:i+1]
                start = None

def _statement_complete(text):
    quote, escaped = None, False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == '\\' and quote is not None:
            escaped = True
            continue
        if quote is not None:
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            continue
        if ch == ';':
            return True
    return False

def _parse_insert_statement(statement, table, schema_columns):
    rx = re.compile(
        rf'^\s*INSERT\s+INTO\s+`?{re.escape(table)}`?'
        rf'(?:\s*\((.*?)\))?\s+VALUES\s*',
        re.I | re.S
    )
    m = rx.match(statement)
    if not m:
        return
    cols = (
        [c.strip().strip('`') for c in m.group(1).split(',')]
        if m.group(1) else list(schema_columns)
    )
    values_text = statement[m.end():]
    if ';' in values_text:
        values_text = values_text.rsplit(';', 1)[0]
    for tup in _iter_tuple_strings(values_text):
        vals = _parse_tuple_text(tup)
        if len(vals) != len(cols):
            raise RuntimeError(
                f'Column/value mismatch in {table}: {len(cols)} vs {len(vals)}'
            )
        yield dict(zip(cols, vals))

def iter_table_rows(dump_path, schema, target_tables):
    table_cols = _table_columns(schema)
    targets = set(target_tables)
    collecting, target, buf = False, None, []
    with open_dump_text(dump_path) as f:
        for line in f:
            if not collecting:
                m = _INSERT_RE.match(line)
                if not m:
                    continue
                table = m.group(1) or m.group(2)
                if table not in targets:
                    continue
                collecting, target, buf = True, table, [line]
            else:
                buf.append(line)

            statement = ''.join(buf)
            if not _statement_complete(statement):
                continue

            table = target
            collecting, target, buf = False, None, []
            for row in _parse_insert_statement(statement, table, table_cols[table]):
                yield table, row

def iter_relevant_log_rows(dump_path, schema, detection):
    targets = set(detection['log_tables'])
    course_map = (
        {t:_partition_course_id(t) for t in targets}
        if detection['source_layout'] == 'partitioned_course_log_tables'
        else {}
    )
    for table, row in iter_table_rows(dump_path, schema, targets):
        yield table, course_map.get(table), row

def row_core(course_from_table, row):
    low = {str(k).lower():v for k,v in row.items()}
    user = str(low.get('userid') or '')
    course = (
        str(course_from_table)
        if course_from_table is not None
        else str(low.get('courseid') or low.get('course') or '')
    )
    raw_t = (
        low.get('timecreated')
        if low.get('timecreated') not in (None, '')
        else low.get('time')
    )
    try:
        t = int(float(raw_t)) if raw_t not in (None, '') else None
    except Exception:
        t = None
    return low, user, course, t

def table_by_signature(schema, required_columns, preferred_suffix=None, excluded_suffixes=()):
    cols = _table_columns(schema)
    required = {str(x).lower() for x in required_columns}
    candidates = []

    for table, names in cols.items():
        low = {str(x).lower() for x in names}
        if not required <= low:
            continue
        if any(str(table).endswith(suf) for suf in excluded_suffixes):
            continue
        candidates.append(table)

    if preferred_suffix:
        preferred = [
            t for t in candidates
            if t == preferred_suffix or t.endswith('_' + preferred_suffix)
        ]
        if len(preferred) == 1:
            return preferred[0]
        if preferred:
            candidates = preferred

    if len(candidates) == 1:
        return candidates[0]

    return sorted(candidates, key=lambda x: (len(str(x)), str(x)))[0] if candidates else None

# ----------------------------
# V4 generic helpers
# ----------------------------
def safe_int(x, default=None):
    try:
        if x in (None, '', 'None', 'nan'):
            return default
        return int(float(x))
    except Exception:
        return default

def strip_html_text(value):
    if value in (None, ''):
        return ''
    s = html.unescape(str(value))
    s = re.sub(r'<script\b[^>]*>.*?</script>', ' ', s, flags=re.I|re.S)
    s = re.sub(r'<style\b[^>]*>.*?</style>', ' ', s, flags=re.I|re.S)
    s = re.sub(r'<[^>]+>', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def modality_from_mime_filename(mimetype='', filename='', url=''):
    mime = str(mimetype or '').lower().strip()
    name = (str(filename or '') + ' ' + str(url or '')).lower()

    if mime.startswith('image/'):
        return 'image'
    if mime.startswith('audio/'):
        return 'audio'
    if mime.startswith('video/'):
        return 'video'
    if mime.startswith('text/'):
        return 'text'
    if mime in {
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/rtf'
    }:
        return 'text'
    if mime in {
        'application/vnd.ms-powerpoint',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation'
    }:
        return 'presentation'

    # Extension fallback only when MIME is absent/uninformative.
    if re.search(r'\.(png|jpe?g|gif|svg|webp)(?:\?|#|\s|$)', name):
        return 'image'
    if re.search(r'\.(mp3|wav|ogg|m4a|aac)(?:\?|#|\s|$)', name):
        return 'audio'
    if re.search(r'\.(mp4|webm|mov|mkv|avi)(?:\?|#|\s|$)', name):
        return 'video'
    if re.search(r'\.(pdf|txt|docx?|rtf)(?:\?|#|\s|$)', name):
        return 'text'
    if re.search(r'\.(pptx?|odp)(?:\?|#|\s|$)', name):
        return 'presentation'
    return 'other'


# ============================================================
# Stage 00 — raw/schema audit
# ============================================================
DUMP_PATH = locate_dump(DATASET_DIR)
print('Detected dump:', DUMP_PATH)
print('Size (GB):', round(DUMP_PATH.stat().st_size / 1024**3, 3))

schema_path = OUT_DIR / '00_mysql_table_schema.csv'
tables_path = OUT_DIR / '00_mysql_tables.csv'
detection_path = OUT_DIR / '00_mysql_detection.json'

if FORCE_REBUILD or not (
    file_ready(schema_path) and file_ready(tables_path) and file_ready(detection_path)
):
    schema, tables = scan_schema(DUMP_PATH)
    detection = detect_structure(schema)
    save_csv6(schema, schema_path, index=False)
    save_csv6(tables, tables_path, index=False)
    detection_path.write_text(json.dumps(detection, indent=2), encoding='utf-8')
else:
    print('[cached] Stage 00 raw/schema audit')
    schema = pd.read_csv(schema_path)
    tables = pd.read_csv(tables_path)
    detection = json.loads(detection_path.read_text(encoding='utf-8'))

assert detection.get('can_auto_build'), detection

structural_tables = {
    'modules': table_by_signature(
        schema, {'id','name'}, preferred_suffix='modules',
        excluded_suffixes=('_course_modules',)
    ),
    'course_modules': table_by_signature(
        schema, {'id','course','module','instance','section'}, preferred_suffix='course_modules'
    ),
    'course_sections': table_by_signature(
        schema, {'id','course','section','sequence'}, preferred_suffix='course_sections'
    ),
    'course': table_by_signature(
        schema, {'id','fullname','shortname'}, preferred_suffix='course'
    ),
    'context': table_by_signature(
        schema, {'id','contextlevel','instanceid'}, preferred_suffix='context'
    ),
    'files': table_by_signature(
        schema, {'id','contextid','filename','component','filearea'}, preferred_suffix='files'
    ),
    'page': table_by_signature(
        schema, {'id','content'}, preferred_suffix='page'
    ),
    'book_chapters': table_by_signature(
        schema, {'id','bookid','content'}, preferred_suffix='book_chapters'
    ),
    'lesson_pages': table_by_signature(
        schema, {'id','lessonid','contents'}, preferred_suffix='lesson_pages'
    ),
    'url': table_by_signature(
        schema, {'id','externalurl'}, preferred_suffix='url'
    ),
}
print(json.dumps({
    'source_layout': detection['source_layout'],
    'n_log_tables': len(detection['log_tables']),
    'structural_tables': structural_tables
}, indent=2))

# ============================================================
# Stage 01 — structural / semantic metadata
# ============================================================
course_metadata_path = OUT_DIR / '01_course_metadata.csv'
object_inventory_path = OUT_DIR / '01_course_object_inventory.csv'
navigation_order_path = OUT_DIR / '01_course_navigation_order.csv'
modality_inventory_path = OUT_DIR / '01_course_modality_inventory.csv'
structural_audit_path = OUT_DIR / '01_structural_semantic_audit.json'


def build_structural_semantic_metadata():
    targets = sorted({t for t in structural_tables.values() if t})
    if not targets:
        course_df = pd.DataFrame(columns=[
            'course','fullname','shortname','official_start','official_end'
        ])
        object_inv = pd.DataFrame(columns=[
            'course','family','object_id','cmid','instance','visible'
        ])
        navigation_order = pd.DataFrame(columns=[
            'course','cmid','family','section_number',
            'position_in_section','global_position','order_source'
        ])
        modality_inventory = pd.DataFrame(columns=[
            'course','cmid','family','modality',
            'evidence_source','evidence_detail'
        ])
        audit = {'available': False, 'reason': 'No structural tables detected'}
        save_csv6(course_df, course_metadata_path, index=False)
        save_csv6(object_inv, object_inventory_path, index=False)
        save_csv6(navigation_order, navigation_order_path, index=False)
        save_csv6(modality_inventory, modality_inventory_path, index=False)
        structural_audit_path.write_text(
            json.dumps(audit, indent=2), encoding='utf-8'
        )
        return course_df, object_inv, navigation_order, modality_inventory, audit

    module_map = {}
    courses = []
    cms = []
    sections = []
    context_to_cmid = {}
    file_modalities_by_context = defaultdict(set)
    page_text_ids = set()
    book_text_ids = set()
    lesson_text_ids = set()
    url_modality_by_instance = {}

    table_to_key = {v:k for k,v in structural_tables.items() if v}

    for table, row in iter_table_rows(DUMP_PATH, schema, targets):
        key = table_to_key.get(table)
        low = {str(k).lower():v for k,v in row.items()}

        if key == 'modules':
            mid = str(low.get('id') or '')
            name = str(low.get('name') or '').lower().strip()
            if mid and name:
                module_map[mid] = name

        elif key == 'course':
            cid = str(low.get('id') or '')
            if cid and cid not in EXCLUDE_COURSE_IDS:
                courses.append({
                    'course': cid,
                    'fullname': str(low.get('fullname') or ''),
                    'shortname': str(low.get('shortname') or ''),
                    'official_start': safe_int(low.get('startdate')),
                    'official_end': safe_int(low.get('enddate')),
                })

        elif key == 'course_modules':
            cid = str(low.get('course') or '')
            cmid = str(low.get('id') or '')
            if cid and cmid and cid not in EXCLUDE_COURSE_IDS:
                if str(low.get('deletioninprogress') or '0') == '1':
                    continue
                cms.append({
                    'course': cid,
                    'cmid': cmid,
                    'module_id': str(low.get('module') or ''),
                    'instance': str(low.get('instance') or ''),
                    'section_id': str(low.get('section') or ''),
                    'visible': str(low.get('visible') or ''),
                })

        elif key == 'course_sections':
            cid = str(low.get('course') or '')
            if cid and cid not in EXCLUDE_COURSE_IDS:
                sections.append({
                    'section_id': str(low.get('id') or ''),
                    'course': cid,
                    'section_number': safe_int(low.get('section'), 0),
                    'sequence': str(low.get('sequence') or ''),
                    'name': str(low.get('name') or ''),
                    'visible': str(low.get('visible') or ''),
                })

        elif key == 'context':
            # Moodle module context level is 70. We use only module contexts
            # so files can be attached to a concrete course-module id (cmid).
            level = safe_int(low.get('contextlevel'))
            context_id = str(low.get('id') or '')
            instanceid = str(low.get('instanceid') or '')
            if level == 70 and context_id and instanceid:
                context_to_cmid[context_id] = instanceid

        elif key == 'files':
            filename = str(low.get('filename') or '')
            if filename in {'', '.'}:
                continue
            filesize = safe_int(low.get('filesize'), 1)
            if filesize == 0:
                continue
            contextid = str(low.get('contextid') or '')
            if not contextid:
                continue
            component = str(low.get('component') or '').lower()
            filearea = str(low.get('filearea') or '').lower()
            # Explicitly exclude obvious user-submission/draft areas.
            if any(tok in component for tok in ('assignsubmission','submission')):
                continue
            if any(tok in filearea for tok in ('submission','draft','backup','trash')):
                continue
            modality = modality_from_mime_filename(
                low.get('mimetype'), filename, ''
            )
            if modality != 'other':
                file_modalities_by_context[contextid].add(
                    (modality, component, filearea)
                )

        elif key == 'page':
            instance = str(low.get('id') or '')
            txt = strip_html_text(low.get('content'))
            if instance and len(txt) >= 40:
                page_text_ids.add(instance)

        elif key == 'book_chapters':
            bookid = str(low.get('bookid') or '')
            txt = strip_html_text(low.get('content'))
            if bookid and len(txt) >= 40:
                book_text_ids.add(bookid)

        elif key == 'lesson_pages':
            lessonid = str(low.get('lessonid') or '')
            txt = strip_html_text(low.get('contents'))
            if lessonid and len(txt) >= 40:
                lesson_text_ids.add(lessonid)

        elif key == 'url':
            instance = str(low.get('id') or '')
            exturl = str(low.get('externalurl') or '')
            modality = modality_from_mime_filename('', '', exturl)
            if instance and modality != 'other':
                url_modality_by_instance[instance] = modality

    course_df = pd.DataFrame(courses).drop_duplicates('course') if courses else pd.DataFrame(
        columns=['course','fullname','shortname','official_start','official_end']
    )
    cm_df = pd.DataFrame(
        cms,
        columns=['course','cmid','module_id','instance','section_id','visible']
    )
    section_df = pd.DataFrame(
        sections,
        columns=['section_id','course','section_number','sequence','name','visible']
    )

    if not cm_df.empty:
        cm_df['family'] = cm_df['module_id'].map(module_map).fillna('')
        cm_df = cm_df[cm_df['family'] != ''].copy()

    # Object inventory.
    if cm_df.empty:
        object_inv = pd.DataFrame(columns=['course','family','object_id','cmid','instance','visible'])
    else:
        object_inv = cm_df.assign(
            object_id=cm_df['family'] + ':' + cm_df['cmid'].astype(str)
        )[['course','family','object_id','cmid','instance','visible']].drop_duplicates()

    # Canonical course-module order from course_sections.sequence.
    nav_rows = []
    if not cm_df.empty:
        cm_lookup = cm_df.set_index('cmid').to_dict('index')
        courses_with_seq = set()

        if not section_df.empty:
            for course, g in section_df.groupby('course'):
                g = g.sort_values(['section_number','section_id'], kind='stable')
                global_pos = 0
                for sr in g.itertuples(index=False):
                    seq = [x.strip() for x in str(sr.sequence).split(',') if x.strip()]
                    for pos_in_section, cmid in enumerate(seq, start=1):
                        meta = cm_lookup.get(cmid)
                        if not meta or str(meta['course']) != str(course):
                            continue
                        global_pos += 1
                        nav_rows.append({
                            'course': str(course),
                            'cmid': str(cmid),
                            'family': meta['family'],
                            'section_number': int(sr.section_number or 0),
                            'position_in_section': pos_in_section,
                            'global_position': global_pos,
                            'order_source': 'course_sections_sequence'
                        })
                        courses_with_seq.add(str(course))

        # Fallback for courses without usable section sequence.
        for course, g in cm_df.groupby('course'):
            course = str(course)
            if course in courses_with_seq:
                continue
            tmp = g.copy()
            tmp['_section_num'] = pd.to_numeric(tmp['section_id'], errors='coerce').fillna(10**12)
            tmp['_cm_num'] = pd.to_numeric(tmp['cmid'], errors='coerce').fillna(10**12)
            tmp = tmp.sort_values(['_section_num','_cm_num','cmid'], kind='stable')
            for j, r in enumerate(tmp.itertuples(index=False), start=1):
                nav_rows.append({
                    'course': course,
                    'cmid': str(r.cmid),
                    'family': r.family,
                    'section_number': safe_int(r.section_id, 0),
                    'position_in_section': np.nan,
                    'global_position': j,
                    'order_source': 'fallback_section_cmid'
                })

    navigation_order = pd.DataFrame(
        nav_rows,
        columns=[
            'course','cmid','family','section_number',
            'position_in_section','global_position','order_source'
        ]
    )

    # Modality inventory at course-module level.
    modality_rows = []
    if not cm_df.empty:
        cm_by_id = cm_df.set_index('cmid').to_dict('index')

        # File MIME / filename evidence via module context.
        for contextid, evidence_set in file_modalities_by_context.items():
            cmid = context_to_cmid.get(contextid)
            meta = cm_by_id.get(str(cmid)) if cmid is not None else None
            if not meta:
                continue
            family = str(meta['family'])
            if family not in CONTENT_MODULES:
                continue
            for modality, component, filearea in evidence_set:
                modality_rows.append({
                    'course': str(meta['course']),
                    'cmid': str(cmid),
                    'family': family,
                    'modality': modality,
                    'evidence_source': 'files_mime_or_extension',
                    'evidence_detail': f'{component}:{filearea}'
                })

        # HTML text evidence from page/book/lesson structures.
        for r in cm_df.itertuples(index=False):
            family = str(r.family)
            instance = str(r.instance)
            text_supported = (
                (family == 'page' and instance in page_text_ids) or
                (family == 'book' and instance in book_text_ids) or
                (family == 'lesson' and instance in lesson_text_ids)
            )
            if text_supported:
                modality_rows.append({
                    'course': str(r.course),
                    'cmid': str(r.cmid),
                    'family': family,
                    'modality': 'text',
                    'evidence_source': 'html_text_content',
                    'evidence_detail': family
                })

            if family == 'url' and instance in url_modality_by_instance:
                modality_rows.append({
                    'course': str(r.course),
                    'cmid': str(r.cmid),
                    'family': family,
                    'modality': url_modality_by_instance[instance],
                    'evidence_source': 'url_extension',
                    'evidence_detail': 'externalurl'
                })

    modality_inventory = pd.DataFrame(
        modality_rows,
        columns=[
            'course','cmid','family','modality',
            'evidence_source','evidence_detail'
        ]
    )
    if not modality_inventory.empty:
        modality_inventory = modality_inventory.drop_duplicates(
            ['course','cmid','modality','evidence_source','evidence_detail']
        )

    save_csv6(course_df, course_metadata_path, index=False)
    save_csv6(object_inv, object_inventory_path, index=False)
    save_csv6(navigation_order, navigation_order_path, index=False)
    save_csv6(modality_inventory, modality_inventory_path, index=False)
    audit = {
        'available': True,
        'tables': structural_tables,
        'module_names_detected': sorted(set(module_map.values())),
        'n_courses_metadata': int(len(course_df)),
        'n_course_module_objects': int(len(object_inv)),
        'n_navigation_rows': int(len(navigation_order)),
        'navigation_order_sources': (
            navigation_order['order_source'].value_counts().to_dict()
            if len(navigation_order) else {}
        ),
        'n_modality_rows': int(len(modality_inventory)),
        'modality_counts': (
            modality_inventory['modality'].value_counts().to_dict()
            if len(modality_inventory) else {}
        ),
        'modality_policy': (
            'Only explicit MIME/extension or HTML text evidence is used; '
            'generic resource type is not mapped to visual/verbal.'
        )
    }
    structural_audit_path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return course_df, object_inv, navigation_order, modality_inventory, audit

stage01_required = [
    structural_audit_path,
    course_metadata_path,
    object_inventory_path,
    navigation_order_path,
    modality_inventory_path,
]
stage01_ready = (not FORCE_REBUILD) and all(file_ready(p) for p in stage01_required)

if stage01_ready:
    print('[cached] Stage 01 structural/semantic metadata | cache valid.')
    structural_audit = json.loads(
        structural_audit_path.read_text(encoding='utf-8')
    )
    course_metadata = safe_read_csv(
        course_metadata_path,
        dtype={'course':str},
        empty_columns=['course','fullname','shortname','official_start','official_end']
    )
    structural_inventory = safe_read_csv(
        object_inventory_path,
        dtype={'course':str,'cmid':str},
        empty_columns=['course','family','object_id','cmid','instance','visible']
    )
    navigation_order = safe_read_csv(
        navigation_order_path,
        dtype={'course':str,'cmid':str},
        empty_columns=[
            'course','cmid','family','section_number',
            'position_in_section','global_position','order_source'
        ]
    )
    modality_inventory = safe_read_csv(
        modality_inventory_path,
        dtype={'course':str,'cmid':str},
        empty_columns=[
            'course','cmid','family','modality',
            'evidence_source','evidence_detail'
        ]
    )
else:
    existing_bad = [
        str(p.name) for p in stage01_required
        if Path(p).exists() and not file_ready(p)
    ]
    if existing_bad and not FORCE_REBUILD:
        print(
            '[CACHE-TIDAK-LENGKAP] Stage 01 akan dibangun ulang. '
            'File kosong/tidak valid: ' + ', '.join(existing_bad)
        )
    course_metadata, structural_inventory, navigation_order, modality_inventory, structural_audit = build_structural_semantic_metadata()

print(json.dumps(structural_audit, indent=2))

# ============================================================
# Stage 02 — empirical P95 + official course windows
# ============================================================
course_window_path = OUT_DIR / '02_course_windows.csv'
course_time_db = WORK_DIR / '_02_course_times.sqlite'


def build_course_windows():
    if course_time_db.exists():
        course_time_db.unlink()

    con = sqlite3.connect(course_time_db)
    con.execute('PRAGMA journal_mode=OFF')
    con.execute('PRAGMA synchronous=OFF')
    con.execute('PRAGMA temp_store=MEMORY')
    con.execute('CREATE TABLE course_time(course TEXT NOT NULL, t INTEGER NOT NULL)')

    batch, parsed, valid = [], 0, 0
    for _, course_from_table, row in iter_relevant_log_rows(DUMP_PATH, schema, detection):
        parsed += 1
        _, user, course, t = row_core(course_from_table, row)
        if (
            user in {'','0','None'} or course in EXCLUDE_COURSE_IDS or
            course in {'','None'} or t is None
        ):
            continue
        batch.append((course, t))
        valid += 1
        if len(batch) >= 200_000:
            con.executemany('INSERT INTO course_time(course,t) VALUES (?,?)', batch)
            con.commit(); batch = []
        if parsed % PROGRESS_EVERY == 0:
            print(f'Window pass parsed {parsed:,}; valid timestamps {valid:,}')

    if batch:
        con.executemany('INSERT INTO course_time(course,t) VALUES (?,?)', batch)
        con.commit()

    con.execute('CREATE INDEX idx_course_time ON course_time(course,t)')
    con.commit()

    stats = pd.read_sql_query(
        '''SELECT course, COUNT(*) AS n_events,
                  MIN(t) AS empirical_start, MAX(t) AS raw_max_time
           FROM course_time GROUP BY course''',
        con,
        dtype={'course':str}
    )

    rows = []
    for r in stats.itertuples(index=False):
        offset = max(0, int(math.ceil(COURSE_END_QUANTILE * int(r.n_events))) - 1)
        p95 = int(con.execute(
            'SELECT t FROM course_time WHERE course=? ORDER BY t LIMIT 1 OFFSET ?',
            (str(r.course), offset)
        ).fetchone()[0])
        rows.append({
            'course': str(r.course),
            'n_events': int(r.n_events),
            'empirical_start': int(r.empirical_start),
            'empirical_end_p95': p95,
            'raw_max_time': int(r.raw_max_time),
            'p95_rank_1based': offset + 1,
        })
    con.close()

    out = pd.DataFrame(rows)
    if not course_metadata.empty:
        meta = course_metadata[['course','official_start','official_end']].copy()
        meta['course'] = meta['course'].astype(str)
        out = out.merge(meta, on='course', how='left')
    else:
        out['official_start'] = np.nan
        out['official_end'] = np.nan

    final_rows = []
    for r in out.itertuples(index=False):
        emp_s = int(r.empirical_start)
        emp_e = int(r.empirical_end_p95)
        off_s = safe_int(r.official_start)
        off_e = safe_int(r.official_end)

        valid_official = False
        if USE_OFFICIAL_COURSE_DATES and off_s and off_e and off_e > off_s:
            dur_days = (off_e - off_s) / 86400
            valid_official = 1 <= dur_days <= MAX_REASONABLE_OFFICIAL_DURATION_DAYS

        if valid_official:
            start, end, source = off_s, off_e, 'official_start_end'
        elif USE_OFFICIAL_COURSE_DATES and off_s and off_s < emp_e:
            start, end, source = off_s, emp_e, 'official_start_empirical_p95_end'
        elif USE_OFFICIAL_COURSE_DATES and off_e and off_e > emp_s:
            start, end, source = emp_s, off_e, 'empirical_start_official_end'
        else:
            start, end, source = emp_s, emp_e, 'empirical_p95'

        if end <= start:
            start, end, source = emp_s, emp_e, 'empirical_p95_fallback_invalid_window'

        final_rows.append({
            **r._asdict(),
            'course_start': int(start),
            'course_end': int(end),
            'window_source': source,
            'duration_days': max(0.0, (end-start)/86400),
            'tail_days_after_p95': max(0.0, (int(r.raw_max_time)-emp_e)/86400),
        })

    final = pd.DataFrame(final_rows).sort_values('course')
    save_csv6(final, course_window_path, index=False)
    audit = {
        'parsed_rows': parsed,
        'valid_timestamp_rows': valid,
        'courses': int(len(final)),
        'course_end_quantile': COURSE_END_QUANTILE,
        'window_source_counts': final['window_source'].value_counts().to_dict()
    }
    (OUT_DIR / '02_course_window_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return final

if FORCE_REBUILD or not file_ready(course_window_path):
    course_windows = build_course_windows()
else:
    print('[cached] Stage 02 course windows')
    course_windows = pd.read_csv(course_window_path, dtype={'course':str})

course_windows['course'] = course_windows['course'].astype(str)
window_map = {
    r.course:(int(r.course_start), int(r.course_end))
    for r in course_windows.itertuples(index=False)
}
print('Courses with final observation windows:', len(course_windows))
preview(course_windows.head())

# ============================================================
# Stage 02 — event taxonomy
# ============================================================
def norm_module(low):
    component = str(low.get('component') or '').lower().strip()
    module = str(low.get('module') or '').lower().strip()
    src = component or module
    if src.startswith('mod_'):
        src = src[4:]
    return src or module

def norm_action(low):
    action = str(low.get('action') or '').lower().strip()
    aliases = {
        'viewed':'view',
        'submitted':'submit',
        'created':'add',
        'updated':'update',
        'started':'start',
        'ended':'end'
    }
    return aliases.get(action, action)

def taxonomy(low):
    module = norm_module(low)
    action = norm_action(low)
    eventname = str(low.get('eventname') or '').lower()

    if 'course_viewed' in eventname:
        return ('course_view', 'overview', 'overview')

    if module == 'course':
        if action == 'view':
            return ('course_view', 'overview', 'overview')
        if action == 'view section':
            return ('course_section_view', 'overview', 'overview')
        return None

    if module in CONTENT_MODULES:
        valid = {'view','view chapter','start','end'}
        if action in valid:
            return (f'{module}_view', 'content', module)
        return None

    if module == 'assign':
        mp = {
            'view':'assign_view',
            'view submit assignment form':'assign_submission_form_view',
            'view confirm submit assignment form':'assign_submit_confirm_view',
            'submit':'assign_submit',
            'submit for grading':'assign_submit_for_grading',
            'submission statement accepted':'assign_submission_statement',
            'view feedback':'assign_feedback_view',
        }
        if action in mp:
            return (mp[action], 'assignment', 'assign')
        return None

    if module == 'quiz':
        mp = {
            'view':'quiz_view',
            'attempt':'quiz_attempt_start',
            'continue attempt':'quiz_attempt_continue',
            'review':'quiz_review',
            'close attempt':'quiz_close_attempt',
            'view summary':'quiz_summary_view',
            'report':'quiz_report_view'
        }
        if action in mp:
            return (mp[action], 'quiz', 'quiz')
        return None

    if module == 'forum':
        mp = {
            'view forum':'forum_view',
            'view forums':'forum_index_view',
            'view discussion':'forum_discussion_view',
            'add post':'forum_post_add',
            'add discussion':'forum_discussion_add',
            'update post':'forum_post_update',
            'delete post':'forum_post_delete',
            'delete discussion':'forum_discussion_delete',
            'search':'forum_search'
        }
        if action in mp:
            return (mp[action], 'forum', 'forum')
        return None

    if (
        module in OTHER_INTERACTIVE_MODULES and
        action in {'view','view all','attempt','submit','add','update'}
    ):
        return (
            f'{module}_{action.replace(" ","_")}',
            'other_interactive',
            module
        )

    return None

def object_id_from_row(low, module):
    cmid = str(low.get('cmid') or '').strip()
    if cmid not in {'','0','None','nan'}:
        return f'{module}:{cmid}'

    info = str(low.get('info') or '').strip()
    if info not in {'','0','None','nan'}:
        return f'{module}:info:{info}'

    return None

taxonomy_note = {
    'content_modules': sorted(CONTENT_MODULES),
    'other_interactive_modules': sorted(OTHER_INTERACTIVE_MODULES),
    'time_semantics': 'event timestamps are used to reconstruct sessionized dwell proxies; not direct Moodle duration fields',
    'forum_post': 'actual add post/add discussion events',
    'quiz_revision_warning': 'continue attempt/review are retained as observable events but are not renamed Graf quiz_revisions',
    'design': 'atomic event -> family -> broad group; recoverable Graf features plus generic behavioral features are retained'
}
(OUT_DIR / '03_taxonomy_config.json').write_text(
    json.dumps(taxonomy_note, indent=2),
    encoding='utf-8'
)
print(json.dumps(taxonomy_note, indent=2))

# ============================================================
# Stage 04 — streaming aggregation
# ============================================================
agg_db = WORK_DIR / '_04_rich_aggregate.sqlite'
vocab_path = OUT_DIR / '04_event_vocabulary.csv'
aggregation_audit_path = OUT_DIR / '04_aggregation_audit.json'

def flush_batch(
    con,
    event_counter,
    family_counter,
    broad_counter,
    uc_counter,
    object_batch,
    course_object_batch,
    active_day_batch,
    family_day_batch,
    broad_day_batch
):
    if event_counter:
        con.executemany(
            '''
            INSERT INTO event_agg(
                user,course,event_class,event_count,first_t,last_t
            )
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(user,course,event_class) DO UPDATE SET
                event_count=event_count+excluded.event_count,
                first_t=MIN(event_agg.first_t,excluded.first_t),
                last_t=MAX(event_agg.last_t,excluded.last_t)
            ''',
            [
                (u,c,e,v[0],v[1],v[2])
                for (u,c,e),v in event_counter.items()
            ]
        )

    if family_counter:
        con.executemany(
            '''
            INSERT INTO family_agg(
                user,course,family,event_count,first_t,last_t,
                early_count,middle_count,late_count
            )
            VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(user,course,family) DO UPDATE SET
                event_count=event_count+excluded.event_count,
                first_t=MIN(family_agg.first_t,excluded.first_t),
                last_t=MAX(family_agg.last_t,excluded.last_t),
                early_count=early_count+excluded.early_count,
                middle_count=middle_count+excluded.middle_count,
                late_count=late_count+excluded.late_count
            ''',
            [
                (u,c,f,*v)
                for (u,c,f),v in family_counter.items()
            ]
        )

    if broad_counter:
        con.executemany(
            '''
            INSERT INTO broad_agg(
                user,course,broad_group,event_count,first_t,last_t,
                early_count,middle_count,late_count
            )
            VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(user,course,broad_group) DO UPDATE SET
                event_count=event_count+excluded.event_count,
                first_t=MIN(broad_agg.first_t,excluded.first_t),
                last_t=MAX(broad_agg.last_t,excluded.last_t),
                early_count=early_count+excluded.early_count,
                middle_count=middle_count+excluded.middle_count,
                late_count=late_count+excluded.late_count
            ''',
            [
                (u,c,g,*v)
                for (u,c,g),v in broad_counter.items()
            ]
        )

    if uc_counter:
        con.executemany(
            '''
            INSERT INTO uc_activity(
                user,course,mapped_event_count,first_t,last_t
            )
            VALUES (?,?,?,?,?)
            ON CONFLICT(user,course) DO UPDATE SET
                mapped_event_count=mapped_event_count+excluded.mapped_event_count,
                first_t=MIN(uc_activity.first_t,excluded.first_t),
                last_t=MAX(uc_activity.last_t,excluded.last_t)
            ''',
            [
                (u,c,v[0],v[1],v[2])
                for (u,c),v in uc_counter.items()
            ]
        )

    if object_batch:
        con.executemany(
            '''
            INSERT OR IGNORE INTO object_seen(
                user,course,family,object_id
            ) VALUES (?,?,?,?)
            ''',
            list(object_batch)
        )

    if course_object_batch:
        con.executemany(
            '''
            INSERT OR IGNORE INTO course_object_observed(
                course,family,object_id
            ) VALUES (?,?,?)
            ''',
            list(course_object_batch)
        )

    if active_day_batch:
        con.executemany(
            '''
            INSERT OR IGNORE INTO active_day(
                user,course,day_id
            ) VALUES (?,?,?)
            ''',
            list(active_day_batch)
        )

    if family_day_batch:
        con.executemany(
            '''
            INSERT OR IGNORE INTO family_day(
                user,course,family,day_id
            ) VALUES (?,?,?,?)
            ''',
            list(family_day_batch)
        )

    if broad_day_batch:
        con.executemany(
            '''
            INSERT OR IGNORE INTO broad_day(
                user,course,broad_group,day_id
            ) VALUES (?,?,?,?)
            ''',
            list(broad_day_batch)
        )

    con.commit()

def _update_counter(counter, key, t, bucket):
    if key not in counter:
        counter[key] = [1, t, t, 0, 0, 0]
    else:
        v = counter[key]
        v[0] += 1
        v[1] = min(v[1], t)
        v[2] = max(v[2], t)

    if bucket == 'early':
        counter[key][3] += 1
    elif bucket == 'middle':
        counter[key][4] += 1
    else:
        counter[key][5] += 1

def aggregate_fullcourse():
    if agg_db.exists():
        agg_db.unlink()

    con = sqlite3.connect(agg_db)
    con.execute('PRAGMA journal_mode=OFF')
    con.execute('PRAGMA synchronous=OFF')
    con.execute('PRAGMA temp_store=MEMORY')
    con.executescript(
        '''
        CREATE TABLE event_agg(
            user TEXT, course TEXT, event_class TEXT,
            event_count INTEGER, first_t INTEGER, last_t INTEGER,
            PRIMARY KEY(user,course,event_class)
        );

        CREATE TABLE family_agg(
            user TEXT, course TEXT, family TEXT,
            event_count INTEGER, first_t INTEGER, last_t INTEGER,
            early_count INTEGER, middle_count INTEGER, late_count INTEGER,
            PRIMARY KEY(user,course,family)
        );

        CREATE TABLE broad_agg(
            user TEXT, course TEXT, broad_group TEXT,
            event_count INTEGER, first_t INTEGER, last_t INTEGER,
            early_count INTEGER, middle_count INTEGER, late_count INTEGER,
            PRIMARY KEY(user,course,broad_group)
        );

        CREATE TABLE uc_activity(
            user TEXT, course TEXT,
            mapped_event_count INTEGER, first_t INTEGER, last_t INTEGER,
            PRIMARY KEY(user,course)
        );

        CREATE TABLE object_seen(
            user TEXT, course TEXT, family TEXT, object_id TEXT,
            PRIMARY KEY(user,course,family,object_id)
        );

        CREATE TABLE course_object_observed(
            course TEXT, family TEXT, object_id TEXT,
            PRIMARY KEY(course,family,object_id)
        );

        CREATE TABLE active_day(
            user TEXT, course TEXT, day_id INTEGER,
            PRIMARY KEY(user,course,day_id)
        );

        CREATE TABLE family_day(
            user TEXT, course TEXT, family TEXT, day_id INTEGER,
            PRIMARY KEY(user,course,family,day_id)
        );

        CREATE TABLE broad_day(
            user TEXT, course TEXT, broad_group TEXT, day_id INTEGER,
            PRIMARY KEY(user,course,broad_group,day_id)
        );
        '''
    )

    event_counter = {}
    family_counter = {}
    broad_counter = {}
    uc_counter = {}
    object_batch = set()
    course_object_batch = set()
    active_day_batch = set()
    family_day_batch = set()
    broad_day_batch = set()

    vocab = Counter()
    parsed = in_window = mapped = 0

    for _, course_from_table, row in iter_relevant_log_rows(
        DUMP_PATH, schema, detection
    ):
        parsed += 1
        low, user, course, t = row_core(course_from_table, row)

        if (
            user in {'','0','None'} or
            course in EXCLUDE_COURSE_IDS or
            course not in window_map or
            t is None
        ):
            continue

        start, end = window_map[course]
        if t < start or t > end:
            continue

        in_window += 1
        module = norm_module(low)
        action = norm_action(low)
        tx = taxonomy(low)

        vocab[
            (module, action, 'MAPPED' if tx else 'UNMAPPED', tx[0] if tx else '')
        ] += 1

        if tx is None:
            continue

        event_class, broad_group, family = tx
        mapped += 1

        duration = max(end - start, 1)
        rel = (t - start) / duration
        bucket = 'early' if rel <= 0.25 else ('middle' if rel <= 0.75 else 'late')

        # Atomic event aggregation
        ekey = (user, course, event_class)
        if ekey not in event_counter:
            event_counter[ekey] = [1, t, t]
        else:
            v = event_counter[ekey]
            v[0] += 1
            v[1] = min(v[1], t)
            v[2] = max(v[2], t)

        # Family and broad aggregation
        _update_counter(family_counter, (user,course,family), t, bucket)
        _update_counter(broad_counter, (user,course,broad_group), t, bucket)

        # User-course overall
        ukey = (user, course)
        if ukey not in uc_counter:
            uc_counter[ukey] = [1, t, t]
        else:
            v = uc_counter[ukey]
            v[0] += 1
            v[1] = min(v[1], t)
            v[2] = max(v[2], t)

        day_id = int(t // 86400)
        active_day_batch.add((user,course,day_id))
        family_day_batch.add((user,course,family,day_id))
        broad_day_batch.add((user,course,broad_group,day_id))

        oid = object_id_from_row(low, module)
        if oid:
            object_batch.add((user,course,family,oid))
            course_object_batch.add((course,family,oid))

            if broad_group == 'content':
                object_batch.add((user,course,'content',oid))
                course_object_batch.add((course,'content',oid))

            if broad_group in {'assignment','quiz'}:
                object_batch.add((user,course,'assessment',oid))
                course_object_batch.add((course,'assessment',oid))

        if mapped % 100_000 == 0:
            flush_batch(
                con,
                event_counter, family_counter, broad_counter, uc_counter,
                object_batch, course_object_batch, active_day_batch,
                family_day_batch, broad_day_batch
            )
            event_counter = {}
            family_counter = {}
            broad_counter = {}
            uc_counter = {}
            object_batch = set()
            course_object_batch = set()
            active_day_batch = set()
            family_day_batch = set()
            broad_day_batch = set()

        if parsed % PROGRESS_EVERY == 0:
            print(
                f'Pass-2 parsed {parsed:,}; '
                f'in-window {in_window:,}; mapped {mapped:,}'
            )

    flush_batch(
        con,
        event_counter, family_counter, broad_counter, uc_counter,
        object_batch, course_object_batch, active_day_batch,
        family_day_batch, broad_day_batch
    )
    con.close()

    vocab_df = pd.DataFrame([
        {
            'module':k[0],
            'action':k[1],
            'mapping_status':k[2],
            'event_class':k[3],
            'count':v
        }
        for k,v in vocab.most_common()
    ])
    save_csv6(vocab_df, vocab_path, index=False)
    audit = {
        'parsed_rows': parsed,
        'rows_in_fullcourse_window': in_window,
        'mapped_behavior_rows': mapped,
        'mapped_share_of_inwindow': mapped / in_window if in_window else 0
    }
    aggregation_audit_path.write_text(
        json.dumps(audit, indent=2),
        encoding='utf-8'
    )
    return audit

downstream_feature_bank = OUT_DIR / '05_rich_feature_bank_all_user_course.csv'

if (not FORCE_REBUILD) and file_ready(downstream_feature_bank):
    print('[cached] Stage 04 streaming aggregation | downstream feature bank sudah tersedia.')
    audit = (
        json.loads(aggregation_audit_path.read_text(encoding='utf-8'))
        if file_ready(aggregation_audit_path) else {}
    )
elif (not FORCE_REBUILD) and agg_db.exists():
    print('[cached] Stage 04 streaming aggregation | temporary aggregate DB masih tersedia.')
    audit = (
        json.loads(aggregation_audit_path.read_text(encoding='utf-8'))
        if file_ready(aggregation_audit_path) else {}
    )
else:
    audit = aggregate_fullcourse()
    print(json.dumps(audit, indent=2))


# ============================================================
# Stage 04b — raw-boundary sessionized dwell
# ============================================================
dwell_db = WORK_DIR / '_04b_raw_boundary_dwell.sqlite'
dwell_long_path = OUT_DIR / '04b_sessionized_dwell_long.csv'
event_object_unique_path = OUT_DIR / '04b_event_object_unique.csv'
dwell_thresholds_path = OUT_DIR / '04b_dwell_thresholds.csv'
dwell_audit_path = OUT_DIR / '04b_dwell_audit.json'


def build_sessionized_dwell():
    if dwell_db.exists():
        dwell_db.unlink()

    con = sqlite3.connect(dwell_db)
    con.execute('PRAGMA journal_mode=OFF')
    con.execute('PRAGMA synchronous=OFF')
    con.execute('PRAGMA temp_store=MEMORY')
    con.execute('''
        CREATE TABLE raw_event(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            course TEXT NOT NULL,
            t INTEGER NOT NULL,
            module TEXT,
            action TEXT,
            cmid TEXT,
            object_id TEXT,
            is_mapped INTEGER NOT NULL,
            event_class TEXT,
            broad_group TEXT,
            family TEXT
        )
    ''')

    batch = []
    parsed = in_window = mapped = 0

    for _, course_from_table, row in iter_relevant_log_rows(DUMP_PATH, schema, detection):
        parsed += 1
        low, user, course, t = row_core(course_from_table, row)
        if (
            user in {'','0','None'} or course in EXCLUDE_COURSE_IDS or
            course not in window_map or t is None
        ):
            continue
        start, end = window_map[course]
        if t < start or t > end:
            continue
        in_window += 1

        module = norm_module(low)
        action = norm_action(low)
        cmid = str(low.get('cmid') or '').strip()
        if cmid in {'','0','None','nan'}:
            cmid = None

        tx = taxonomy(low)
        if tx is None:
            event_class = broad_group = family = None
            oid = None
            is_mapped = 0
        else:
            event_class, broad_group, family = tx
            oid = object_id_from_row(low, module)
            is_mapped = 1
            mapped += 1

        batch.append((
            user, course, int(t), module, action, cmid, oid,
            is_mapped, event_class, broad_group, family
        ))

        if len(batch) >= 200_000:
            con.executemany(
                '''INSERT INTO raw_event(
                    user,course,t,module,action,cmid,object_id,is_mapped,
                    event_class,broad_group,family
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                batch
            )
            con.commit(); batch = []

        if parsed % PROGRESS_EVERY == 0:
            print(f'Dwell pass parsed {parsed:,}; in-window {in_window:,}; mapped {mapped:,}')

    if batch:
        con.executemany(
            '''INSERT INTO raw_event(
                user,course,t,module,action,cmid,object_id,is_mapped,
                event_class,broad_group,family
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
            batch
        )
        con.commit()

    con.execute('CREATE INDEX idx_raw_event_user_course_time ON raw_event(user,course,t,id)')
    con.execute('CREATE INDEX idx_raw_event_mapped ON raw_event(is_mapped,event_class)')
    con.execute('CREATE INDEX idx_raw_event_cmid ON raw_event(course,cmid)')
    con.commit()

    evobj = pd.read_sql_query(
        '''SELECT user,course,event_class,
                  COUNT(DISTINCT object_id) AS n_unique_objects
           FROM raw_event
           WHERE is_mapped=1 AND object_id IS NOT NULL
           GROUP BY user,course,event_class''',
        con,
        dtype={'user':str,'course':str}
    )
    save_csv6(evobj, event_object_unique_path, index=False)
    # LEAD is computed over ALL raw events. Only current mapped event is classified.
    dwell_sql = '''
    WITH ordered AS (
        SELECT *,
               LEAD(t) OVER (PARTITION BY user,course ORDER BY t,id) AS next_raw_t
        FROM raw_event
    ), classified AS (
        SELECT *,
            CASE
                WHEN is_mapped=0 THEN NULL
                WHEN event_class='quiz_review' THEN 'quiz_stay_results_proxy'
                WHEN event_class='course_view' THEN 'navigation_overview_stay'
                WHEN event_class='course_section_view' THEN 'outline_stay_proxy'
                WHEN broad_group='content' THEN 'content_stay'
                WHEN broad_group='forum' THEN 'forum_stay'
                WHEN broad_group='assignment' THEN 'exercise_stay_proxy'
                WHEN broad_group='quiz' THEN 'selfass_stay_proxy'
                ELSE NULL
            END AS dwell_type,
            CASE
                WHEN event_class='quiz_review' THEN 600
                WHEN event_class='course_view' THEN 180
                WHEN event_class='course_section_view' THEN 180
                WHEN broad_group='content' THEN 600
                WHEN broad_group='forum' THEN 600
                WHEN broad_group='assignment' THEN 1200
                WHEN broad_group='quiz' THEN 600
                ELSE NULL
            END AS critical_sec
        FROM ordered
    )
    SELECT user,course,dwell_type,
           MAX(critical_sec) AS critical_sec,
           SUM(CASE WHEN next_raw_t>t AND (next_raw_t-t)<=critical_sec
                    THEN (next_raw_t-t) ELSE 0 END) AS sessionized_sec,
           SUM(CASE WHEN next_raw_t>t AND (next_raw_t-t)<=critical_sec
                    THEN 1 ELSE 0 END) AS valid_gap_n,
           SUM(CASE WHEN next_raw_t>t AND (next_raw_t-t)>critical_sec
                    THEN 1 ELSE 0 END) AS idle_excluded_n,
           SUM(CASE WHEN next_raw_t=t THEN 1 ELSE 0 END) AS zero_gap_n,
           SUM(CASE WHEN next_raw_t IS NULL THEN 1 ELSE 0 END) AS no_next_n
    FROM classified
    WHERE dwell_type IS NOT NULL
    GROUP BY user,course,dwell_type
    '''

    dwell = pd.read_sql_query(dwell_sql, con, dtype={'user':str,'course':str})
    con.close()

    dwell['sessionized_min'] = dwell['sessionized_sec'] / 60.0
    dwell['avg_valid_gap_sec'] = np.divide(
        dwell['sessionized_sec'], dwell['valid_gap_n'],
        out=np.zeros(len(dwell), dtype=float),
        where=dwell['valid_gap_n'].to_numpy() > 0
    )
    dwell['idle_excluded_share'] = np.divide(
        dwell['idle_excluded_n'], dwell['valid_gap_n'] + dwell['idle_excluded_n'],
        out=np.zeros(len(dwell), dtype=float),
        where=(dwell['valid_gap_n'] + dwell['idle_excluded_n']).to_numpy() > 0
    )
    save_csv6(dwell, dwell_long_path, index=False)
    thresholds = pd.DataFrame([
        {'dwell_type':k,'critical_sec':v,'critical_min':v/60.0,'policy':DWELL_POLICY}
        for k,v in GRAF_CRITICAL_SECONDS.items()
    ])
    save_csv6(thresholds, dwell_thresholds_path, index=False)
    audit = {
        'policy': DWELL_POLICY,
        'boundary_method': 'next valid raw learner event within same learner-course',
        'current_event_requirement': 'current event must be taxonomy-mapped to receive dwell type',
        'over_critical_gap_treatment': 'session break; 0 attributed dwell',
        'parsed_rows': int(parsed),
        'rows_in_window': int(in_window),
        'mapped_rows': int(mapped),
        'critical_seconds': GRAF_CRITICAL_SECONDS,
        'dwell_rows': int(len(dwell)),
        'event_object_unique_rows': int(len(evobj))
    }
    dwell_audit_path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return dwell, evobj, audit

stage05c_done = (
    file_ready(OUT_DIR / '05c_semantic_structure_audit.json')
    and file_ready(OUT_DIR / '05_rich_feature_bank_all_user_course.csv')
)
persistent_dwell_ready = (
    file_ready(dwell_long_path)
    and file_ready(event_object_unique_path)
    and file_ready(dwell_audit_path)
)

if (not FORCE_REBUILD) and persistent_dwell_ready and (dwell_db.exists() or stage05c_done):
    print('[cached] Stage 04b sessionized dwell')
    dwell_long = pd.read_csv(
        dwell_long_path, dtype={'user':str,'course':str}, low_memory=False
    )
    event_object_unique = pd.read_csv(
        event_object_unique_path, dtype={'user':str,'course':str}, low_memory=False
    )
    dwell_audit = json.loads(dwell_audit_path.read_text(encoding='utf-8'))
else:
    dwell_long, event_object_unique, dwell_audit = build_sessionized_dwell()

print(json.dumps(dwell_audit, indent=2))
preview(dwell_long.head(20))


# ============================================================
# Stage 05 — rich numerical feature bank
# ============================================================
feature_bank_path = OUT_DIR / '05_rich_feature_bank_all_user_course.csv'
feature_dictionary_path = OUT_DIR / '05_feature_dictionary.csv'
feature_bank_audit_path = OUT_DIR / '05_feature_bank_audit.json'

def safe_div(a, b, scale=1.0):
    # Robust element-wise division.
    # np.broadcast_arrays prevents scalar defaults (e.g. 0) from producing
    # a 0-D output when the denominator is a Series/array.
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    a_arr, b_arr = np.broadcast_arrays(a_arr, b_arr)

    return np.divide(
        a_arr * scale,
        b_arr,
        out=np.zeros(a_arr.shape, dtype=float),
        where=b_arr != 0
    )

def build_feature_bank():
    con = sqlite3.connect(agg_db)

    uc = pd.read_sql_query(
        'SELECT * FROM uc_activity',
        con,
        dtype={'user':str,'course':str}
    )
    ev = pd.read_sql_query(
        'SELECT * FROM event_agg',
        con,
        dtype={'user':str,'course':str}
    )
    fam = pd.read_sql_query(
        'SELECT * FROM family_agg',
        con,
        dtype={'user':str,'course':str}
    )
    broad = pd.read_sql_query(
        'SELECT * FROM broad_agg',
        con,
        dtype={'user':str,'course':str}
    )
    days = pd.read_sql_query(
        '''
        SELECT user,course,COUNT(*) AS active_days
        FROM active_day
        GROUP BY user,course
        ''',
        con,
        dtype={'user':str,'course':str}
    )
    fdays = pd.read_sql_query(
        '''
        SELECT user,course,family,COUNT(*) AS active_days
        FROM family_day
        GROUP BY user,course,family
        ''',
        con,
        dtype={'user':str,'course':str}
    )
    bdays = pd.read_sql_query(
        '''
        SELECT user,course,broad_group,COUNT(*) AS active_days
        FROM broad_day
        GROUP BY user,course,broad_group
        ''',
        con,
        dtype={'user':str,'course':str}
    )
    obj = pd.read_sql_query(
        '''
        SELECT user,course,family,COUNT(*) AS n_unique
        FROM object_seen
        GROUP BY user,course,family
        ''',
        con,
        dtype={'user':str,'course':str}
    )
    cob = pd.read_sql_query(
        '''
        SELECT course,family,COUNT(*) AS n_course_objects
        FROM course_object_observed
        GROUP BY course,family
        ''',
        con,
        dtype={'course':str}
    )
    con.close()

    rich = uc.merge(days, on=['user','course'], how='left')
    rich['active_days'] = rich['active_days'].fillna(0).astype(float)

    # Course windows.
    w = course_windows[
        ['course','course_start','course_end','duration_days','window_source']
    ].copy()
    w['course'] = w['course'].astype(str)
    rich = rich.merge(w, on='course', how='left')

    duration_sec = (
        rich['course_end'] - rich['course_start']
    ).clip(lower=1)
    course_weeks = np.maximum(duration_sec / 604800, 1.0)
    course_days = np.maximum(duration_sec / 86400, 1.0)

    rich['total_behavior_events'] = rich['mapped_event_count'].astype(float)
    rich['behavior_event_rate_week'] = safe_div(
        rich['total_behavior_events'], course_weeks
    )
    rich['active_span_days'] = (
        (rich['last_t'] - rich['first_t']).clip(lower=0) / 86400
    )
    rich['activity_regularization'] = safe_div(
        rich['active_days'], np.maximum(rich['active_span_days'], 1.0)
    )
    rich['active_day_ratio_course'] = safe_div(
        rich['active_days'], course_days
    )
    rich['first_activity_pct'] = (
        (rich['first_t'] - rich['course_start']) / duration_sec * 100
    ).clip(0,100)
    rich['last_activity_pct'] = (
        (rich['last_t'] - rich['course_start']) / duration_sec * 100
    ).clip(0,100)

    # Atomic event counts: retain provenance.
    evp = ev.pivot_table(
        index=['user','course'],
        columns='event_class',
        values='event_count',
        aggfunc='sum',
        fill_value=0
    )
    evp.columns = ['ev_' + str(c) for c in evp.columns]
    evp = evp.reset_index()
    rich = rich.merge(evp, on=['user','course'], how='left')

    event_cols = [c for c in rich.columns if c.startswith('ev_')]
    rich[event_cols] = rich[event_cols].fillna(0).astype(float)

    # User unique objects.
    op = obj.pivot_table(
        index=['user','course'],
        columns='family',
        values='n_unique',
        aggfunc='sum',
        fill_value=0
    )
    op.columns = ['obj_' + str(c) + '_unique' for c in op.columns]
    op = op.reset_index()
    rich = rich.merge(op, on=['user','course'], how='left')

    for c in [c for c in rich.columns if c.startswith('obj_')]:
        rich[c] = rich[c].fillna(0).astype(float)

    # Structural denominator preferred; observed denominator is fallback.
    observed_den = cob.copy()
    observed_den['denominator_source'] = 'observed_fallback'

    if not structural_inventory.empty:
        struct_den = (
            structural_inventory
            .groupby(['course','family'])
            .size()
            .rename('n_course_objects')
            .reset_index()
        )
        struct_den['course'] = struct_den['course'].astype(str)
        struct_den['denominator_source'] = 'course_modules'
        den = pd.concat([struct_den, observed_den], ignore_index=True)
        den['_priority'] = den['denominator_source'].map(
            {'course_modules':0, 'observed_fallback':1}
        )
        den = (
            den.sort_values(['course','family','_priority'])
               .drop_duplicates(['course','family'], keep='first')
               .drop(columns='_priority')
        )
    else:
        den = observed_den

    dp = den.pivot_table(
        index='course',
        columns='family',
        values='n_course_objects',
        aggfunc='max',
        fill_value=0
    )
    dp.columns = ['course_' + str(c) + '_objects' for c in dp.columns]
    dp = dp.reset_index()
    rich = rich.merge(dp, on='course', how='left')

    # Family active days.
    fdp = fdays.pivot_table(
        index=['user','course'],
        columns='family',
        values='active_days',
        aggfunc='sum',
        fill_value=0
    )
    fdp.columns = [str(c) + '_active_days' for c in fdp.columns]
    fdp = fdp.reset_index()
    rich = rich.merge(fdp, on=['user','course'], how='left')

    # Broad active days.
    bdp = bdays.pivot_table(
        index=['user','course'],
        columns='broad_group',
        values='active_days',
        aggfunc='sum',
        fill_value=0
    )
    bdp.columns = [str(c) + '_active_days' for c in bdp.columns]
    bdp = bdp.reset_index()
    rich = rich.merge(bdp, on=['user','course'], how='left')

    # Generic family feature family:
    # count, share, rate, active days, active-day ratio,
    # unique objects, coverage, revisit.
    family_feature_rows = []

    for family in GENERIC_FAMILIES:
        sub = fam[fam['family'] == family][
            ['user','course','event_count','first_t','last_t',
             'early_count','middle_count','late_count']
        ].copy()
        if sub.empty:
            continue

        pref = family + '_'
        sub = sub.rename(columns={
            'event_count': pref + 'event_count',
            'first_t': pref + 'first_t',
            'last_t': pref + 'last_t',
            'early_count': pref + 'early_count',
            'middle_count': pref + 'middle_count',
            'late_count': pref + 'late_count'
        })
        rich = rich.merge(sub, on=['user','course'], how='left')

        cnt = rich.get(pref+'event_count', pd.Series(0.0,index=rich.index)).fillna(0)
        unique = rich.get(
            'obj_'+family+'_unique',
            pd.Series(0.0,index=rich.index)
        ).fillna(0)
        avail = rich.get(
            'course_'+family+'_objects',
            pd.Series(0.0,index=rich.index)
        ).fillna(0)
        fad = rich.get(
            family+'_active_days',
            pd.Series(0.0,index=rich.index)
        ).fillna(0)

        rich[pref+'event_share_pct'] = safe_div(
            cnt, rich['total_behavior_events'], 100
        )
        rich[pref+'event_rate_week'] = safe_div(cnt, course_weeks)
        rich[pref+'active_day_ratio'] = safe_div(fad, rich['active_days'])
        rich[pref+'coverage_pct'] = safe_div(unique, avail, 100)
        rich[pref+'revisit_ratio'] = safe_div(cnt, unique)

        family_feature_rows.append(family)

    # Broad groups: intensity + temporal distribution.
    for group in BROAD_GROUPS:
        sub = broad[broad['broad_group'] == group][
            ['user','course','event_count','first_t','last_t',
             'early_count','middle_count','late_count']
        ].copy()

        pref = group + '_'
        if not sub.empty:
            sub = sub.rename(columns={
                'event_count': pref+'event_count',
                'first_t': pref+'first_t',
                'last_t': pref+'last_t',
                'early_count': pref+'early_count',
                'middle_count': pref+'middle_count',
                'late_count': pref+'late_count'
            })
            rich = rich.merge(sub, on=['user','course'], how='left')

        # A broad group may be absent completely. Create its columns explicitly
        # so downstream support aggregation never fails with KeyError.
        required_broad = {
            pref+'event_count': 0.0,
            pref+'first_t': np.nan,
            pref+'last_t': np.nan,
            pref+'early_count': 0.0,
            pref+'middle_count': 0.0,
            pref+'late_count': 0.0,
        }
        for col_name, default_value in required_broad.items():
            if col_name not in rich.columns:
                rich[col_name] = default_value

        cnt = rich[pref+'event_count'].fillna(0)
        early = rich.get(pref+'early_count', pd.Series(0.0,index=rich.index)).fillna(0)
        middle = rich.get(pref+'middle_count', pd.Series(0.0,index=rich.index)).fillna(0)
        late = rich.get(pref+'late_count', pd.Series(0.0,index=rich.index)).fillna(0)
        gad = rich.get(pref+'active_days', pd.Series(0.0,index=rich.index)).fillna(0)

        rich[pref+'share_pct'] = safe_div(
            cnt, rich['total_behavior_events'], 100
        )
        rich[pref+'rate_week'] = safe_div(cnt, course_weeks)
        rich[pref+'active_day_ratio'] = safe_div(gad, rich['active_days'])
        rich[pref+'early_share_pct'] = safe_div(early, cnt, 100)
        rich[pref+'middle_share_pct'] = safe_div(middle, cnt, 100)
        rich[pref+'late_share_pct'] = safe_div(late, cnt, 100)

        ft = rich.get(pref+'first_t', pd.Series(np.nan,index=rich.index))
        lt = rich.get(pref+'last_t', pd.Series(np.nan,index=rich.index))
        rich[pref+'first_pct'] = (
            (ft - rich['course_start']) / duration_sec * 100
        ).clip(0,100)
        rich[pref+'last_pct'] = (
            (lt - rich['course_start']) / duration_sec * 100
        ).clip(0,100)
        rich[pref+'span_pct'] = (
            (lt - ft) / duration_sec * 100
        ).clip(lower=0, upper=100)

    # Direct action-level behavioral measures.
    def evc(name):
        return rich.get(
            'ev_'+name,
            pd.Series(0.0, index=rich.index)
        ).fillna(0).astype(float)

    rich['overview_view_count'] = evc('course_view') + evc('course_section_view')

    content_event_names = [
        'resource_view','url_view','page_view','book_view','lesson_view',
        'folder_view','glossary_view','wiki_view','scorm_view','imscp_view'
    ]
    rich['content_view_count'] = sum(
        (evc(x) for x in content_event_names),
        start=pd.Series(0.0,index=rich.index)
    )

    rich['assignment_view_count'] = (
        evc('assign_view')
        + evc('assign_submission_form_view')
        + evc('assign_submit_confirm_view')
    )

    # Do not sum submit and submit-for-grading blindly as independent
    # submissions. Keep both raw counts plus a conservative max proxy.
    rich['assignment_submit_event_count'] = evc('assign_submit')
    rich['assignment_submit_for_grading_count'] = evc('assign_submit_for_grading')
    rich['assignment_submit_conservative_count'] = np.maximum(
        rich['assignment_submit_event_count'],
        rich['assignment_submit_for_grading_count']
    )

    rich['quiz_view_count'] = evc('quiz_view') + evc('quiz_summary_view')
    rich['quiz_attempt_start_count'] = evc('quiz_attempt_start')
    rich['quiz_attempt_continue_count'] = evc('quiz_attempt_continue')
    rich['quiz_review_count'] = evc('quiz_review')
    rich['quiz_close_attempt_count'] = evc('quiz_close_attempt')

    rich['forum_read_count'] = (
        evc('forum_view')
        + evc('forum_index_view')
        + evc('forum_discussion_view')
    )
    rich['forum_post_count'] = (
        evc('forum_post_add')
        + evc('forum_discussion_add')
    )
    rich['forum_post_update_count'] = evc('forum_post_update')

    # Higher-level ratios: names remain operational, not theory labels.
    assign_avail = rich.get(
        'course_assign_objects',
        pd.Series(0.0,index=rich.index)
    ).fillna(0)
    quiz_avail = rich.get(
        'course_quiz_objects',
        pd.Series(0.0,index=rich.index)
    ).fillna(0)
    forum_avail = rich.get(
        'course_forum_objects',
        pd.Series(0.0,index=rich.index)
    ).fillna(0)

    rich['assignment_submission_rate_per_available'] = safe_div(
        rich['assignment_submit_conservative_count'], assign_avail
    )
    rich['quiz_attempt_rate_per_available'] = safe_div(
        rich['quiz_attempt_start_count'], quiz_avail
    )
    rich['quiz_review_per_attempt'] = safe_div(
        rich['quiz_review_count'], rich['quiz_attempt_start_count']
    )
    rich['quiz_continue_per_attempt'] = safe_div(
        rich['quiz_attempt_continue_count'], rich['quiz_attempt_start_count']
    )
    rich['quiz_close_per_attempt'] = safe_div(
        rich['quiz_close_attempt_count'], rich['quiz_attempt_start_count']
    )
    rich['forum_post_per_forum'] = safe_div(
        rich['forum_post_count'], forum_avail
    )
    rich['forum_post_read_ratio'] = safe_div(
        rich['forum_post_count'],
        rich['forum_read_count'] + rich['forum_post_count']
    )
    rich['forum_read_rate_week'] = safe_div(
        rich['forum_read_count'], course_weeks
    )
    rich['forum_post_rate_week'] = safe_div(
        rich['forum_post_count'], course_weeks
    )

    # Cross-family relative behavior.
    zero = pd.Series(0.0, index=rich.index)
    content_n = rich.get('content_event_count', zero)
    assignment_n = rich.get('assignment_event_count', zero)
    quiz_n = rich.get('quiz_event_count', zero)
    forum_n = rich.get('forum_event_count', zero)
    overview_n = rich.get('overview_event_count', zero)

    rich['content_to_assessment_event_ratio'] = safe_div(
        content_n, assignment_n + quiz_n
    )
    rich['assessment_to_content_event_ratio'] = safe_div(
        assignment_n + quiz_n, content_n
    )
    rich['forum_to_content_event_ratio'] = safe_div(
        forum_n, content_n
    )
    rich['overview_to_content_event_ratio'] = safe_div(
        overview_n, content_n
    )

    # Number of module families actually used by the learner.
    family_count_cols = [
        f + '_event_count'
        for f in GENERIC_FAMILIES
        if f + '_event_count' in rich.columns
    ]
    if family_count_cols:
        rich['behavioral_family_breadth'] = (
            rich[family_count_cols].fillna(0).gt(0).sum(axis=1)
        ).astype(float)
    else:
        rich['behavioral_family_breadth'] = 0.0

    # Course behavioral support is a course-design attribute used only
    # for one-course selection, not as a clustering feature.
    support_source_cols = [
        'overview_event_count',
        'content_event_count',
        'assignment_event_count',
        'quiz_event_count',
        'forum_event_count',
    ]
    for col_name in support_source_cols:
        if col_name not in rich.columns:
            rich[col_name] = 0.0

    course_support = (
        rich.groupby('course', as_index=False)
        .agg(
            support_overview=('overview_event_count', lambda s: int((s.fillna(0)>0).any())),
            support_content=('content_event_count', lambda s: int((s.fillna(0)>0).any())),
            support_assignment=('assignment_event_count', lambda s: int((s.fillna(0)>0).any())),
            support_quiz=('quiz_event_count', lambda s: int((s.fillna(0)>0).any())),
            support_forum=('forum_event_count', lambda s: int((s.fillna(0)>0).any())),
        )
    )
    course_support['behavioral_support_n'] = course_support[
        ['support_overview','support_content','support_assignment',
         'support_quiz','support_forum']
    ].sum(axis=1)
    rich = rich.merge(course_support, on='course', how='left')

    # Stable identifiers.
    rich['user'] = rich['user'].astype(str)
    rich['course'] = rich['course'].astype(str)
    rich['unit_id'] = rich['user'] + '|' + rich['course']

    # Replace +/-inf in numerical feature bank; preserve NaN timing when
    # a behavior never occurred.
    num_cols = rich.select_dtypes(include=[np.number]).columns
    rich[num_cols] = rich[num_cols].replace([np.inf,-np.inf], np.nan)

    save_csv6(rich, feature_bank_path, index=False)
    # Feature dictionary / provenance.
    raw_identifier = {
        'user','course','unit_id','first_t','last_t','course_start',
        'course_end','duration_days','window_source'
    }
    audit_only = {
        'mapped_event_count','support_overview','support_content',
        'support_assignment','support_quiz','support_forum',
        'behavioral_support_n'
    }

    rows = []
    for c in rich.columns:
        if c in raw_identifier:
            role = 'identifier_or_audit'
        elif c in audit_only or c.startswith('course_'):
            role = 'audit_or_denominator'
        elif c.endswith('_first_t') or c.endswith('_last_t'):
            role = 'audit_timestamp'
        elif c.startswith('ev_'):
            role = 'atomic_event_evidence'
        else:
            role = 'engineered_behavior_candidate'

        if 'revision' in c.lower():
            warning = 'do_not_equate_with_Graf_answer_revision_without_answer-history'
        else:
            warning = ''

        rows.append({
            'feature': c,
            'role': role,
            'warning': warning,
            'definition': 'Derived from raw Moodle event taxonomy; exact formula is in notebook code.'
        })

    fd = pd.DataFrame(rows)
    save_csv6(fd, feature_dictionary_path, index=False)
    audit = {
        'rows_user_course': int(len(rich)),
        'unique_users': int(rich['user'].nunique()),
        'unique_courses': int(rich['course'].nunique()),
        'columns_total': int(rich.shape[1]),
        'numeric_columns': int(len(rich.select_dtypes(include=[np.number]).columns)),
        'engineered_behavior_candidates': int((fd['role']=='engineered_behavior_candidate').sum()),
        'atomic_event_evidence_columns': int((fd['role']=='atomic_event_evidence').sum()),
        'uses_outcome_grade_label': False,
        'uses_direct_dwell_duration': False,
        'structural_denominator_available': bool(len(structural_inventory))
    }
    feature_bank_audit_path.write_text(
        json.dumps(audit, indent=2),
        encoding='utf-8'
    )

    return rich, fd, audit

stage05_cached = (
    file_ready(feature_bank_path)
    and file_ready(feature_dictionary_path)
    and file_ready(feature_bank_audit_path)
)

if (not FORCE_REBUILD) and stage05_cached:
    print('[cached] Stage 05 rich numerical feature bank')
    rich = pd.read_csv(
        feature_bank_path,
        dtype={'user':str,'course':str,'unit_id':str},
        low_memory=False
    )
    feature_dictionary = pd.read_csv(feature_dictionary_path)
    feature_bank_audit = json.loads(
        feature_bank_audit_path.read_text(encoding='utf-8')
    )
else:
    rich, feature_dictionary, feature_bank_audit = build_feature_bank()

print(json.dumps(feature_bank_audit, indent=2))
print('Feature bank shape:', rich.shape)
preview(rich.head())

# ============================================================
# Stage 05b — merge sessionized dwell + recoverable Graf features
# ============================================================
graf_spec_path = OUT_DIR / '05b_graf_recoverable_feature_specification.csv'


if (not FORCE_REBUILD) and (file_ready(graf_spec_path) and file_ready(feature_bank_path) and file_ready(feature_dictionary_path)):
    print('[cached] Stage 05b Graf/sessionized feature enrichment')
    graf_spec = pd.read_csv(graf_spec_path)
    rich = pd.read_csv(
        feature_bank_path,
        dtype={'user':str,'course':str,'unit_id':str},
        low_memory=False
    )
    feature_dictionary = pd.read_csv(feature_dictionary_path)
    feature_bank_audit = json.loads(
        feature_bank_audit_path.read_text(encoding='utf-8')
    ) if file_ready(feature_bank_audit_path) else {}
else:
    # A. Merge dwell metrics
    dwell = dwell_long.copy()
    metrics = [
        'sessionized_sec',
        'sessionized_min',
        'avg_valid_gap_sec',
        'valid_gap_n',
        'idle_excluded_n',
        'idle_excluded_share'
    ]

    for metric in metrics:
        p = dwell.pivot_table(
            index=['user','course'],
            columns='dwell_type',
            values=metric,
            aggfunc='first'
        )
        p.columns = [f'{dtype}__{metric}' for dtype in p.columns]
        p = p.reset_index()
        rich = rich.merge(p, on=['user','course'], how='left')

    dwell_metric_cols = [
        c for c in rich.columns
        if '__sessionized_' in c
        or '__avg_valid_gap_sec' in c
        or '__valid_gap_n' in c
        or '__idle_excluded_' in c
    ]
    if dwell_metric_cols:
        rich[dwell_metric_cols] = rich[dwell_metric_cols].fillna(0.0)

    duration_sec = (
        rich['course_end'] - rich['course_start']
    ).clip(lower=1)
    course_weeks = np.maximum(duration_sec / 604800, 1.0)

    def zero_series():
        return pd.Series(0.0, index=rich.index)

    def series_or_zero(name):
        if name in rich.columns:
            return pd.to_numeric(rich[name], errors='coerce').fillna(0.0)
        return zero_series()

    # Primary stay variables.
    rich['content_stay'] = series_or_zero(
        'content_stay__sessionized_min'
    )
    rich['forum_stay'] = safe_div(
        series_or_zero('forum_stay__sessionized_min'),
        course_weeks
    )
    rich['selfass_stay_proxy'] = series_or_zero(
        'selfass_stay_proxy__sessionized_min'
    )
    rich['exercise_stay_proxy'] = series_or_zero(
        'exercise_stay_proxy__sessionized_min'
    )
    rich['navigation_overview_stay'] = series_or_zero(
        'navigation_overview_stay__sessionized_min'
    )
    rich['outline_stay_proxy'] = series_or_zero(
        'outline_stay_proxy__sessionized_min'
    )
    rich['quiz_stay_results_proxy'] = series_or_zero(
        'quiz_stay_results_proxy__avg_valid_gap_sec'
    )

    # B. Event-specific unique objects.
    if not event_object_unique.empty:
        evobjp = event_object_unique.pivot_table(
            index=['user','course'],
            columns='event_class',
            values='n_unique_objects',
            aggfunc='max',
            fill_value=0
        )
        evobjp.columns = [
            'evobj_' + str(c) + '_unique'
            for c in evobjp.columns
        ]
        evobjp = evobjp.reset_index()
        rich = rich.merge(evobjp, on=['user','course'], how='left')

    evobj_cols = [c for c in rich.columns if c.startswith('evobj_')]
    if evobj_cols:
        rich[evobj_cols] = rich[evobj_cols].fillna(0.0)

    # C. Recoverable Graf-aligned visit/action indicators.
    content_den_cols = [
        f'course_{m}_objects'
        for m in CONTENT_MODULES
        if f'course_{m}_objects' in rich.columns
    ]
    if content_den_cols:
        content_available = rich[content_den_cols].fillna(0).sum(axis=1)
    else:
        content_available = series_or_zero('course_content_objects')

    rich['content_visit'] = safe_div(
        series_or_zero('obj_content_unique'),
        content_available,
        100
    )

    rich['selfass_visit_proxy'] = safe_div(
        series_or_zero('evobj_quiz_attempt_start_unique'),
        series_or_zero('course_quiz_objects'),
        100
    )

    assign_submit_unique = np.maximum(
        series_or_zero('evobj_assign_submit_unique').to_numpy(),
        series_or_zero(
            'evobj_assign_submit_for_grading_unique'
        ).to_numpy()
    )
    rich['exercise_visit_proxy'] = safe_div(
        assign_submit_unique,
        series_or_zero('course_assign_objects'),
        100
    )

    # Graf forum_visit/forum_post are per week.
    rich['forum_visit'] = series_or_zero('forum_read_rate_week')
    rich['forum_post'] = series_or_zero('forum_post_rate_week')

    # Graf navigation_overview_visit is relative to visited learning objects.
    overview_n = series_or_zero('overview_view_count')
    visited_learning_events = (
        overview_n
        + series_or_zero('content_view_count')
        + series_or_zero('assignment_view_count')
        + series_or_zero('quiz_view_count')
        + series_or_zero('forum_read_count')
    )
    rich['navigation_overview_visit'] = safe_div(
        overview_n,
        visited_learning_events,
        100
    )

    # Course-section view is retained only as an outline proxy.
    rich['outline_visit_proxy'] = safe_div(
        series_or_zero('ev_course_section_view'),
        np.maximum(visited_learning_events, 1),
        100
    )

    # D. Graf feature specification.
    graf_rows = [
        ['content_visit','GRAF_ALIGNED_RECONSTRUCTION',
         'unique visited content objects / available content objects * 100',
         'percent','75%, 100%','10 min',
         'Content family reconstructed from Moodle content modules.'],
        ['content_stay','GRAF_ALIGNED_SESSIONIZED_PROXY',
         'sum valid consecutive-event gaps for content; gaps >10 min excluded',
         'minutes per learner-course','50%, 75% of expected time','10 min',
         'Expected-time denominator is deferred; not direct Moodle duration.'],
        ['selfass_visit_proxy','GRAF_PROXY',
         'unique quiz objects actually attempted / available quiz objects * 100',
         'percent','25%, 75%','10 min',
         'Quiz is an operational self-assessment proxy.'],
        ['selfass_stay_proxy','GRAF_ALIGNED_SESSIONIZED_PROXY',
         'sum valid quiz-interaction gaps, excluding explicit quiz-review result dwell',
         'minutes per learner-course','50%, 75% of expected time','10 min',
         'Sessionized proxy.'],
        ['exercise_visit_proxy','GRAF_PROXY',
         'unique assignments actually submitted / available assignments * 100',
         'percent','25%, 75%','20 min',
         'Assignment is used as operational exercise evidence.'],
        ['exercise_stay_proxy','GRAF_ALIGNED_SESSIONIZED_PROXY',
         'sum valid assignment/exercise gaps; gaps >20 min excluded',
         'minutes per learner-course','50%, 75% of expected time','20 min',
         'Sessionized proxy.'],
        ['forum_visit','GRAF_DIRECT_BEHAVIOR',
         'forum read/view events per course week',
         'visits per week','7, 50 visits/week','10 min',
         'Uses observable forum reads.'],
        ['forum_stay','GRAF_ALIGNED_SESSIONIZED_PROXY',
         'sessionized forum dwell minutes / course weeks',
         'minutes per week','5, 30 min/week','10 min',
         'Unit matches Graf forum_stay thresholds.'],
        ['forum_post','GRAF_DIRECT_BEHAVIOR',
         'actual add-post + add-discussion events per course week',
         'posts per week','1, 10 posts/week','n/a',
         'Actual forum contribution events.'],
        ['navigation_overview_visit','GRAF_ALIGNED_RECONSTRUCTION',
         'overview/course views / major visited learning events * 100',
         'percent','10%, 20%','3 min',
         'Course/overview operationalization.'],
        ['navigation_overview_stay','GRAF_ALIGNED_SESSIONIZED_PROXY',
         'sessionized dwell on course-overview events; gaps >3 min excluded',
         'minutes per learner-course','50%, 75% of expected time','3 min',
         'Expected-time denominator deferred.'],
        ['outline_visit_proxy','GRAF_PROXY',
         'course-section views / major visited learning events * 100',
         'percent','75%, 150% of available outlines','3 min',
         'Course section is only an outline proxy.'],
        ['outline_stay_proxy','GRAF_ALIGNED_SESSIONIZED_PROXY',
         'sessionized dwell on course-section events; gaps >3 min excluded',
         'minutes per learner-course','50%, 75% of expected time','3 min',
         'Course section is an outline proxy.'],
        ['quiz_stay_results_proxy','GRAF_ALIGNED_SESSIONIZED_PROXY',
         'average valid dwell seconds on quiz-review events',
         'seconds per review event','30 sec, 60 sec','10 min',
         'quiz_review is used as result/review-page evidence.'],
        ['quiz_revisions','NOT_RECOVERABLE',
         'not generated',
         'n/a','20%, 50%','n/a',
         'continue-attempt/review events are not true answer revisions before submission.'],
        ['example_visit/example_stay','NOT_RECOVERABLE',
         'not generated',
         'n/a','visit 25%,75%; stay 50%,75% expected time','20 min',
         'Raw dump lacks explicit example metadata.'],
    ]

    graf_spec = pd.DataFrame(
        graf_rows,
        columns=[
            'feature','status','numeric_definition','unit',
            'graf_occurrence_thresholds','critical_idle_threshold','note'
        ]
    )
    save_csv6(graf_spec, graf_spec_path, index=False)
    # E. Update feature dictionary.
    new_defs = {
        'content_visit':'Graf-aligned content coverage reconstruction.',
        'content_stay':'Sessionized content dwell proxy.',
        'selfass_visit_proxy':'Unique attempted quiz coverage proxy.',
        'selfass_stay_proxy':'Sessionized quiz/self-assessment dwell proxy.',
        'exercise_visit_proxy':'Unique submitted assignment coverage proxy.',
        'exercise_stay_proxy':'Sessionized assignment/exercise dwell proxy.',
        'forum_visit':'Forum read/view rate per week.',
        'forum_stay':'Sessionized forum dwell minutes per week.',
        'forum_post':'Actual forum posts/discussions per week.',
        'navigation_overview_visit':'Overview visit share reconstruction.',
        'navigation_overview_stay':'Sessionized overview dwell proxy.',
        'outline_visit_proxy':'Course-section visit proxy.',
        'outline_stay_proxy':'Sessionized course-section dwell proxy.',
        'quiz_stay_results_proxy':'Average sessionized seconds on quiz-review events.'
    }

    fd = feature_dictionary.copy()
    existing = set(fd['feature'].astype(str))
    rows_to_add = []

    for f, definition in new_defs.items():
        if f not in existing:
            rows_to_add.append({
                'feature': f,
                'role': 'graf_aligned_behavior_candidate',
                'warning': (
                    'SESSIONIZED_PROXY_NOT_DIRECT_DURATION'
                    if 'stay' in f else ''
                ),
                'definition': definition
            })

    for c in rich.columns:
        if (
            '__sessionized_' in c
            or '__avg_valid_gap_sec' in c
            or '__valid_gap_n' in c
            or '__idle_excluded_' in c
            or c.startswith('evobj_')
        ) and c not in existing:
            rows_to_add.append({
                'feature': c,
                'role': (
                    'sessionized_dwell_audit'
                    if '__' in c else
                    'atomic_event_object_evidence'
                ),
                'warning': '',
                'definition': (
                    'Technical evidence generated for sessionized '
                    'behavioral feature construction.'
                )
            })

    if rows_to_add:
        fd = pd.concat([fd, pd.DataFrame(rows_to_add)], ignore_index=True)

    feature_dictionary = fd.drop_duplicates('feature', keep='last')
    save_csv6(feature_dictionary, feature_dictionary_path, index=False)
    # Persist enriched numeric feature bank.
    save_csv6(rich, feature_bank_path, index=False)
    feature_bank_audit.update({
        'columns_total_after_dwell_enrichment': int(rich.shape[1]),
        'sessionized_dwell_enabled': True,
        'dwell_policy': DWELL_POLICY,
        'uses_direct_moodle_duration': False,
        'graf_recoverable_features_added': sorted(new_defs)
    })
    feature_bank_audit_path.write_text(
        json.dumps(feature_bank_audit, indent=2),
        encoding='utf-8'
    )

    print('Enriched feature-bank shape:', rich.shape)
    preview(graf_spec)


# ============================================================
# Stage 05c — navigation/sequence + modality feature engineering
# ============================================================
navigation_feature_path = OUT_DIR / '05c_navigation_features.csv'
modality_feature_path = OUT_DIR / '05c_modality_features.csv'
semantic_structure_audit_path = OUT_DIR / '05c_semantic_structure_audit.json'



if (not FORCE_REBUILD) and (file_ready(semantic_structure_audit_path) and file_ready(feature_bank_path) and file_ready(feature_dictionary_path)):
    print('[cached] Stage 05c navigation/sequence + modality')
    rich = pd.read_csv(
        feature_bank_path,
        dtype={'user':str,'course':str,'unit_id':str},
        low_memory=False
    )
    feature_dictionary = pd.read_csv(feature_dictionary_path)
    navigation_features = (
        safe_read_csv(navigation_feature_path, dtype={'user':str,'course':str}, empty_columns=['user','course'])
        if file_ready(navigation_feature_path) else pd.DataFrame()
    )
    modality_features = (
        safe_read_csv(modality_feature_path, dtype={'user':str,'course':str}, empty_columns=['user','course'])
        if file_ready(modality_feature_path) else pd.DataFrame()
    )
    semantic_structure_audit = json.loads(
        semantic_structure_audit_path.read_text(encoding='utf-8')
    )
    feature_bank_audit = json.loads(
        feature_bank_audit_path.read_text(encoding='utf-8')
    ) if file_ready(feature_bank_audit_path) else {}
else:
    def build_navigation_and_modality_features():
        con = sqlite3.connect(dwell_db)

        # ---------- Navigation order ----------
        nav = navigation_order.copy()
        if not nav.empty:
            nav['course'] = nav['course'].astype(str)
            nav['cmid'] = nav['cmid'].astype(str)
            con.execute('DROP TABLE IF EXISTS nav_order')
            con.execute('''CREATE TABLE nav_order(
                course TEXT, cmid TEXT, section_number INTEGER,
                global_position INTEGER, order_source TEXT,
                PRIMARY KEY(course,cmid)
            )''')
            con.executemany(
                'INSERT OR REPLACE INTO nav_order(course,cmid,section_number,global_position,order_source) VALUES (?,?,?,?,?)',
                [
                    (str(r.course),str(r.cmid),safe_int(r.section_number,0),safe_int(r.global_position,0),str(r.order_source))
                    for r in nav.itertuples(index=False)
                ]
            )
            con.commit()

            nav_sql = '''
            WITH obj AS (
                SELECT r.id,r.user,r.course,r.t,r.cmid,
                       n.section_number,n.global_position,n.order_source
                FROM raw_event r
                JOIN nav_order n ON r.course=n.course AND r.cmid=n.cmid
                WHERE r.is_mapped=1 AND r.cmid IS NOT NULL
            ), tr AS (
                SELECT *,
                       LAG(global_position) OVER(PARTITION BY user,course ORDER BY t,id) AS prev_pos,
                       LAG(section_number) OVER(PARTITION BY user,course ORDER BY t,id) AS prev_section
                FROM obj
            )
            SELECT user,course,
                   SUM(CASE WHEN prev_pos IS NOT NULL THEN 1 ELSE 0 END) AS object_transition_n,
                   SUM(CASE WHEN prev_pos IS NOT NULL AND global_position-prev_pos=1 THEN 1 ELSE 0 END) AS sequential_next_n,
                   SUM(CASE WHEN prev_pos IS NOT NULL AND global_position-prev_pos>1 THEN 1 ELSE 0 END) AS forward_skip_n,
                   SUM(CASE WHEN prev_pos IS NOT NULL AND global_position-prev_pos<0 THEN 1 ELSE 0 END) AS backward_transition_n,
                   SUM(CASE WHEN prev_pos IS NOT NULL AND global_position-prev_pos=0 THEN 1 ELSE 0 END) AS same_object_repeat_n,
                   SUM(CASE WHEN prev_pos IS NOT NULL AND section_number=prev_section THEN 1 ELSE 0 END) AS same_section_transition_n,
                   SUM(CASE WHEN prev_pos IS NOT NULL AND section_number<>prev_section THEN 1 ELSE 0 END) AS cross_section_transition_n,
                   AVG(CASE WHEN prev_pos IS NOT NULL THEN ABS(global_position-prev_pos) END) AS mean_abs_position_jump,
                   MAX(CASE WHEN order_source='course_sections_sequence' THEN 1 ELSE 0 END) AS has_canonical_order
            FROM tr
            GROUP BY user,course
            '''
            navf = pd.read_sql_query(nav_sql, con, dtype={'user':str,'course':str})
        else:
            navf = pd.DataFrame(columns=['user','course'])

        # Overview re-entry / bridge behavior from mapped-event sequence.
        overview_sql = '''
        WITH m AS (
            SELECT id,user,course,t,event_class,
                   CASE WHEN event_class IN ('course_view','course_section_view') THEN 1 ELSE 0 END AS is_overview
            FROM raw_event
            WHERE is_mapped=1
        ), x AS (
            SELECT *,
                   LAG(is_overview) OVER(PARTITION BY user,course ORDER BY t,id) AS prev_overview,
                   LEAD(is_overview) OVER(PARTITION BY user,course ORDER BY t,id) AS next_overview
            FROM m
        )
        SELECT user,course,
               SUM(CASE WHEN is_overview=1 THEN 1 ELSE 0 END) AS overview_sequence_events,
               SUM(CASE WHEN is_overview=1 AND COALESCE(prev_overview,1)=0 THEN 1 ELSE 0 END) AS return_to_overview_n,
               SUM(CASE WHEN is_overview=1 AND COALESCE(prev_overview,1)=0 AND COALESCE(next_overview,1)=0 THEN 1 ELSE 0 END) AS overview_bridge_n,
               COUNT(*) AS mapped_sequence_events
        FROM x GROUP BY user,course
        '''
        overview_seq = pd.read_sql_query(overview_sql, con, dtype={'user':str,'course':str})

        if navf.empty:
            navf = overview_seq
        else:
            navf = navf.merge(overview_seq, on=['user','course'], how='outer')

        # Rates; unavailable structural order remains distinguishable via has_canonical_order / course support.
        if not navf.empty:
            navf = navf.fillna(0)

            # All navigation numerators/denominators must be 1-D Series aligned
            # to navf.index. Some SQL-derived columns may be absent entirely;
            # in that case use an all-zero Series rather than scalar 0.
            nav_zero = pd.Series(0.0, index=navf.index, dtype=float)

            def nav_col(name):
                if name in navf.columns:
                    return pd.to_numeric(
                        navf[name], errors='coerce'
                    ).fillna(0.0).astype(float)
                return nav_zero.copy()

            n = nav_col('object_transition_n')
            navf['sequential_transition_ratio'] = safe_div(
                nav_col('sequential_next_n'), n
            )
            navf['forward_skip_ratio'] = safe_div(
                nav_col('forward_skip_n'), n
            )
            navf['backward_navigation_ratio'] = safe_div(
                nav_col('backward_transition_n'), n
            )
            navf['same_object_repeat_ratio'] = safe_div(
                nav_col('same_object_repeat_n'), n
            )
            navf['same_section_transition_ratio'] = safe_div(
                nav_col('same_section_transition_n'), n
            )
            navf['cross_section_jump_ratio'] = safe_div(
                nav_col('cross_section_transition_n'), n
            )

            mapped_n = nav_col('mapped_sequence_events')
            navf['return_to_overview_rate'] = safe_div(
                nav_col('return_to_overview_n'), mapped_n
            )
            navf['overview_bridge_rate'] = safe_div(
                nav_col('overview_bridge_n'), mapped_n
            )

            save_csv6(navf, navigation_feature_path, index=False)
        # ---------- Modality features ----------
        modf = pd.DataFrame(columns=['user','course'])
        if not modality_inventory.empty:
            mi = modality_inventory[['course','cmid','modality']].drop_duplicates().copy()
            mi['course'] = mi['course'].astype(str)
            mi['cmid'] = mi['cmid'].astype(str)

            con.execute('DROP TABLE IF EXISTS cm_modality')
            con.execute('''CREATE TABLE cm_modality(
                course TEXT, cmid TEXT, modality TEXT,
                PRIMARY KEY(course,cmid,modality)
            )''')
            con.executemany(
                'INSERT OR IGNORE INTO cm_modality(course,cmid,modality) VALUES (?,?,?)',
                [(r.course,r.cmid,r.modality) for r in mi.itertuples(index=False)]
            )
            con.commit()

            modality_sql = '''
            WITH ordered AS (
                SELECT *, LEAD(t) OVER(PARTITION BY user,course ORDER BY t,id) AS next_raw_t
                FROM raw_event
            )
            SELECT r.user,r.course,m.modality,
                   COUNT(*) AS visit_count,
                   COUNT(DISTINCT r.cmid) AS unique_objects,
                   SUM(CASE WHEN r.broad_group='content' AND r.next_raw_t>r.t AND (r.next_raw_t-r.t)<=600
                            THEN (r.next_raw_t-r.t) ELSE 0 END) AS stay_sec
            FROM ordered r
            JOIN cm_modality m ON r.course=m.course AND r.cmid=m.cmid
            WHERE r.is_mapped=1 AND r.cmid IS NOT NULL
            GROUP BY r.user,r.course,m.modality
            '''
            mod_long = pd.read_sql_query(modality_sql, con, dtype={'user':str,'course':str})

            denom = mi.groupby(['course','modality'])['cmid'].nunique().rename('available_objects').reset_index()
            if not mod_long.empty:
                mod_long = mod_long.merge(denom, on=['course','modality'], how='left')
                mod_long['coverage_pct'] = safe_div(mod_long['unique_objects'], mod_long['available_objects'], 100)
                mod_long['stay_min'] = mod_long['stay_sec'] / 60.0

                parts = []
                for metric in ['visit_count','unique_objects','coverage_pct','stay_min']:
                    p = mod_long.pivot_table(index=['user','course'], columns='modality', values=metric, aggfunc='sum', fill_value=0)
                    p.columns = [f'{c}_material_{metric}' for c in p.columns]
                    parts.append(p)
                modf = pd.concat(parts, axis=1).reset_index()
                save_csv6(modf, modality_feature_path, index=False)
        con.close()

        # Merge into rich bank.
        out = rich.copy()
        if not navf.empty:
            out = out.merge(navf, on=['user','course'], how='left')
        if not modf.empty:
            out = out.merge(modf, on=['user','course'], how='left')

        # Structural course-level capability columns.
        if not navigation_order.empty:
            nav_course = navigation_order.groupby('course').agg(
                course_navigation_object_count=('cmid','nunique'),
                course_canonical_navigation_order=('order_source', lambda s: int((s=='course_sections_sequence').any())),
                course_section_count=('section_number','nunique')
            ).reset_index()
            out = out.merge(nav_course, on='course', how='left')
        else:
            out['course_navigation_object_count'] = 0
            out['course_canonical_navigation_order'] = 0
            out['course_section_count'] = 0

        if not modality_inventory.empty:
            mc = modality_inventory.groupby(['course','modality'])['cmid'].nunique().unstack(fill_value=0)
            mc.columns = [f'course_{c}_modality_objects' for c in mc.columns]
            mc = mc.reset_index()
            out = out.merge(mc, on='course', how='left')

        # Fill newly engineered numeric metrics, but keep capability columns explicit.
        new_numeric = [
            c for c in out.columns
            if c not in rich.columns and pd.api.types.is_numeric_dtype(out[c])
        ]
        if new_numeric:
            out[new_numeric] = out[new_numeric].fillna(0.0)

        # Add dictionary roles.
        fd = feature_dictionary.copy()
        existing = set(fd['feature'].astype(str))
        add = []
        nav_candidates = {
            'sequential_transition_ratio','forward_skip_ratio','backward_navigation_ratio',
            'same_object_repeat_ratio','same_section_transition_ratio','cross_section_jump_ratio',
            'return_to_overview_rate','overview_bridge_rate','mean_abs_position_jump'
        }
        for f in nav_candidates:
            if f in out.columns and f not in existing:
                add.append({
                    'feature':f,'role':'semantic_navigation_candidate','warning':'',
                    'definition':'Structural navigation/sequence feature derived from course-module order and learner event sequence.'
                })
        for f in [c for c in out.columns if re.match(r'^(text|image|audio|video|presentation)_material_', c)]:
            if f not in existing:
                add.append({
                    'feature':f,'role':'modality_behavior_candidate','warning':'',
                    'definition':'Content-modality behavior supported by explicit MIME/extension/HTML structural evidence.'
                })
        for f in [c for c in out.columns if c.startswith('course_') and c not in existing]:
            add.append({
                'feature':f,'role':'audit_or_denominator','warning':'',
                'definition':'Course structural/modality capability used for semantic observability; not a learner clustering feature.'
            })
        if add:
            fd = pd.concat([fd,pd.DataFrame(add)], ignore_index=True).drop_duplicates('feature', keep='last')

        audit = {
            'navigation_features_generated': [c for c in nav_candidates if c in out.columns],
            'modality_features_generated': [c for c in out.columns if re.match(r'^(text|image|audio|video|presentation)_material_', c)],
            'canonical_navigation_courses': int(
                navigation_order.loc[navigation_order['order_source']=='course_sections_sequence','course'].nunique()
            ) if len(navigation_order) else 0,
            'courses_with_modality_metadata': int(modality_inventory['course'].nunique()) if len(modality_inventory) else 0,
        }
        semantic_structure_audit_path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
        return out, fd, navf, modf, audit

    rich, feature_dictionary, navigation_features, modality_features, semantic_structure_audit = build_navigation_and_modality_features()
    save_csv6(rich, feature_bank_path, index=False)
    save_csv6(feature_dictionary, feature_dictionary_path, index=False)
    feature_bank_audit.update({
        'columns_total_after_navigation_modality': int(rich.shape[1]),
        'semantic_structure_audit': semantic_structure_audit
    })
    feature_bank_audit_path.write_text(json.dumps(feature_bank_audit, indent=2), encoding='utf-8')

    print(json.dumps(semantic_structure_audit, indent=2))
    print('Feature-bank shape after V4 navigation/modality:', rich.shape)


# ============================================================
# Stage 06 — quality & redundancy audit
# ============================================================
quality_path = OUT_DIR / '06_feature_quality_audit.csv'
duplicate_path = OUT_DIR / '06_exact_duplicate_features.csv'


if (not FORCE_REBUILD) and file_ready(quality_path) and file_ready(duplicate_path):
    print('[cached] Stage 06 quality & redundancy audit')
    quality = pd.read_csv(quality_path)
    duplicate_df = safe_read_csv(duplicate_path, empty_columns=['feature','duplicate_of']) if file_ready(duplicate_path) else pd.DataFrame(
        columns=['feature','duplicate_of']
    )
    fd_role = feature_dictionary.set_index('feature')['role'].to_dict()
    candidate_cols = [
        col for col in rich.columns
        if fd_role.get(col) in {
            'engineered_behavior_candidate',
            'atomic_event_evidence',
            'graf_aligned_behavior_candidate',
            'semantic_navigation_candidate',
            'modality_behavior_candidate'
        }
        and pd.api.types.is_numeric_dtype(rich[col])
    ]
    print('Candidate numeric features:', len(candidate_cols))
    print('Exact duplicates:', len(duplicate_df))
    print(quality['candidate_status'].value_counts(dropna=False))
    preview(
        quality.sort_values(
            ['candidate_status','nonzero_share'],
            ascending=[True,False]
        ).head(80)
    )
else:
    fd_role = feature_dictionary.set_index('feature')['role'].to_dict()

    candidate_cols = [
        c for c in rich.columns
        if fd_role.get(c) in {
            'engineered_behavior_candidate',
            'atomic_event_evidence',
            'graf_aligned_behavior_candidate',
            'semantic_navigation_candidate',
            'modality_behavior_candidate'
        }
        and pd.api.types.is_numeric_dtype(rich[c])
    ]

    quality_rows = []
    for c in candidate_cols:
        s = pd.to_numeric(rich[c], errors='coerce')
        valid = s.dropna()
        quality_rows.append({
            'feature': c,
            'role': fd_role.get(c,''),
            'n': len(s),
            'missing_share': float(s.isna().mean()),
            'zero_share': float((s.fillna(0)==0).mean()),
            'nonzero_share': float((s.fillna(0)!=0).mean()),
            'nunique': int(valid.nunique()),
            'min': float(valid.min()) if len(valid) else np.nan,
            'q25': float(valid.quantile(.25)) if len(valid) else np.nan,
            'median': float(valid.median()) if len(valid) else np.nan,
            'q75': float(valid.quantile(.75)) if len(valid) else np.nan,
            'max': float(valid.max()) if len(valid) else np.nan,
            'candidate_status': (
                'DROP_CONSTANT'
                if valid.nunique() <= 1 else
                'REVIEW_SPARSE'
                if (s.fillna(0)!=0).mean() < .005 else
                'REVIEW_MISSING'
                if s.isna().mean() > .20 else
                'ELIGIBLE_FOR_LITERATURE_REVIEW'
            )
        })

    quality = pd.DataFrame(quality_rows)

    # Exact duplicate fingerprints only; near-correlation is deferred until
    # the literature-supported candidate set is known.
    fingerprints = {}
    dups = []
    for c in candidate_cols:
        s = pd.to_numeric(rich[c], errors='coerce').fillna(-9.87654321e99)
        h = hashlib.sha1(
            pd.util.hash_pandas_object(s, index=False).values.tobytes()
        ).hexdigest()
        if h in fingerprints and s.equals(
            pd.to_numeric(
                rich[fingerprints[h]], errors='coerce'
            ).fillna(-9.87654321e99)
        ):
            dups.append({
                'feature': c,
                'duplicate_of': fingerprints[h]
            })
        else:
            fingerprints[h] = c

    duplicate_df = pd.DataFrame(
        dups,
        columns=['feature','duplicate_of']
    )

    if len(duplicate_df):
        quality = quality.merge(
            duplicate_df.assign(exact_duplicate=True),
            on='feature',
            how='left'
        )
        quality['exact_duplicate'] = quality['exact_duplicate'].fillna(False)
    else:
        quality['duplicate_of'] = ''
        quality['exact_duplicate'] = False

    save_csv6(quality, quality_path, index=False)
    save_csv6(duplicate_df, duplicate_path, index=False)
    print('Candidate numeric features:', len(candidate_cols))
    print('Exact duplicates:', len(duplicate_df))
    print(quality['candidate_status'].value_counts(dropna=False))
    preview(quality.sort_values(
        ['candidate_status','nonzero_share'],
        ascending=[True,False]
    ).head(80))


# ============================================================
# Stage 07 — 20-semantic-group coverage matrix & course capability
# ============================================================
semantic_reference_path = OUT_DIR / '07_semantic_group_reference.csv'
semantic_feature_matrix_path = OUT_DIR / '07_semantic_feature_matrix.csv'
semantic_gap_path = OUT_DIR / '07_semantic_gap_report.csv'
course_capability_path = OUT_DIR / '07_course_semantic_capability.csv'


if (not FORCE_REBUILD) and (file_ready(semantic_reference_path) and file_ready(semantic_feature_matrix_path) and file_ready(semantic_gap_path) and file_ready(course_capability_path) and file_ready(feature_bank_path) and file_ready(feature_dictionary_path)):
    print('[cached] Stage 07 semantic coverage & course capability')
    semantic_reference = pd.read_csv(semantic_reference_path)
    semantic_matrix = pd.read_csv(semantic_feature_matrix_path)
    gap_report = pd.read_csv(semantic_gap_path)
    course_cap = pd.read_csv(course_capability_path, dtype={'course':str})
    rich = pd.read_csv(
        feature_bank_path,
        dtype={'user':str,'course':str,'unit_id':str},
        low_memory=False
    )
    feature_dictionary = pd.read_csv(feature_dictionary_path)
    print('Semantic groups:', len(semantic_reference), '| preferences:', semantic_reference['preference'].nunique())
    print(gap_report['coverage_status'].value_counts())
    preview(gap_report)
else:
    SEMANTIC_GROUPS = [
        ('Active/Reflective','Active','trying_something_out','Trying something out'),
        ('Active/Reflective','Active','social_oriented','Social oriented'),
        ('Active/Reflective','Reflective','think_about_material','Think about material'),
        ('Active/Reflective','Reflective','impersonal_oriented','Impersonal oriented'),
        ('Sensing/Intuitive','Sensing','existing_ways','Existing ways'),
        ('Sensing/Intuitive','Sensing','concrete_material','Concrete material'),
        ('Sensing/Intuitive','Sensing','careful_with_details','Careful with details'),
        ('Sensing/Intuitive','Intuitive','new_ways','New ways'),
        ('Sensing/Intuitive','Intuitive','abstract_material','Abstract material'),
        ('Sensing/Intuitive','Intuitive','not_careful_with_details','Not careful with details'),
        ('Visual/Verbal','Visual','pictures','Pictures'),
        ('Visual/Verbal','Verbal','spoken_words','Spoken words'),
        ('Visual/Verbal','Verbal','written_words','Written words'),
        ('Visual/Verbal','Verbal','difficulty_with_visual_style','Difficulty with visual style'),
        ('Sequential/Global','Sequential','detail_oriented','Detail oriented'),
        ('Sequential/Global','Sequential','sequential_progress','Sequential progress'),
        ('Sequential/Global','Sequential','from_parts_to_whole','From parts to the whole'),
        ('Sequential/Global','Global','overall_picture','Overall picture'),
        ('Sequential/Global','Global','nonsequential_progress','Non-sequential progress'),
        ('Sequential/Global','Global','relations_connections','Relations/connections'),
    ]
    semantic_reference = pd.DataFrame(
        SEMANTIC_GROUPS,
        columns=['dimension','preference','semantic_group_id','semantic_group']
    )
    assert len(semantic_reference) == 20
    assert semantic_reference['preference'].nunique() == 8
    assert semantic_reference['dimension'].nunique() == 4
    save_csv6(semantic_reference, semantic_reference_path, index=False)
    # expected_direction: '+' means HIGH feature values support the semantic group;
    # '-' means LOW feature values support it. Direction is not yet converted to L/M/H.
    MAP = []
    def add_map(dim, pref, sg, feature, direction, source, evidence_type, strength, note=''):
        MAP.append({
            'dimension':dim,'preference':pref,'semantic_group_id':sg,
            'feature':feature,'expected_direction':direction,
            'source':source,'evidence_type':evidence_type,'strength':strength,'note':note
        })

    # Active / Reflective — Graf Table 5.10 semantics.
    for f,d in [
        ('content_visit','-'),('content_stay','-'),('outline_stay_proxy','-'),
        ('selfass_visit_proxy','+'),('exercise_visit_proxy','+'),('exercise_stay_proxy','+'),
        ('quiz_stay_results_proxy','-')
    ]: add_map('Active/Reflective','Active','trying_something_out',f,d,'Graf 2007','Graf-aligned/proxy','STRONG')
    add_map('Active/Reflective','Active','social_oriented','forum_visit','-','Graf 2007','direct behavior','STRONG')
    add_map('Active/Reflective','Active','social_oriented','forum_post','+','Graf 2007','direct behavior','VERY_STRONG')
    for f,d in [
        ('content_visit','+'),('content_stay','+'),('outline_stay_proxy','+'),
        ('selfass_visit_proxy','-'),('selfass_stay_proxy','+'),('exercise_visit_proxy','-'),
        ('exercise_stay_proxy','-'),('quiz_stay_results_proxy','+')
    ]: add_map('Active/Reflective','Reflective','think_about_material',f,d,'Graf 2007','Graf-aligned/proxy','STRONG')
    add_map('Active/Reflective','Reflective','impersonal_oriented','forum_visit','+','Graf 2007','direct behavior','STRONG')
    add_map('Active/Reflective','Reflective','impersonal_oriented','forum_post','-','Graf 2007','direct behavior','VERY_STRONG')

    # Sensing / Intuitive — recoverable subset from Graf Table 5.11.
    add_map('Sensing/Intuitive','Sensing','existing_ways','selfass_visit_proxy','+','Graf 2007','proxy','STRONG')
    add_map('Sensing/Intuitive','Sensing','existing_ways','exercise_visit_proxy','+','Graf 2007','proxy','STRONG')
    add_map('Sensing/Intuitive','Sensing','concrete_material','content_visit','-','Graf 2007','reconstruction','MODERATE')
    add_map('Sensing/Intuitive','Sensing','concrete_material','content_stay','-','Graf 2007','sessionized proxy','MODERATE')
    add_map('Sensing/Intuitive','Sensing','careful_with_details','selfass_stay_proxy','+','Graf 2007','sessionized proxy','STRONG')
    add_map('Sensing/Intuitive','Sensing','careful_with_details','quiz_stay_results_proxy','+','Graf 2007','sessionized proxy','STRONG')
    add_map('Sensing/Intuitive','Intuitive','new_ways','selfass_visit_proxy','-','Graf 2007','proxy','MODERATE')
    add_map('Sensing/Intuitive','Intuitive','abstract_material','content_visit','+','Graf 2007','reconstruction','MODERATE')
    add_map('Sensing/Intuitive','Intuitive','abstract_material','content_stay','+','Graf 2007','sessionized proxy','MODERATE')
    add_map('Sensing/Intuitive','Intuitive','not_careful_with_details','selfass_stay_proxy','-','Graf 2007','sessionized proxy','STRONG')
    add_map('Sensing/Intuitive','Intuitive','not_careful_with_details','quiz_stay_results_proxy','-','Graf 2007','sessionized proxy','STRONG')

    # Visual / Verbal — Graf + explicit modality extension suggested by Graf discussion.
    add_map('Visual/Verbal','Visual','pictures','content_visit','-','Graf 2007','Graf pattern','MODERATE')
    add_map('Visual/Verbal','Visual','pictures','forum_post','-','Graf 2007','Graf pattern','MODERATE')
    add_map('Visual/Verbal','Visual','pictures','image_material_coverage_pct','+','Graf semantic construct + MIME extension','explicit modality','VERY_STRONG')
    add_map('Visual/Verbal','Visual','pictures','image_material_stay_min','+','Graf semantic construct + MIME extension','explicit modality/sessionized','STRONG')
    add_map('Visual/Verbal','Verbal','spoken_words','audio_material_coverage_pct','+','Graf 2007 suggested online pattern','explicit modality','VERY_STRONG')
    add_map('Visual/Verbal','Verbal','spoken_words','audio_material_stay_min','+','Graf 2007 suggested online pattern','explicit modality/sessionized','STRONG')
    add_map('Visual/Verbal','Verbal','spoken_words','video_material_coverage_pct','+','Graf 2007 suggested online candidate','multimodal candidate','MODERATE','Video is not guaranteed to contain spoken narration; use with lower strength.')
    for f,d in [('content_visit','+'),('forum_visit','+'),('forum_stay','+'),('forum_post','+')]:
        add_map('Visual/Verbal','Verbal','written_words',f,d,'Graf 2007','Graf pattern','STRONG')
    add_map('Visual/Verbal','Verbal','written_words','text_material_coverage_pct','+','Graf semantic construct + structural text metadata','explicit modality','VERY_STRONG')
    add_map('Visual/Verbal','Verbal','written_words','text_material_stay_min','+','Graf semantic construct + structural text metadata','explicit modality/sessionized','STRONG')
    # difficulty_with_visual_style intentionally remains without behavioral proxy unless question semantics exist.

    # Sequential / Global — Graf Table 5.13 + structural navigation reconstruction.
    for f in ['outline_visit_proxy','outline_stay_proxy','navigation_overview_visit','navigation_overview_stay']:
        add_map('Sequential/Global','Sequential','detail_oriented',f,'-','Graf 2007','Graf-aligned/proxy','STRONG')
    add_map('Sequential/Global','Sequential','sequential_progress','sequential_transition_ratio','+','Graf semantic navigation + structural reconstruction','navigation reconstruction','VERY_STRONG')
    add_map('Sequential/Global','Sequential','sequential_progress','forward_skip_ratio','-','Graf navigation_skip semantics + structural reconstruction','navigation reconstruction','VERY_STRONG')
    add_map('Sequential/Global','Sequential','sequential_progress','return_to_overview_rate','-','Graf overview/navigation semantics + sequence reconstruction','navigation reconstruction','STRONG')
    add_map('Sequential/Global','Sequential','sequential_progress','cross_section_jump_ratio','-','Graf sequential-progress semantics + structural reconstruction','navigation reconstruction','MODERATE')
    for f in ['outline_visit_proxy','outline_stay_proxy','navigation_overview_visit','navigation_overview_stay']:
        add_map('Sequential/Global','Sequential','from_parts_to_whole',f,'-','Graf 2007','Graf-aligned/proxy','STRONG')
    for f in ['outline_visit_proxy','outline_stay_proxy','navigation_overview_visit','navigation_overview_stay']:
        add_map('Sequential/Global','Global','overall_picture',f,'+','Graf 2007','Graf-aligned/proxy','STRONG')
    add_map('Sequential/Global','Global','nonsequential_progress','forward_skip_ratio','+','Graf navigation_skip semantics + structural reconstruction','navigation reconstruction','VERY_STRONG')
    add_map('Sequential/Global','Global','nonsequential_progress','cross_section_jump_ratio','+','Graf non-sequential semantics + structural reconstruction','navigation reconstruction','STRONG')
    add_map('Sequential/Global','Global','nonsequential_progress','return_to_overview_rate','+','Graf overview/navigation semantics + sequence reconstruction','navigation reconstruction','STRONG')
    add_map('Sequential/Global','Global','nonsequential_progress','sequential_transition_ratio','-','Graf non-sequential semantics + structural reconstruction','navigation reconstruction','VERY_STRONG')
    add_map('Sequential/Global','Global','relations_connections','navigation_overview_visit','+','Graf 2007','Graf pattern','STRONG')
    add_map('Sequential/Global','Global','relations_connections','navigation_overview_stay','+','Graf 2007','Graf pattern','STRONG')
    add_map('Sequential/Global','Global','relations_connections','cross_section_jump_ratio','+','Graf relations concept + structural extension','exploratory structural proxy','MODERATE')

    semantic_matrix = pd.DataFrame(MAP)
    semantic_matrix['strength_weight'] = semantic_matrix['strength'].map(SEMANTIC_STRENGTH_WEIGHT).fillna(0.0)

    # Global empirical availability from quality audit.
    qmap = quality.set_index('feature').to_dict('index') if len(quality) else {}
    semantic_matrix['feature_exists'] = semantic_matrix['feature'].isin(rich.columns)
    semantic_matrix['quality_status'] = semantic_matrix['feature'].map(
        lambda f: qmap.get(f,{}).get('candidate_status','NOT_IN_QUALITY_AUDIT')
    )
    semantic_matrix['nonzero_share'] = semantic_matrix['feature'].map(
        lambda f: qmap.get(f,{}).get('nonzero_share',np.nan)
    )
    semantic_matrix['globally_usable'] = (
        semantic_matrix['feature_exists'] &
        ~semantic_matrix['quality_status'].isin(['DROP_CONSTANT'])
    )
    save_csv6(semantic_matrix, semantic_feature_matrix_path, index=False)
    # Explicit missing/gap requirements from Graf semantics.
    GAPS = [
        ('Sensing/Intuitive','Sensing','existing_ways','example_visit/example_stay or ques_develop','NOT_RECOVERABLE'),
        ('Sensing/Intuitive','Sensing','concrete_material','example_visit/example_stay; ques_facts','NOT_RECOVERABLE'),
        ('Sensing/Intuitive','Sensing','careful_with_details','ques_detail; true quiz_revisions','NOT_RECOVERABLE'),
        ('Sensing/Intuitive','Intuitive','new_ways','example_*; ques_develop','PARTIAL'),
        ('Sensing/Intuitive','Intuitive','abstract_material','example_*; ques_concepts; ques_develop','PARTIAL'),
        ('Sensing/Intuitive','Intuitive','not_careful_with_details','ques_detail; true quiz_revisions','PARTIAL'),
        ('Visual/Verbal','Visual','pictures','ques_graphics performance','PARTIAL_REPLACED_BY_MODALITY_IF_AVAILABLE'),
        ('Visual/Verbal','Verbal','spoken_words','audio/video/voice interaction','RECOVERABLE_IF_MODALITY_AVAILABLE'),
        ('Visual/Verbal','Verbal','difficulty_with_visual_style','ques_graphics performance','NOT_RECOVERABLE'),
        ('Sequential/Global','Sequential','detail_oriented','ques_detail performance','PARTIAL'),
        ('Sequential/Global','Global','overall_picture','ques_overview performance','PARTIAL'),
        ('Sequential/Global','Global','relations_connections','ques_overview/interpret/develop performance','PARTIAL'),
    ]
    gap_report = pd.DataFrame(GAPS, columns=['dimension','preference','semantic_group_id','missing_or_additional_evidence','gap_status'])

    # Add actual available-feature counts per semantic group.
    semantic_matrix['usable_strength_weight'] = np.where(
        semantic_matrix['globally_usable'], semantic_matrix['strength_weight'], 0.0
    )
    summary = semantic_matrix.groupby(['dimension','preference','semantic_group_id']).agg(
        mapped_features=('feature','nunique'),
        usable_features=('globally_usable','sum'),
        max_strength_weight=('usable_strength_weight','max')
    ).reset_index()
    gap_report = semantic_reference.merge(summary, on=['dimension','preference','semantic_group_id'], how='left').merge(
        gap_report, on=['dimension','preference','semantic_group_id'], how='left'
    )
    gap_report[['mapped_features','usable_features','max_strength_weight']] = gap_report[['mapped_features','usable_features','max_strength_weight']].fillna(0)
    gap_report['coverage_status'] = np.where(
        gap_report['usable_features'] == 0, 'GAP',
        np.where(gap_report['max_strength_weight'] >= 2.5, 'COVERED_STRONG', 'COVERED_PROXY_OR_MODERATE')
    )
    save_csv6(gap_report, semantic_gap_path, index=False)
    # ---------- Course feature support ----------
    # One course-level row is enough for structural capability columns.
    course_base = rich.sort_values(['course','user']).drop_duplicates('course').copy()
    course_support_rows = []

    content_den_cols = [c for c in course_base.columns if c.startswith('course_') and c.endswith('_objects') and any(f'course_{m}_objects'==c for m in CONTENT_MODULES)]
    content_available = course_base[content_den_cols].fillna(0).sum(axis=1) if content_den_cols else pd.Series(0,index=course_base.index)

    for r_idx, r in course_base.iterrows():
        support = {}
        support['content_visit'] = support['content_stay'] = float(content_available.loc[r_idx]) > 0
        support['selfass_visit_proxy'] = support['selfass_stay_proxy'] = support['quiz_stay_results_proxy'] = float(r.get('course_quiz_objects',0) or 0) > 0
        support['exercise_visit_proxy'] = support['exercise_stay_proxy'] = float(r.get('course_assign_objects',0) or 0) > 0
        support['forum_visit'] = support['forum_stay'] = support['forum_post'] = float(r.get('course_forum_objects',0) or 0) > 0
        support['navigation_overview_visit'] = support['navigation_overview_stay'] = float(r.get('support_overview',0) or 0) > 0
        support['outline_visit_proxy'] = support['outline_stay_proxy'] = float(r.get('course_section_count',0) or 0) > 0

        nav_ok = (
            float(r.get('course_navigation_object_count',0) or 0) >= 2
            and float(r.get('course_canonical_navigation_order',0) or 0) > 0
        )
        for f in ['sequential_transition_ratio','forward_skip_ratio','backward_navigation_ratio','same_object_repeat_ratio','same_section_transition_ratio','cross_section_jump_ratio','return_to_overview_rate','overview_bridge_rate']:
            support[f] = nav_ok and f in rich.columns

        for mod in ['text','image','audio','video','presentation']:
            cap = float(r.get(f'course_{mod}_modality_objects',0) or 0) > 0
            for suffix in ['visit_count','unique_objects','coverage_pct','stay_min']:
                support[f'{mod}_material_{suffix}'] = cap and f'{mod}_material_{suffix}' in rich.columns

        row = {'course':str(r['course'])}
        for mr in semantic_matrix.itertuples(index=False):
            if not bool(mr.globally_usable):
                continue
            key = f"sg__{mr.semantic_group_id}"
            row[key] = row.get(key, 0.0) + (mr.strength_weight if support.get(mr.feature, False) else 0.0)
        course_support_rows.append(row)

    course_cap = pd.DataFrame(course_support_rows).fillna(0.0)
    sg_cols = [f'sg__{x}' for x in semantic_reference['semantic_group_id']]
    for c in sg_cols:
        if c not in course_cap.columns:
            course_cap[c] = 0.0

    course_cap['semantic_group_covered_n'] = (course_cap[sg_cols] > 0).sum(axis=1)
    course_cap['semantic_group_strong_n'] = (course_cap[sg_cols] >= 2.5).sum(axis=1)
    course_cap['semantic_capability_score'] = course_cap[sg_cols].sum(axis=1)

    # Preference and bipolar-dimension coverage.
    for pref in semantic_reference['preference'].unique():
        groups = semantic_reference.loc[semantic_reference['preference']==pref,'semantic_group_id'].tolist()
        course_cap[f'pref__{pref.lower()}'] = (course_cap[[f'sg__{g}' for g in groups]].sum(axis=1) > 0).astype(int)

    pref_cols = [f'pref__{p.lower()}' for p in semantic_reference['preference'].unique()]
    course_cap['preference_covered_n'] = course_cap[pref_cols].sum(axis=1)

    pairs = [
        ('Active','Reflective'),('Sensing','Intuitive'),('Visual','Verbal'),('Sequential','Global')
    ]
    for a,b in pairs:
        course_cap[f'dim_pair__{a.lower()}_{b.lower()}'] = (
            (course_cap[f'pref__{a.lower()}']>0) & (course_cap[f'pref__{b.lower()}']>0)
        ).astype(int)
    pair_cols = [c for c in course_cap.columns if c.startswith('dim_pair__')]
    course_cap['dimension_pair_covered_n'] = course_cap[pair_cols].sum(axis=1)
    save_csv6(course_cap, course_capability_path, index=False)
    # Merge capability scores into rich feature bank as audit columns only.
    rich = rich.merge(
        course_cap[['course','dimension_pair_covered_n','preference_covered_n','semantic_group_covered_n','semantic_group_strong_n','semantic_capability_score']],
        on='course', how='left'
    )
    for c in ['dimension_pair_covered_n','preference_covered_n','semantic_group_covered_n','semantic_group_strong_n','semantic_capability_score']:
        rich[c] = rich[c].fillna(0)

    # Dictionary roles for capability audit columns.
    fd = feature_dictionary.copy()
    existing = set(fd['feature'].astype(str))
    add = []
    for c in ['dimension_pair_covered_n','preference_covered_n','semantic_group_covered_n','semantic_group_strong_n','semantic_capability_score']:
        if c not in existing:
            add.append({'feature':c,'role':'semantic_capability_audit','warning':'','definition':'Course-level semantic observability/capability; not a learner clustering feature.'})
    if add:
        fd = pd.concat([fd,pd.DataFrame(add)], ignore_index=True).drop_duplicates('feature',keep='last')
    feature_dictionary = fd

    save_csv6(rich, feature_bank_path, index=False)
    save_csv6(feature_dictionary, feature_dictionary_path, index=False)
    print('Semantic groups:', len(semantic_reference), '| preferences:', semantic_reference['preference'].nunique())
    print(gap_report['coverage_status'].value_counts())
    preview(gap_report)


# ============================================================
# Stage 08 — semantic-aware one-course-per-learner selection
# ============================================================
selected_path = OUT_DIR / '08_selected_user_course_rich_numeric.csv'
selection_audit_path = OUT_DIR / '08_course_selection_audit.json'


if (not FORCE_REBUILD) and (file_ready(selected_path) and file_ready(selection_audit_path)):
    print('[cached] Stage 08 semantic-aware course selection')
    selected = pd.read_csv(
        selected_path,
        dtype={'user':str,'course':str,'unit_id':str},
        low_memory=False
    )
    selection_audit = json.loads(
        selection_audit_path.read_text(encoding='utf-8')
    )
    print(json.dumps(selection_audit, indent=2))
    preview(
        selected[
            ['user','course','dimension_pair_covered_n','preference_covered_n',
             'semantic_group_covered_n','semantic_capability_score',
             'total_behavior_events']
        ].head()
    )
else:
    sel = rich.copy()
    sel['_course_num'] = pd.to_numeric(sel['course'], errors='coerce').fillna(10**18)

    sort_cols = [
        'user',
        'dimension_pair_covered_n',
        'preference_covered_n',
        'semantic_group_strong_n',
        'semantic_group_covered_n',
        'semantic_capability_score',
        'total_behavior_events',
        'active_days',
        'last_t',
        '_course_num','course'
    ]
    ascending = [True,False,False,False,False,False,False,False,False,True,True]

    sel = sel.sort_values(sort_cols, ascending=ascending, kind='stable')
    sel['n_courses_user'] = sel.groupby('user')['course'].transform('size')
    sel['selection_rank'] = sel.groupby('user').cumcount() + 1
    selected = sel[sel['selection_rank']==1].drop(columns=['_course_num']).copy()
    save_csv6(selected, selected_path, index=False)
    selection_audit = {
        'user_course_rows_before_selection': int(len(rich)),
        'unique_users_before_selection': int(rich['user'].nunique()),
        'selected_rows': int(len(selected)),
        'selected_unique_users': int(selected['user'].nunique()),
        'users_with_multiple_courses': int((selected['n_courses_user']>1).sum()),
        'selection_rule': (
            'max dimension_pair_covered_n; preference_covered_n; semantic_group_strong_n; '
            'semantic_group_covered_n; semantic_capability_score; total_behavior_events; '
            'active_days; latest last_t; smallest course id'
        ),
        'selected_dimension_pair_distribution': selected['dimension_pair_covered_n'].value_counts().sort_index().to_dict(),
        'selected_preference_coverage_distribution': selected['preference_covered_n'].value_counts().sort_index().to_dict(),
        'selected_semantic_group_coverage_summary': {
            'min': float(selected['semantic_group_covered_n'].min()),
            'median': float(selected['semantic_group_covered_n'].median()),
            'max': float(selected['semantic_group_covered_n'].max()),
        },
        'filters_users_for_incomplete_semantic_support': False,
        'uses_outcome_grade_label': False
    }
    selection_audit_path.write_text(json.dumps(selection_audit, indent=2), encoding='utf-8')

    print(json.dumps(selection_audit, indent=2))
    preview(selected[['user','course','dimension_pair_covered_n','preference_covered_n','semantic_group_covered_n','semantic_capability_score','total_behavior_events']].head())


# ============================================================
# Stage 09 — validation & manifest
# ============================================================

manifest_cache_path = OUT_DIR / '09_preprocessing_manifest.json'
validation_cache_path = OUT_DIR / '09_validation.json'
if (not FORCE_REBUILD) and file_ready(manifest_cache_path) and file_ready(validation_cache_path):
    print('[cached] Stage 09 validation & manifest')
    checks = json.loads(validation_cache_path.read_text(encoding='utf-8'))
    print(json.dumps(checks, indent=2))
else:
    forbidden_tokens = ('grade','target','label','class','outcome','bin_target')

    fd_role = feature_dictionary.set_index('feature')['role'].to_dict()
    model_candidate_roles = {
        'engineered_behavior_candidate','atomic_event_evidence','graf_aligned_behavior_candidate',
        'semantic_navigation_candidate','modality_behavior_candidate'
    }
    model_candidate_cols = [c for c in selected.columns if fd_role.get(c) in model_candidate_roles]
    forbidden = [c for c in model_candidate_cols if any(tok in c.lower() for tok in forbidden_tokens)]

    checks = {
        'one_row_per_user_selected': bool(selected['user'].is_unique),
        'no_outcome_like_candidate_names': bool(len(forbidden)==0),
        'n_selected_users': int(len(selected)),
        'n_feature_bank_columns': int(rich.shape[1]),
        'n_model_candidate_columns': int(len(model_candidate_cols)),
        'semantic_groups_defined': int(len(semantic_reference)),
        'fslsm_preferences_defined': int(semantic_reference['preference'].nunique()),
        'fslsm_dimensions_defined': int(semantic_reference['dimension'].nunique()),
        'sessionized_dwell_enabled': True,
        'dwell_boundary': 'next valid raw learner event',
        'uses_direct_moodle_duration': False,
        'numeric_output_decimals': NUMERIC_DECIMALS,
        'rerun_policy': 'SKIP-SUDAH for completed persistent stages',
        'categorization_LMH_done': False,
        'clustering_done': False,
        'forbidden_candidate_names': forbidden,
    }
    assert checks['one_row_per_user_selected'], checks
    assert checks['no_outcome_like_candidate_names'], checks
    assert checks['semantic_groups_defined'] == 20, checks
    assert checks['fslsm_preferences_defined'] == 8, checks
    assert checks['fslsm_dimensions_defined'] == 4, checks

    manifest = {
        'pipeline': (
            'RAW MYSQL -> SCHEMA AUDIT -> STRUCTURAL/SEMANTIC METADATA -> '
            'OFFICIAL/P95 WINDOW -> EVENT TAXONOMY -> USER-COURSE AGGREGATION -> '
            'RAW-BOUNDARY SESSIONIZED DWELL -> RICH NUMERICAL FEATURE BANK -> '
            'GRAF RECOVERABLE FEATURES -> NAVIGATION/SEQUENCE -> MODALITY -> '
            'QUALITY/REDUNDANCY -> 20 SEMANTIC GROUP COVERAGE -> '
            'COURSE SEMANTIC CAPABILITY -> SEMANTIC-AWARE ONE COURSE PER LEARNER'
        ),
        'source_dump': str(DUMP_PATH),
        'uses_dataset_EP': False,
        'uses_outcome_grade_label': False,
        'uses_direct_dwell_duration': False,
        'uses_sessionized_dwell_proxy': True,
        'dwell_policy': DWELL_POLICY,
        'course_window_policy': 'official start/end when valid; empirical P95 fallback',
        'semantic_target': '20 characteristic semantic groups -> 8 FSLSM preferences -> 4 dimensions',
        'numeric_output_decimals': NUMERIC_DECIMALS,
        'rerun_policy': 'SKIP-SUDAH for completed persistent stages',
        'categorization_LMH_done': False,
        'reason_no_LMH_yet': (
            'Final features and feature-specific thresholds must be frozen only after '
            'semantic coverage, quality, redundancy, and dimensional-balance review.'
        ),
        'outputs': {
            'schema':'00_mysql_table_schema.csv',
            'structural_audit':'01_structural_semantic_audit.json',
            'navigation_order':'01_course_navigation_order.csv',
            'modality_inventory':'01_course_modality_inventory.csv',
            'course_windows':'02_course_windows.csv',
            'taxonomy':'03_taxonomy_config.json',
            'event_vocabulary':'04_event_vocabulary.csv',
            'sessionized_dwell':'04b_sessionized_dwell_long.csv',
            'rich_feature_bank':'05_rich_feature_bank_all_user_course.csv',
            'graf_features':'05b_graf_recoverable_feature_specification.csv',
            'navigation_features':'05c_navigation_features.csv',
            'modality_features':'05c_modality_features.csv',
            'feature_quality':'06_feature_quality_audit.csv',
            'semantic_reference':'07_semantic_group_reference.csv',
            'semantic_feature_matrix':'07_semantic_feature_matrix.csv',
            'semantic_gap_report':'07_semantic_gap_report.csv',
            'course_semantic_capability':'07_course_semantic_capability.csv',
            'selected_user_course':'08_selected_user_course_rich_numeric.csv'
        },
        'validation': checks
    }

    (OUT_DIR / '09_preprocessing_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (OUT_DIR / '09_validation.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')

    print(json.dumps(checks, indent=2))
    print('\nNEXT: freeze literature-supported final feature set, audit dimension balance, then build L/M/H.')


# Public-study export: only the 13 variables used in the manuscript.
STUDY_FEATURES = ['content_visit', 'content_stay', 'selfass_visit_proxy', 'selfass_stay_proxy', 'exercise_visit_proxy', 'exercise_stay_proxy', 'forum_visit', 'forum_stay', 'forum_post', 'quiz_stay_results_proxy', 'navigation_overview_visit', 'navigation_overview_stay', 'return_to_overview_rate']
missing=[c for c in STUDY_FEATURES if c not in selected.columns]
if missing: raise RuntimeError(f'Missing final study features: {missing}')
public_numeric=selected[['user']+STUDY_FEATURES].copy()
public_numeric.to_csv(OUT_DIR/'main_lms_numeric_from_raw.csv',index=False,float_format='%.6f')
print('Wrote', OUT_DIR/'main_lms_numeric_from_raw.csv', len(public_numeric))
