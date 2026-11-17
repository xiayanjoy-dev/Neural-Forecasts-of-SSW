from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent
m=json.loads((r/'MANIFEST.json').read_text())
for name,v in m['files'].items():
 p=r/name
 assert p.is_file(),name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'],name
print('SOURCE_MANIFEST_OK',len(m['files']))
