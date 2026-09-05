"""VeriFYD Lens Security V1: additive static inspection of quarantined files."""
from __future__ import annotations
from pathlib import Path
import struct, zipfile

EXECUTABLE_EXTS={'.exe','.dll','.scr','.com','.msi','.cpl','.sys','.ocx'}
SCRIPT_EXTS={'.ps1','.psm1','.bat','.cmd','.vbs','.vbe','.js','.jse','.wsf','.wsh','.hta'}
SHORTCUT_EXTS={'.lnk','.url'}
ARCHIVE_EXTS={'.zip','.jar','.apk','.docx','.xlsx','.pptx','.odt','.ods','.odp'}
MEDIA_EXTS={'.jpg','.jpeg','.png','.gif','.webp','.heic','.heif','.mp4','.mov','.m4v','.avi','.mkv','.webm','.mpg','.mpeg','.mp3','.wav','.m4a','.aac','.flac','.ogg','.oga','.opus'}
DOCUMENT_EXTS={'.pdf','.doc','.docx','.xls','.xlsx','.ppt','.pptx','.rtf','.odt','.ods','.odp'}
MAX_STATIC_READ=32*1024*1024
MAX_ZIP_ENTRIES=5000
MAX_ZIP_UNCOMPRESSED=1024*1024*1024
MAX_ZIP_RATIO=250.0

def _safe_read(path,limit=MAX_STATIC_READ):
    with Path(path).open('rb') as fh:return fh.read(limit)

def _double_extension(filename):
    lower=(filename or '').lower(); risky=tuple(EXECUTABLE_EXTS|SCRIPT_EXTS|SHORTCUT_EXTS)
    decoys=('.pdf','.doc','.docx','.xls','.xlsx','.ppt','.pptx','.jpg','.jpeg','.png','.gif','.webp','.txt','.csv','.rtf')
    return any(lower.endswith(d+r) for d in decoys for r in risky)

def _expected_magic(ext,data):
    if not data:return None
    if ext in EXECUTABLE_EXTS:return data.startswith(b'MZ')
    if ext=='.pdf':return data.startswith(b'%PDF-')
    if ext=='.png':return data.startswith(b'\x89PNG\r\n\x1a\n')
    if ext in {'.jpg','.jpeg'}:return data.startswith(b'\xff\xd8\xff')
    if ext=='.gif':return data.startswith((b'GIF87a',b'GIF89a'))
    if ext in ARCHIVE_EXTS:return data.startswith((b'PK\x03\x04',b'PK\x05\x06',b'PK\x07\x08'))
    return None

def _valid_pe_at(data,off):
    try:
        if off<0 or off+0x40>len(data) or data[off:off+2]!=b'MZ':return False
        pe_rel=struct.unpack_from('<I',data,off+0x3C)[0]; pe=off+pe_rel
        return pe+4<=len(data) and data[pe:pe+4]==b'PE\x00\x00'
    except Exception:return False

def _find_embedded_pe(data,allow_offset_zero=False):
    pos=data.find(b'MZ',0 if allow_offset_zero else 1); checks=0
    while pos!=-1 and checks<256:
        if _valid_pe_at(data,pos):return pos
        pos=data.find(b'MZ',pos+2); checks+=1
    return None

def _pdf_findings(data):
    low=data.lower(); delta=0; findings=[]; hard=False
    for marker,penalty,label in [
        (b'/javascript',-18,'PDF contains JavaScript'),(b'/js',-12,'PDF contains a JavaScript action marker'),
        (b'/openaction',-12,'PDF contains an automatic OpenAction'),(b'/launch',-35,'PDF contains a Launch action'),
        (b'/embeddedfile',-12,'PDF contains an embedded file'),(b'/richmedia',-12,'PDF contains RichMedia content')]:
        if marker in low:delta+=penalty;findings.append(label)
    if b'/launch' in low and (b'/javascript' in low or b'/js' in low):hard=True;findings.append('PDF combines launch behavior with script content')
    return delta,findings,hard

def _script_findings(data):
    text=data[:2_000_000].decode('utf-8',errors='ignore').lower();delta=0;findings=[];hits=0
    for needle,penalty,label in [
        ('powershell',-8,'Script references PowerShell'),('invoke-expression',-18,'Script uses Invoke-Expression'),('iex(',-18,'Script uses IEX execution'),
        ('frombase64string',-18,'Script decodes Base64 content'),('downloadstring',-20,'Script downloads remote content'),('webclient',-10,'Script uses a web client'),
        ('bitsadmin',-25,'Script invokes BITSAdmin'),('certutil',-18,'Script invokes CertUtil'),('mshta',-25,'Script invokes MSHTA'),('rundll32',-18,'Script invokes Rundll32'),('regsvr32',-18,'Script invokes Regsvr32')]:
        if needle in text:delta+=penalty;findings.append(label);hits+=1
    hard=hits>=3
    if hard:findings.append('Script contains multiple high-risk execution/download techniques')
    return max(delta,-85),findings,hard

def _zip_findings(path,outer_ext):
    delta=0;findings=[];hard=False;meta={'zip_entries':0,'zip_uncompressed_bytes':0,'zip_compressed_bytes':0,'zip_ratio':0.0}
    try:
        with zipfile.ZipFile(path,'r') as zf:
            infos=zf.infolist();total_u=sum(max(0,i.file_size) for i in infos);total_c=sum(max(0,i.compress_size) for i in infos);ratio=total_u/max(1,total_c)
            meta.update(zip_entries=len(infos),zip_uncompressed_bytes=total_u,zip_compressed_bytes=total_c,zip_ratio=round(ratio,2))
            if len(infos)>MAX_ZIP_ENTRIES:delta-=35;hard=True;findings.append(f'Archive contains an excessive number of entries ({len(infos)})')
            if total_u>MAX_ZIP_UNCOMPRESSED:delta-=40;hard=True;findings.append('Archive expands beyond the Lens static-inspection safety limit')
            if ratio>MAX_ZIP_RATIO and total_u>50*1024*1024:delta-=45;hard=True;findings.append(f'Archive has an extreme compression ratio ({ratio:.1f}:1)')
            suspicious=[];traversal=[];macros=[]
            for info in infos[:MAX_ZIP_ENTRIES]:
                name=info.filename.replace('\\','/');lower=name.lower();parts=[p for p in lower.split('/') if p not in ('','.')]
                if '..' in parts or lower.startswith('/'):traversal.append(name)
                if Path(lower).suffix in EXECUTABLE_EXTS|SCRIPT_EXTS|SHORTCUT_EXTS:suspicious.append(name)
                if lower.endswith('vbaproject.bin') or '/macros/' in lower:macros.append(name)
            if traversal:delta-=40;hard=True;findings.append('Archive contains path-traversal entries')
            if suspicious:
                delta-=min(45,15+5*min(len(suspicious),6));findings.append(f'Archive contains {len(suspicious)} executable/script/shortcut item(s)')
                if outer_ext in MEDIA_EXTS|DOCUMENT_EXTS:hard=True;findings.append('Executable/script content is embedded inside a file presented as media/document')
            if macros:
                delta-=20;findings.append('Office-style container includes VBA/macro content')
                if outer_ext in {'.docx','.xlsx','.pptx'}:delta-=20;hard=True;findings.append('Macro payload appears inside a normally macro-free Office extension')
    except zipfile.BadZipFile:
        if outer_ext in ARCHIVE_EXTS:delta-=25;findings.append('File extension indicates a ZIP-based container, but archive structure is invalid')
    except Exception as exc:findings.append(f'Archive inspection unavailable ({type(exc).__name__})')
    return max(delta,-90),findings,hard,meta

def scan_static_security(path,filename=''):
    p=Path(path);name=filename or p.name;ext=Path(name).suffix.lower();result={'engine':'verifyd_static_v1','status':'CLEAN_HINT','score_delta':0,'hard_block':False,'findings':[],'details':{'extension':ext,'size_bytes':0,'magic_match':None,'embedded_pe_offset':None}}
    if not p.exists() or not p.is_file():result.update(status='REVIEW',score_delta=-20,findings=['Static security inspection could not access the quarantined file']);return result
    result['details']['size_bytes']=p.stat().st_size
    try:data=_safe_read(p)
    except Exception as exc:result.update(status='REVIEW',score_delta=-20,findings=[f'Static security inspection could not read the file ({type(exc).__name__})']);return result
    delta=0;findings=[];hard=False
    if _double_extension(name):delta-=55;hard=True;findings.append('Suspicious double file extension detected')
    magic=_expected_magic(ext,data);result['details']['magic_match']=magic
    if magic is False:delta-=30;findings.append('File signature does not match its filename extension')
    embedded=_find_embedded_pe(data,allow_offset_zero=(ext in EXECUTABLE_EXTS));result['details']['embedded_pe_offset']=embedded
    if embedded is not None and ext not in EXECUTABLE_EXTS:delta-=70;hard=True;findings.append(f'Valid Windows executable payload detected inside a non-executable file (offset {embedded})')
    if ext=='.pdf' or data.startswith(b'%PDF-'):
        d,f,h=_pdf_findings(data);delta+=d;findings.extend(f);hard=hard or h
    if ext in SCRIPT_EXTS:
        d,f,h=_script_findings(data);delta+=d;findings.extend(f);hard=hard or h
    if ext in ARCHIVE_EXTS or data.startswith((b'PK\x03\x04',b'PK\x05\x06',b'PK\x07\x08')):
        d,f,h,meta=_zip_findings(p,ext);delta+=d;findings.extend(f);hard=hard or h;result['details'].update(meta)
    if ext in SHORTCUT_EXTS:delta-=25;findings.append('Windows shortcut/link file requires additional review')
    if ext in EXECUTABLE_EXTS:delta-=15;findings.append('Executable file requires antivirus/reputation review before release')
    delta=max(-95,min(0,int(delta)));status='BLOCK' if hard else 'REVIEW' if delta<0 else 'CLEAN_HINT'
    if not findings:findings.append('No additional static structural warning detected')
    result.update(status=status,score_delta=delta,hard_block=bool(hard),findings=findings);return result
