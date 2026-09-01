from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from urllib.parse import urlparse, unquote
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import hashlib, ipaddress, mimetypes, os, re, shutil, socket, subprocess, time, uuid
import httpx

app = FastAPI(title="VeriFYD Lens Agent", version="0.4.2")
BASE = Path.home() / "VeriFYD" / "Lens"
QUARANTINE = BASE / "Quarantine"
DOWNLOADS = Path.home() / "Downloads"
QUARANTINE.mkdir(parents=True, exist_ok=True); DOWNLOADS.mkdir(parents=True, exist_ok=True)
MAX_DOWNLOAD_BYTES = 250 * 1024 * 1024
SCAN_STATE = {}
EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="verifyd-lens")
CLOUD_URL = os.environ.get("VERIFYD_LENS_CLOUD_URL","https://verifyd-backend.onrender.com").rstrip("/")
CLOUD_KEY = os.environ.get("VERIFYD_LENS_API_KEY","").strip()
CLOUD_POLL_SECONDS = int(os.environ.get("VERIFYD_LENS_POLL_SECONDS","3"))
CLOUD_TIMEOUT_SECONDS = int(os.environ.get("VERIFYD_LENS_CLOUD_TIMEOUT","240"))

class UrlScanRequest(BaseModel): url: str
def clamp(v): return max(0,min(100,int(v)))
def classify(score): return "LOW CONCERN" if score >= 80 else "REVIEW RECOMMENDED" if score >= 50 else "HIGH CONCERN"

def safe_filename(name):
    name=unquote(name or "").strip().replace("\\","_").replace("/","_")
    name=re.sub(r'[<>:"|?*\x00-\x1f]',"_",name).rstrip(". ")
    return name[:180] or "download.bin"
def filename_from_url(url): return safe_filename(Path(unquote(urlparse(url).path)).name or "download.bin")

def resolve_public_host(host):
    if not host: raise HTTPException(400,"URL has no hostname.")
    try: infos=socket.getaddrinfo(host,None)
    except socket.gaierror: raise HTTPException(400,"Could not resolve source hostname.")
    for info in infos:
        ip=ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            raise HTTPException(400,"Private/local network download targets are blocked.")

def url_findings(url):
    p=urlparse(url)
    if p.scheme not in {"http","https"}: raise HTTPException(400,"Only HTTP/HTTPS URLs are supported.")
    resolve_public_host(p.hostname or "")
    host=(p.hostname or "").lower(); path=p.path.lower(); findings=[]; score=90
    if p.scheme!="https": score-=12; findings.append("Source uses unencrypted HTTP")
    if any(host.endswith(t) for t in {".zip",".mov",".top",".click",".xyz"}):
        score-=18; findings.append("Source domain deserves additional review")
    if any(path.endswith(x) for x in [".pdf.exe",".docx.exe",".xlsx.exe",".pptx.exe",".jpg.exe",".jpeg.exe",".png.exe",".txt.exe"]):
        score-=55; findings.append("Suspicious double file extension detected")
    if not findings: findings.append("No obvious URL-level warning detected")
    return clamp(score), findings

def sha256_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def find_mpcmdrun():
    pd=Path(os.environ.get("ProgramData",r"C:\ProgramData"))/"Microsoft"/"Windows Defender"/"Platform"
    if pd.exists():
        c=sorted([p/"MpCmdRun.exe" for p in pd.iterdir() if (p/"MpCmdRun.exe").exists()],reverse=True)
        if c: return c[0]
    legacy=Path(os.environ.get("ProgramFiles",r"C:\Program Files"))/"Windows Defender"/"MpCmdRun.exe"
    return legacy if legacy.exists() else None

def ps_defender(path):
    pp=str(path).replace("'","''")
    script=("$ErrorActionPreference='Stop';$start=Get-Date;"
            f"Start-MpScan -ScanType CustomScan -ScanPath '{pp}';"
            "$hit=Get-MpThreatDetection|?{$_.InitialDetectionTime -ge $start.AddSeconds(-10) -and "
            f"(($_.Resources -join ' ') -like '*{pp}*')"+"}|select -First 1;"
            "if($hit){'VERIFYD_THREAT'}else{'VERIFYD_CLEAN'}")
    try:
        cp=subprocess.run(["powershell.exe","-NoProfile","-NonInteractive","-Command",script],capture_output=True,text=True,timeout=210)
        out=((cp.stdout or "")+"\n"+(cp.stderr or "")).strip()
        if "VERIFYD_THREAT" in out:
            return {"status":"THREAT_OR_WARNING","score_delta":-75,"finding":"Microsoft Defender reported a threat or security warning","method":"PowerShell Start-MpScan","detail":out[-1000:]}
        if cp.returncode==0 and "VERIFYD_CLEAN" in out:
            return {"status":"NO_KNOWN_THREAT_REPORTED","score_delta":0,"finding":"Microsoft Defender reported no known threat in this scan","method":"PowerShell Start-MpScan","detail":out[-1000:]}
        return {"status":"UNAVAILABLE","score_delta":0,"finding":"Microsoft Defender could not complete a reliable scan; security score was not penalized","method":"PowerShell Start-MpScan","detail":out[-1000:]}
    except Exception as e:
        return {"status":"UNAVAILABLE","score_delta":0,"finding":"Microsoft Defender could not complete a reliable scan; security score was not penalized","method":"PowerShell Start-MpScan","detail":type(e).__name__}

def defender_scan(path):
    exe=find_mpcmdrun()
    if not exe: return ps_defender(path)
    try:
        cp=subprocess.run([str(exe),"-Scan","-ScanType","3","-File",str(path),"-ReturnHR"],capture_output=True,text=True,timeout=180)
        out=((cp.stdout or "")+"\n"+(cp.stderr or "")).strip(); low=out.lower()
        if any(x in low for x in ["threat found","threat detected","malware detected","threat(s) found","found threats"]):
            return {"status":"THREAT_OR_WARNING","score_delta":-75,"finding":"Microsoft Defender reported a threat or security warning","method":"MpCmdRun","detail":out[-1000:]}
        if cp.returncode==0:
            return {"status":"NO_KNOWN_THREAT_REPORTED","score_delta":0,"finding":"Microsoft Defender reported no known threat in this scan","method":"MpCmdRun","detail":out[-1000:]}
        fb=ps_defender(path); fb["mpcmdrun_exit"]=cp.returncode; return fb
    except Exception: return ps_defender(path)

def content_type_check(filename,ct):
    expected,_=mimetypes.guess_type(filename); actual=(ct or "").split(";")[0].strip().lower()
    if expected and actual and actual not in {"application/octet-stream",expected.lower()} and not (expected.startswith("text/") and actual.startswith("text/")):
        return [f"File type mismatch: extension suggests {expected}, server reported {actual}"]
    return []

def unique_destination(directory,filename):
    c=directory/filename
    if not c.exists(): return c
    for i in range(1,1000):
        n=directory/f"{c.stem} ({i}){c.suffix}"
        if not n.exists(): return n
    raise HTTPException(500,"Could not allocate a unique destination filename.")

def cloud_analyze(path,filename):
    if not CLOUD_KEY: return {"status":"NOT_CONFIGURED","finding":"VeriFYD cloud authenticity analysis is not configured.","authenticity_score":None,"label":None}
    headers={"X-VeriFYD-Lens-Key":CLOUD_KEY}
    try:
        with path.open("rb") as fh:
            r=httpx.post(f"{CLOUD_URL}/lens/analyze",headers=headers,files={"file":(filename,fh,"application/octet-stream")},timeout=120)
        data=r.json()
        if r.status_code>=400: return {"status":"ERROR","finding":f"VeriFYD cloud upload failed ({r.status_code})","authenticity_score":None,"label":None}
        jid=data.get("job_id"); media=data.get("media_type")
        if not jid: return {"status":"ERROR","finding":"VeriFYD cloud did not return a job ID.","authenticity_score":None,"label":None}
        deadline=time.time()+CLOUD_TIMEOUT_SECONDS
        while time.time()<deadline:
            time.sleep(CLOUD_POLL_SECONDS)
            rr=httpx.get(f"{CLOUD_URL}/lens/job/{jid}",headers=headers,timeout=30)
            if rr.status_code==404: continue
            x=rr.json(); state=str(x.get("job_status") or "").lower()
            if state in {"queued","processing","started","busy"}: continue
            if state=="error": return {"status":"ERROR","finding":"VeriFYD cloud analysis failed.","authenticity_score":None,"label":None,"media_type":media}
            return {"status":"COMPLETE","finding":f"VeriFYD authenticity analysis complete ({media or 'file'})","authenticity_score":x.get("authenticity_score"),"label":x.get("label"),"media_type":media,"reasoning":x.get("gpt_reasoning") or ""}
        return {"status":"TIMEOUT","finding":"VeriFYD cloud authenticity analysis timed out.","authenticity_score":None,"label":None,"media_type":media}
    except Exception as e:
        return {"status":"ERROR","finding":f"VeriFYD cloud connection error: {type(e).__name__}","authenticity_score":None,"label":None}

def scan_worker(scan_id,url):
    q=None
    try:
        score,findings=url_findings(url); filename=filename_from_url(url)
        q=unique_destination(QUARANTINE,f"{scan_id[:8]}_{filename}")
        SCAN_STATE[scan_id].update(status="DOWNLOADING",summary="DOWNLOADING",filename=filename,findings=findings+["Downloading into VeriFYD Lens quarantine…"])
        with httpx.stream("GET",url,headers={"User-Agent":"VeriFYD-Lens/0.4.2","Accept":"*/*"},follow_redirects=True,timeout=httpx.Timeout(30.0,read=120.0)) as r:
            r.raise_for_status(); resolve_public_host(urlparse(str(r.url)).hostname or "")
            if r.headers.get("content-length") and int(r.headers["content-length"])>MAX_DOWNLOAD_BYTES: raise RuntimeError("File exceeds the 250 MB MVP limit.")
            total=0
            with q.open("wb") as f:
                for chunk in r.iter_bytes(1024*1024):
                    total+=len(chunk)
                    if total>MAX_DOWNLOAD_BYTES: raise RuntimeError("File exceeds the 250 MB MVP limit.")
                    f.write(chunk)
            findings+=content_type_check(filename,r.headers.get("content-type",""))
            if any("File type mismatch" in x for x in findings): score-=18
        sha=sha256_file(q); findings.append(f"SHA-256 fingerprint created: {sha[:16]}…")
        SCAN_STATE[scan_id].update(status="SECURITY_SCANNING",summary="SECURITY SCANNING",sha256=sha,size_bytes=q.stat().st_size,quarantine_path=str(q),findings=findings)
        defender=defender_scan(q); score+=defender.get("score_delta",0); findings.append(defender["finding"])
        if not q.exists():
            SCAN_STATE[scan_id].update(status="BLOCKED",summary="HIGH CONCERN",security_score=clamp(score),trust_score=clamp(score),authenticity_score=None,defender_status=defender["status"],defender_method=defender.get("method"),findings=findings+["The quarantined file is no longer present after security scanning."],recommended_action="block"); return
        SCAN_STATE[scan_id].update(status="AUTHENTICITY_SCANNING",summary="AUTHENTICITY SCANNING",security_score=clamp(score),trust_score=clamp(score),defender_status=defender["status"],defender_method=defender.get("method"),findings=findings)
        cloud=cloud_analyze(q,filename); findings.append(cloud["finding"])
        auth=cloud.get("authenticity_score"); label=cloud.get("label")
        if label: findings.append(f"VeriFYD authenticity verdict: {label}")
        if isinstance(auth,(int,float)): findings.append(f"VeriFYD authenticity score: {int(round(auth))}/100")
        score=clamp(score)
        SCAN_STATE[scan_id].update(status="QUARANTINED",summary=classify(score),security_score=score,authenticity_score=auth,trust_score=score,authenticity_label=label,authenticity_reasoning=cloud.get("reasoning") or "",cloud_status=cloud.get("status"),media_type=cloud.get("media_type"),findings=findings,sha256=sha,size_bytes=q.stat().st_size,defender_status=defender["status"],defender_method=defender.get("method"),recommended_action="release" if score>=80 else "review" if score>=50 else "block")
    except Exception as e:
        if q and q.exists():
            try:q.unlink()
            except:pass
        SCAN_STATE[scan_id].update(status="ERROR",summary="SCAN FAILED",security_score=None,authenticity_score=None,trust_score=None,findings=[f"Lens scan failed: {type(e).__name__}: {e}"])

@app.get("/health")
def health(): return {"ok":True,"product":"VeriFYD Lens","version":"0.4.2","tagline":"Don't Download Blind.","cloud_configured":bool(CLOUD_KEY),"cloud_url":CLOUD_URL,"automatic_scan_api":True}
@app.post("/scan/start")
def start_scan(req:UrlScanRequest):
    url_findings(req.url); sid=str(uuid.uuid4())
    SCAN_STATE[sid]={"scan_id":sid,"status":"QUEUED","summary":"QUEUED","security_score":None,"authenticity_score":None,"trust_score":None,"findings":["VeriFYD Lens accepted the download for protected analysis."],"source":req.url}
    EXECUTOR.submit(scan_worker,sid,req.url); return {"ok":True,"scan_id":sid,"status":"QUEUED"}
@app.get("/scan/{scan_id}")
def get_scan(scan_id:str):
    if scan_id not in SCAN_STATE: raise HTTPException(404,"Scan not found in this agent session.")
    return SCAN_STATE[scan_id]
@app.post("/release/{scan_id}")
def release(scan_id:str):
    rec=SCAN_STATE.get(scan_id)
    if not rec: raise HTTPException(404,"Scan not found.")
    if rec["status"]!="QUARANTINED": raise HTTPException(409,f"File is not quarantined (status={rec['status']}).")
    src=Path(rec["quarantine_path"])
    if not src.exists(): raise HTTPException(404,"Quarantined file no longer exists.")
    dest=unique_destination(DOWNLOADS,rec["filename"]); shutil.move(str(src),str(dest)); rec["status"]="RELEASED"; rec["released_path"]=str(dest)
    return {"ok":True,"status":"RELEASED","filename":rec["filename"],"released_path":str(dest),"sha256":rec.get("sha256")}
@app.post("/delete/{scan_id}")
def delete(scan_id:str):
    rec=SCAN_STATE.get(scan_id)
    if not rec: raise HTTPException(404,"Scan not found.")
    if rec.get("quarantine_path"): Path(rec["quarantine_path"]).unlink(missing_ok=True)
    rec["status"]="DELETED"; return {"ok":True,"status":"DELETED","filename":rec.get("filename")}


# VERIFYD_SAVE_ANYWHERE_V042
from fastapi.responses import FileResponse as _VeriFYDLensFileResponse

@app.get("/release-file/{scan_id}")
def verifyd_release_file(scan_id: str):
    rec = SCAN_STATE.get(scan_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Scan not found in this agent session.")
    if rec.get("status") not in ("QUARANTINED", "RELEASED"):
        raise HTTPException(status_code=409, detail="Protected copy is not ready to release.")

    raw_path = rec.get("quarantine_path")
    if not raw_path:
        raise HTTPException(status_code=404, detail="Quarantined file path is unavailable.")

    p = Path(raw_path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Quarantined file no longer exists.")

    filename = rec.get("filename") or p.name
    return _VeriFYDLensFileResponse(
        path=str(p),
        filename=filename,
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )

@app.post("/release-confirm/{scan_id}")
def verifyd_release_confirm(scan_id: str):
    rec = SCAN_STATE.get(scan_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Scan not found in this agent session.")

    raw_path = rec.get("quarantine_path")
    if raw_path:
        p = Path(raw_path)
        try:
            if p.exists() and p.is_file():
                p.unlink()
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Protected copy was saved, but quarantine cleanup failed: {exc}",
            )

    rec["status"] = "RELEASED"
    rec["released_path"] = "user-selected"
    return {"ok": True, "status": "RELEASED"}

