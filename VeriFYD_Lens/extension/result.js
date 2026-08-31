const AGENT="http://127.0.0.1:8765";let current=null;function score(v){return Number.isFinite(v)?`${Math.round(v)}/100`:"Not analyzed";}
async function getResult(){const{lastResult}=await chrome.storage.local.get("lastResult");return lastResult||null;}
async function render(){current=await getResult();if(!current)return;
document.getElementById("summary").textContent=current.summary||current.status||"RESULT";
document.getElementById("score").textContent=Number.isFinite(current.trust_score)?`${Math.round(current.trust_score)}/100`:"";
document.getElementById("security").textContent=score(current.security_score);document.getElementById("auth").textContent=score(current.authenticity_score);document.getElementById("trust").textContent=score(current.trust_score);
document.getElementById("verdict").textContent=current.authenticity_label||"Not analyzed";
document.getElementById("reason").textContent=current.authenticity_reasoning||"No VeriFYD reasoning returned.";
document.getElementById("source").textContent=current.source||"—";const f=document.getElementById("findings");f.textContent="";
for(const item of(current.findings||["No findings yet."])){const d=document.createElement("div");d.className="finding";d.textContent=item;f.appendChild(d);}
const bits=[];if(current.filename)bits.push(`Name: ${current.filename}`);if(Number.isFinite(current.size_bytes))bits.push(`Size: ${current.size_bytes.toLocaleString()} bytes`);
if(current.sha256)bits.push(`SHA-256: ${current.sha256}`);if(current.status)bits.push(`Status: ${current.status}`);if(current.media_type)bits.push(`Media type: ${current.media_type}`);
document.getElementById("file").textContent=bits.join("\n")||"Waiting for scan…";const q=current.status==="QUARANTINED"&&current.scan_id;document.getElementById("release").disabled=!q;document.getElementById("delete").disabled=!q;}
chrome.storage.onChanged.addListener((c,a)=>{if(a==="local"&&c.lastResult)render();});
document.getElementById("release").addEventListener("click",async()=>{if(!current?.scan_id)return;document.getElementById("msg").textContent="Releasing…";
try{const r=await fetch(`${AGENT}/release/${current.scan_id}`,{method:"POST"});const d=await r.json();if(!r.ok)throw new Error(d.detail||"Release failed");current.status="RELEASED";
await chrome.storage.local.set({lastResult:current});document.getElementById("msg").textContent=`Released to: ${d.released_path}`;render();}catch(e){document.getElementById("msg").textContent=String(e.message||e);}});
document.getElementById("delete").addEventListener("click",async()=>{if(!current?.scan_id)return;document.getElementById("msg").textContent="Deleting…";
try{const r=await fetch(`${AGENT}/delete/${current.scan_id}`,{method:"POST"});const d=await r.json();if(!r.ok)throw new Error(d.detail||"Delete failed");current.status="DELETED";
await chrome.storage.local.set({lastResult:current});document.getElementById("msg").textContent="Quarantined file deleted.";render();}catch(e){document.getElementById("msg").textContent=String(e.message||e);}});render();