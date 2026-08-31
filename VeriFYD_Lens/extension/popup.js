async function render(){const{lastResult}=await chrome.storage.local.get("lastResult");if(!lastResult)return;
document.getElementById("status").textContent=lastResult.summary||lastResult.status||"Ready";
document.getElementById("score").textContent=Number.isFinite(lastResult.trust_score)?`${lastResult.trust_score}/100`:"";
document.getElementById("details").textContent=(lastResult.findings||[]).join(" • ")||"No additional findings.";}
document.getElementById("open").addEventListener("click",()=>chrome.tabs.create({url:chrome.runtime.getURL("result.html")}));render();