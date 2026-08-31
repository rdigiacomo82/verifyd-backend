const AGENT="http://127.0.0.1:8765";
async function ensureMenu(){try{await chrome.contextMenus.removeAll();}catch(_){}
 chrome.contextMenus.create({id:"verifyd-quarantine-scan",title:"Quarantine & analyze with VeriFYD Lens",contexts:["link"]});}
chrome.runtime.onInstalled.addListener(ensureMenu); chrome.runtime.onStartup.addListener(ensureMenu);
chrome.contextMenus.onClicked.addListener(async(info)=>{
 if(info.menuItemId!=="verifyd-quarantine-scan"||!info.linkUrl)return;
 await chrome.storage.local.set({lastResult:{status:"SCANNING",summary:"SCANNING",source:info.linkUrl,
 findings:["Downloading to quarantine, scanning security, then running VeriFYD authenticity analysis..."]}});
 await chrome.tabs.create({url:chrome.runtime.getURL("result.html")});
 try{
  const r=await fetch(`${AGENT}/download-and-scan`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({url:info.linkUrl})});
  const d=await r.json(); if(!r.ok)throw new Error(d.detail||`Lens agent returned HTTP ${r.status}`);
  await chrome.storage.local.set({lastResult:d});
 }catch(e){await chrome.storage.local.set({lastResult:{status:"ERROR",summary:"SCAN FAILED",source:info.linkUrl,
 findings:[String(e&&e.message?e.message:e)]}});}
});