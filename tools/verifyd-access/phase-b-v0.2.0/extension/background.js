/* VeriFYD Access — opt-in, current-tab navigation + separate trusted camera window.
 * Sensor data crosses ONLY extension message channels; the visited website
 * never receives frames, derived gaze samples, or camera permissions.
 */
const activeTabs = new Set();
const cameraSessions = new Map(); // tabId -> {windowId, nonce}

async function closeCamera(tabId) {
  const session=cameraSessions.get(tabId);
  cameraSessions.delete(tabId);
  if (!session) return;
  try { await chrome.windows.remove(session.windowId); } catch (_) { /* user closed it */ }
  try { await chrome.tabs.sendMessage(tabId, {type:'VFA_SENSOR_STOPPED'}, {frameId:0}); } catch (_) {}
}
async function toggleOnTab(tab) {
  if (!tab || !Number.isInteger(tab.id)) return;
  if (!/^https?:\/\//i.test(tab.url || '')) {
    console.info('VeriFYD Access: open an ordinary http(s) website first.'); return;
  }
  try {
    await chrome.scripting.insertCSS({target:{tabId:tab.id},files:['content.css']});
    await chrome.scripting.executeScript({
      target:{tabId:tab.id},
      files:['navigation-core.js','gaze-core.js','content.js'],
      world:'ISOLATED',
    });
    activeTabs.add(tab.id);
  } catch (err) {
    console.warn('VeriFYD Access could not be started on this page:',err?.message||'unknown');
  }
}
chrome.action.onClicked.addListener(toggleOnTab);
chrome.commands.onCommand.addListener(async command=>{
  if(command!=='toggle-verifiyd-access')return;
  const tabs=await chrome.tabs.query({active:true,currentWindow:true});
  await toggleOnTab(tabs[0]);
});

chrome.runtime.onMessage.addListener((message,sender,reply)=>{
  if (sender.id!==chrome.runtime.id || !message || typeof message.type!=='string') return false;
  const openFromTab=sender.tab && activeTabs.has(sender.tab.id);
  if (message.type==='VFA_OPEN_CAMERA') {
    (async()=>{
      if(!openFromTab){reply({ok:false,reason:'not_authorized'});return;}
      const tabId=sender.tab.id;
      if(cameraSessions.has(tabId)) {
        reply({ok:true,existing:true}); return;
      }
      const nonce=crypto.randomUUID();
      try {
        const popup=await chrome.windows.create({url:chrome.runtime.getURL(`gaze.html?tab=${tabId}&nonce=${encodeURIComponent(nonce)}`),
          type:'popup',width:390,height:540,focused:true});
        cameraSessions.set(tabId,{nonce,windowId:popup.id});
        reply({ok:true});
      }catch(_){reply({ok:false,reason:'popup_failed'});}
    })();return true;
  }
  if(message.type==='VFA_CLOSE_CAMERA') {
    if(!openFromTab){reply({ok:false});return false;}
    closeCamera(sender.tab.id).then(()=>reply({ok:true}));return true;
  }
  if(['VFA_SENSOR_READY','VFA_SENSOR_SAMPLE','VFA_SENSOR_NO_FACE','VFA_SENSOR_STOPPED'].includes(message.type)) {
    const trustedURL=chrome.runtime.getURL('gaze.html');
    if (!(sender.url||'').startsWith(trustedURL+'?')) {reply({ok:false});return false;}
    const tabId=message.tabId;
    const session=cameraSessions.get(tabId);
    if(!session || session.nonce!==message.nonce || !activeTabs.has(tabId)){reply({ok:false});return false;}
    // The browser page never receives this: only its isolated content script.
    const update={type:message.type};
    if(message.type==='VFA_SENSOR_SAMPLE') {
      const x=message.sample?.x, y=message.sample?.y, blink=message.sample?.blink;
      if(!Number.isFinite(x)||!Number.isFinite(y)||typeof blink!=='boolean') {reply({ok:false});return false;}
      update.sample={x:Math.max(-.3,Math.min(1.3,x)),y:Math.max(-.3,Math.min(1.3,y)),blink};
    }
    chrome.tabs.sendMessage(tabId,update,{frameId:0}).then(()=>reply({ok:true})).catch(()=>reply({ok:false}));
    return true;
  }
  return false;
});
chrome.windows.onRemoved.addListener(windowId=>{
  for(const [tabId,session] of cameraSessions) {
    if(session.windowId===windowId){
      cameraSessions.delete(tabId);
      chrome.tabs.sendMessage(tabId,{type:'VFA_SENSOR_STOPPED'},{frameId:0}).catch(()=>{});
    }
  }
});
chrome.tabs.onRemoved.addListener(tabId=>{activeTabs.delete(tabId);closeCamera(tabId);});
chrome.tabs.onUpdated.addListener((tabId,change)=>{if(change.status==='loading') {activeTabs.delete(tabId);closeCamera(tabId);}});
