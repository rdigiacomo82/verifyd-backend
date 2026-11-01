// Pure mocks: confirm extension-on-demand permission boundary and derived-only sensor relay.
const assert=require('assert');
const fs=require('fs');
const vm=require('vm');
const listeners={};const sent=[];const injected=[];const popups=[];
const event=(name)=>({addListener:(fn)=>{listeners[name]=fn;}});
const chrome={
  runtime:{id:'trusted-ext',getURL:(path)=>`chrome-extension://trusted-ext/${path}`,onMessage:event('message')},
  action:{onClicked:event('clicked')},commands:{onCommand:event('command')},
  scripting:{insertCSS:async(info)=>injected.push(['css',info]),executeScript:async(info)=>injected.push(['js',info])},
  windows:{create:async(opts)=>{popups.push(opts);return {id:117}},remove:async(id)=>{assert.equal(id,117)} ,onRemoved:event('windowRemoved')},
  tabs:{query:async()=>[],sendMessage:async(tabId,payload)=>{sent.push({tabId,payload});return{};},
    onUpdated:event('updated'),onRemoved:event('tabRemoved')},
};
vm.runInNewContext(fs.readFileSync(require('path').join(__dirname,'../extension/background.js'),'utf8'),
  {chrome,crypto:{randomUUID:()=> 'test-unique-nonce-abcdef-123456789'},console});
const message=(payload,sender)=>new Promise(resolve=>{
  const immediate=listeners.message(payload,sender,(result)=>resolve(result));
  if(immediate===false)resolve({ok:false});
});
(async()=>{
  let result=await message({type:'VFA_OPEN_CAMERA'},{id:'trusted-ext',tab:{id:41}});
  assert.equal(result.ok,false,'No authorization before user starts the extension');
  await listeners.clicked({id:41,url:'https://example.test/'});
  assert.equal(injected.length,2);
  assert.equal(injected[1][1].world,'ISOLATED');
  assert.ok(injected[1][1].files.includes('gaze-core.js'));
  result=await message({type:'VFA_OPEN_CAMERA'},{id:'trusted-ext',tab:{id:41}});
  assert.equal(result.ok,true);
  assert.ok(popups[0].url.startsWith('chrome-extension://trusted-ext/gaze.html?'));
  const nonce=new URL(popups[0].url).searchParams.get('nonce');
  const sender={id:'trusted-ext',url:popups[0].url};
  result=await message({type:'VFA_SENSOR_SAMPLE',tabId:41,nonce:'wrong',sample:{x:.3,y:.5,blink:false}},sender);
  assert.equal(result.ok,false);assert.equal(sent.length,0);
  result=await message({type:'VFA_SENSOR_SAMPLE',tabId:41,nonce,sample:{x:.3,y:.5,blink:false}},
    {id:'trusted-ext',url:'https://evil.test/?tab=41'});
  assert.equal(result.ok,false);assert.equal(sent.length,0);
  result=await message({type:'VFA_SENSOR_SAMPLE',tabId:41,nonce,sample:{x:.3,y:.5,blink:false}},sender);
  assert.equal(result.ok,true);
  assert.equal(sent[0].tabId,41);
  assert.deepEqual(JSON.parse(JSON.stringify(sent[0].payload)),{type:'VFA_SENSOR_SAMPLE',sample:{x:.3,y:.5,blink:false}});
  assert.ok(!('frame' in sent[0].payload));
  result=await message({type:'VFA_CLOSE_CAMERA'},{id:'trusted-ext',tab:{id:41}});
  assert.equal(result.ok,true);
  assert.equal(sent.at(-1).payload.type,'VFA_SENSOR_STOPPED');
  console.log('PASS extension sensor relay, isolated-world authorization, nonce, camera closure');
})().catch(error=>{console.error(error);process.exit(1)});
