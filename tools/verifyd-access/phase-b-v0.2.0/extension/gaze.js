/* VeriFYD Access: camera capture runs in trusted chrome-extension:// page.
 * Only [normalized iris position, blink boolean] are sent by extension runtime
 * to the isolated browser controller. Frames never leave this camera page.
 */
(async function () {
  'use strict';
  const gaze=globalThis.__verifydAccessGazeCore;
  const query=new URLSearchParams(location.search);
  const tabId=Number(query.get('tab'));
  const nonce=query.get('nonce')||'';
  const status=document.querySelector('#status');
  const video=document.querySelector('#preview');
  const start=document.querySelector('#start');
  const stop=document.querySelector('#stop');
  const done=document.querySelector('#done');
  let model=null,stream=null,generation=0,lastVideoTime=-1,lastFaceAt=0,timer=null;
  if(!Number.isSafeInteger(tabId)||tabId<0||nonce.length<20){status.textContent='Invalid camera session. Close and reopen from the browser panel.';start.disabled=true;return;}
  const notify=(type,extra={})=>chrome.runtime.sendMessage({type,tabId,nonce,...extra}).catch(()=>{});
  const info=(text)=>{status.textContent=text;};
  function cleanStop(note='Camera off. No video is being captured.') {
    generation++;
    if(timer)clearTimeout(timer);timer=null;
    if(stream){stream.getTracks().forEach(track=>track.stop());stream=null;}
    try{video.pause();}catch(_){}
    video.srcObject=null;video.hidden=true;start.disabled=false;stop.disabled=true;
    info(note);notify('VFA_SENSOR_STOPPED');
  }
  async function begin() {
    if(stream||start.disabled)return;
    const life=++generation;
    start.disabled=true;stop.disabled=false;
    info('Loading locally installed Eye Gaze model…');
    try {
      if(!model) {
        const url=chrome.runtime.getURL('vendor/');
        const check=await fetch(url+'vision_bundle.mjs');
        if(!check.ok)throw Error('assets_missing');
        const {FilesetResolver,FaceLandmarker}=await import(url+'vision_bundle.mjs');
        const runtime=await FilesetResolver.forVisionTasks(url+'wasm');
        model=await FaceLandmarker.createFromOptions(runtime,{
          baseOptions:{modelAssetPath:url+'face_landmarker.task',delegate:'CPU'},
          runningMode:'VIDEO',numFaces:1,outputFaceBlendshapes:true,
          minFaceDetectionConfidence:.5,minTrackingConfidence:.5,
        });
      }
      if(life!==generation)return;
      info('Model ready. Granting camera access only to VeriFYD Access…');
      const media=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:'user',width:{ideal:640},height:{ideal:480},frameRate:{ideal:15,max:30}}});
      if(life!==generation){media.getTracks().forEach(t=>t.stop());return;}
      stream=media;video.srcObject=stream;video.hidden=false;
      await video.play();
      lastVideoTime=-1;lastFaceAt=performance.now();
      info('Camera on. Look at the website and select Calibrate gaze there.');
      await notify('VFA_SENSOR_READY');
      tick(life);
    }catch(error){
      if(life!==generation)return;
      const kind=String(error?.name||error?.message||'');
      cleanStop(/assets_missing|vision_bundle|fetch|wasm|model/i.test(kind)
        ?'MediaPipe assets are not installed. Run the bundled asset installer, reload the extension, and reopen this window.'
        : /NotAllowed|PermissionDenied/.test(kind)
          ?'Camera permission denied. You can still use VeriFYD switch and keyboard controls.'
          : `Camera could not start (${kind.slice(0,55)}). Try another camera or restart the extension.`);
    }
  }
  function tick(life) {
    if(life!==generation||!stream)return;
    if(document.hidden){cleanStop('Camera paused because this window is hidden.');return;}
    const now=performance.now();
    if(video.readyState>=2&&video.currentTime!==lastVideoTime){
      lastVideoTime=video.currentTime;
      try{
        const result=model.detectForVideo(video,now);
        const sample=gaze.sampleLandmarks(result.faceLandmarks?.[0],result.faceBlendshapes?.[0]?.categories||[]);
        if(sample){lastFaceAt=now;notify('VFA_SENSOR_SAMPLE',{sample:{x:sample.x,y:sample.y,blink:sample.blink}});}
        else if(now-lastFaceAt>2000){lastFaceAt=now;notify('VFA_SENSOR_NO_FACE');}
      }catch(_){cleanStop('Camera processing stopped. Camera off. Restart or use switch controls.');return;}
    }
    timer=setTimeout(()=>tick(life),120);
  }
  start.addEventListener('click',begin);
  stop.addEventListener('click',()=>cleanStop());
  done.addEventListener('click',()=>window.close());
  window.addEventListener('beforeunload',()=>{generation++;if(stream)stream.getTracks().forEach(t=>t.stop());});
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&stream)cleanStop('Camera stopped because this window is hidden.');});
})();
