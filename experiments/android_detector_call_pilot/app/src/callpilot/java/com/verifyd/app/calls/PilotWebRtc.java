package com.verifyd.app.calls;

import android.content.Context;
import android.os.Handler;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.WebSocket;
import okhttp3.WebSocketListener;
import org.json.JSONException;
import org.json.JSONObject;
import org.webrtc.AudioSource;
import org.webrtc.AudioTrack;
import org.webrtc.DataChannel;
import org.webrtc.IceCandidate;
import org.webrtc.MediaConstraints;
import org.webrtc.MediaStream;
import org.webrtc.PeerConnection;
import org.webrtc.PeerConnectionFactory;
import org.webrtc.RtpReceiver;
import org.webrtc.SdpObserver;
import org.webrtc.SessionDescription;
import org.webrtc.audio.JavaAudioDeviceModule;

/** Audio-only callee using the existing Trust Voice browser offer/answer protocol. */
final class PilotWebRtc {
    private static final int MAX_QUEUED_CANDIDATES = 128;
    private final Context context;
    private final Handler worker;
    private final String callId;
    private final Runnable connected;
    private final Runnable ended;
    private final Runnable timeout = this::finish;
    private final List<IceCandidate> remoteIce = new ArrayList<>();
    private final List<IceCandidate> localIce = new ArrayList<>();
    private final OkHttpClient client = new OkHttpClient.Builder()
            .pingInterval(10, TimeUnit.SECONDS).followRedirects(false)
            .followSslRedirects(false).build();
    private WebSocket socket;
    private PeerConnectionFactory factory;
    private AudioSource source;
    private AudioTrack microphone;
    private PeerConnection peer;
    private boolean offerReceived;
    private boolean remoteDescriptionReady;
    private boolean answerSent;
    private boolean closed;

    PilotWebRtc(Context context, Handler worker, String callId, Runnable connected, Runnable ended) {
        this.context = context;
        this.worker = worker;
        this.callId = callId;
        this.connected = connected;
        this.ended = ended;
    }

    void connect() {
        try {
            PilotCallApi api = new PilotCallApi(context);
            PeerConnectionFactory.initialize(PeerConnectionFactory.InitializationOptions
                    .builder(context).createInitializationOptions());
            JavaAudioDeviceModule audioModule = JavaAudioDeviceModule.builder(context)
                    .createAudioDeviceModule();
            try {
                factory = PeerConnectionFactory.builder().setAudioDeviceModule(audioModule)
                        .createPeerConnectionFactory();
            } finally {
                audioModule.release();
            }

            PeerConnection.RTCConfiguration configuration = new PeerConnection.RTCConfiguration(
                    Collections.singletonList(PeerConnection.IceServer.builder(
                            "stun:stun.l.google.com:19302").createIceServer()));
            configuration.sdpSemantics = PeerConnection.SdpSemantics.UNIFIED_PLAN;
            peer = factory.createPeerConnection(configuration, new PeerEvents());
            if (peer == null) throw new IllegalStateException("Unable to create the native peer");
            source = factory.createAudioSource(new MediaConstraints());
            microphone = factory.createAudioTrack("verifyd-pilot-mic", source);
            microphone.setEnabled(true);
            if (peer.addTrack(microphone, Collections.singletonList("verifyd-native-audio")) == null) {
                throw new IllegalStateException("Unable to add microphone track");
            }

            Request request = new Request.Builder().url(api.mediaUrl(callId))
                    .header("Authorization", "Bearer " + api.deviceToken()).build();
            socket = client.newWebSocket(request, new SocketEvents());
            worker.postDelayed(timeout, 30_000);
        } catch (Exception ignored) {
            finish();
        }
    }

    private final class SocketEvents extends WebSocketListener {
        @Override public void onOpen(WebSocket webSocket, Response response) {
            worker.post(() -> send("call_answer", null, null));
        }

        @Override public void onMessage(WebSocket webSocket, String text) {
            worker.post(() -> handleMessage(text));
        }

        @Override public void onClosing(WebSocket webSocket, int code, String reason) {
            webSocket.close(1000, null);
            worker.post(PilotWebRtc.this::finish);
        }

        @Override public void onFailure(WebSocket webSocket, Throwable error, Response response) {
            worker.post(PilotWebRtc.this::finish);
        }
    }

    private void handleMessage(String text) {
        if (closed) return;
        try {
            JSONObject message = new JSONObject(text);
            if (!callId.equals(message.optString("call_id"))) return;
            switch (message.getString("type")) {
                case "webrtc_offer":
                    if (offerReceived) return;
                    offerReceived = true;
                    JSONObject sdp = message.getJSONObject("sdp");
                    if (!"offer".equals(sdp.getString("type"))) {
                        finish();
                        return;
                    }
                    acceptOffer(sdp.getString("sdp"));
                    break;
                case "webrtc_ice":
                    JSONObject candidate = message.getJSONObject("candidate");
                    IceCandidate item = new IceCandidate(candidate.optString("sdpMid", null),
                            candidate.getInt("sdpMLineIndex"), candidate.getString("candidate"));
                    if (remoteDescriptionReady) {
                        peer.addIceCandidate(item);
                    } else if (remoteIce.size() < MAX_QUEUED_CANDIDATES) {
                        remoteIce.add(item);
                    }
                    break;
                case "call_ended":
                    finish();
                    break;
                default:
                    break;
            }
        } catch (JSONException | RuntimeException ignored) {
            finish();
        }
    }

    private void acceptOffer(String sdp) {
        peer.setRemoteDescription(new SdpCallback() {
            @Override public void onSetSuccess() {
                worker.post(() -> {
                    if (closed) return;
                    remoteDescriptionReady = true;
                    for (IceCandidate item : remoteIce) peer.addIceCandidate(item);
                    remoteIce.clear();
                    peer.createAnswer(new SdpCallback() {
                        @Override public void onCreateSuccess(SessionDescription description) {
                            worker.post(() -> setLocalAnswer(description));
                        }
                    }, new MediaConstraints());
                });
            }
        }, new SessionDescription(SessionDescription.Type.OFFER, sdp));
    }

    private void setLocalAnswer(SessionDescription description) {
        if (closed) return;
        peer.setLocalDescription(new SdpCallback() {
            @Override public void onSetSuccess() {
                worker.post(() -> {
                    if (closed) return;
                    try {
                        JSONObject sdp = new JSONObject().put("type", "answer")
                                .put("sdp", description.description);
                        send("webrtc_answer", "sdp", sdp);
                        answerSent = true;
                        for (IceCandidate item : localIce) sendIce(item);
                        localIce.clear();
                    } catch (JSONException ignored) {
                        finish();
                    }
                });
            }
        }, description);
    }

    private void sendIce(IceCandidate item) {
        try {
            JSONObject json = new JSONObject().put("candidate", item.sdp)
                    .put("sdpMid", item.sdpMid).put("sdpMLineIndex", item.sdpMLineIndex);
            send("webrtc_ice", "candidate", json);
        } catch (JSONException ignored) {
            finish();
        }
    }

    private void send(String kind, String field, JSONObject value) {
        if (closed || socket == null) return;
        try {
            JSONObject message = new JSONObject().put("type", kind);
            if (field != null) message.put(field, value);
            if (!socket.send(message.toString())) finish();
        } catch (JSONException ignored) {
            finish();
        }
    }

    private void finish() {
        if (closed) return;
        close();
        ended.run();
    }

    void close() {
        if (closed) return;
        closed = true;
        worker.removeCallbacks(timeout);
        if (socket != null) {
            socket.send("{\"type\":\"call_end\"}");
            socket.close(1000, null);
        }
        if (peer != null) peer.dispose();
        if (microphone != null) microphone.dispose();
        if (source != null) source.dispose();
        if (factory != null) factory.dispose();
        client.connectionPool().evictAll();
        client.dispatcher().executorService().shutdown();
    }

    private class SdpCallback implements SdpObserver {
        @Override public void onCreateSuccess(SessionDescription description) { }
        @Override public void onSetSuccess() { }
        @Override public void onCreateFailure(String error) { worker.post(PilotWebRtc.this::finish); }
        @Override public void onSetFailure(String error) { worker.post(PilotWebRtc.this::finish); }
    }

    private final class PeerEvents implements PeerConnection.Observer {
        @Override public void onSignalingChange(PeerConnection.SignalingState state) { }
        @Override public void onIceConnectionChange(PeerConnection.IceConnectionState state) { }
        @Override public void onIceConnectionReceivingChange(boolean receiving) { }
        @Override public void onIceGatheringChange(PeerConnection.IceGatheringState state) { }
        @Override public void onIceCandidatesRemoved(IceCandidate[] candidates) { }
        @Override public void onAddStream(MediaStream stream) { }
        @Override public void onRemoveStream(MediaStream stream) { }
        @Override public void onDataChannel(DataChannel channel) { }
        @Override public void onRenegotiationNeeded() { }
        @Override public void onAddTrack(RtpReceiver receiver, MediaStream[] streams) { }

        @Override public void onIceCandidate(IceCandidate candidate) {
            worker.post(() -> {
                if (closed) return;
                if (answerSent) sendIce(candidate);
                else if (localIce.size() < MAX_QUEUED_CANDIDATES) localIce.add(candidate);
            });
        }

        @Override public void onConnectionChange(PeerConnection.PeerConnectionState state) {
            worker.post(() -> {
                if (closed) return;
                if (state == PeerConnection.PeerConnectionState.CONNECTED) {
                    worker.removeCallbacks(timeout);
                    connected.run();
                } else if (state == PeerConnection.PeerConnectionState.DISCONNECTED) {
                    worker.removeCallbacks(timeout);
                    worker.postDelayed(timeout, 15_000);
                } else if (state == PeerConnection.PeerConnectionState.FAILED) {
                    finish();
                }
            });
        }
    }
}
