package com.verifyd.circle.calls;

import android.Manifest;
import android.app.NotificationManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import androidx.browser.customtabs.CustomTabsService;
import androidx.browser.customtabs.CustomTabsSession;
import com.google.firebase.messaging.FirebaseMessaging;
import org.json.JSONObject;

/**
 * Attach this adapter to the *existing* production Circle TWA's CustomTabsSession.
 * Its CustomTabsCallback must forward validation/navigation/channel/message events.
 * NEVER send a Trust Voice web bearer to this class or native app.
 */
public final class CircleTwaBridge {
    public static final Uri ORIGIN = Uri.parse("https://vfvid.com");
    private static final String SCHEMA = "verifyd.twa.native-bridge.v1";
    private final Context context;
    private final Handler main = new Handler(Looper.getMainLooper());
    private final CircleCallApi api;
    private CustomTabsSession session;
    private boolean validated;
    private boolean channelReady;

    public CircleTwaBridge(Context context) {
        this.context = context.getApplicationContext();
        api = new CircleCallApi(this.context);
    }

    public void attach(CustomTabsSession session) {
        this.session = session;
        validated = false;
        channelReady = false;
    }

    public void onRelationshipValidationResult(int relation, Uri requestedOrigin, boolean result) {
        validated = result && relation == CustomTabsService.RELATION_USE_AS_ORIGIN
                && ORIGIN.equals(requestedOrigin);
        if (!validated) channelReady = false;
    }

    public void onNavigationFinished() {
        if (validated && session != null)
            session.requestPostMessageChannel(ORIGIN, ORIGIN, new Bundle());
    }

    public void onMessageChannelReady() {
        if (!validated || session == null) return;
        channelReady = true;
        try { send(api.bridgeHello()); }
        catch (Exception ignored) { channelReady = false; }
    }

    public void onPostMessage(String message) {
        if (!validated || !channelReady || message == null || message.length() > 3072) return;
        try {
            JSONObject data = new JSONObject(message);
            if (!SCHEMA.equals(data.optString("schema"))) return;
            String requestID = data.optString("request_id");
            String action = data.optString("type");
            if (!requestID.matches("req_[0-9a-f]{24}")) return;
            if ("status".equals(action)) {
                sendResult(action, requestID, api.enrolled(), "status");
            } else if ("disable".equals(action)) {
                new Thread(() -> {
                    try {
                        api.unlink();
                        sendResult(action, requestID, false, "disabled");
                    } catch (Exception ignored) {
                        sendResult(action, requestID, api.enrolled(), "error");
                    }
                }, "verifyd-circle-native-disable").start();
            } else if ("redeem".equals(action)) {
                String intent = data.optString("intent_id");
                String challenge = data.optString("challenge");
                if (!intent.matches("nbi_[A-Za-z0-9_-]{32}")
                        || !challenge.matches("[A-Za-z0-9_-]{43}")) return;
                if (!notificationsAllowed()) {
                    context.startActivity(new Intent(context, CircleNotificationsPermissionActivity.class)
                            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK));
                    sendResult(action, requestID, false, "permission_required");
                    return;
                }
                // The user already authorized this via the authenticated
                // web page's enable button; only the signed intent crosses the port.
                FirebaseMessaging.getInstance().getToken().addOnCompleteListener(task -> {
                    if (!task.isSuccessful() || task.getResult() == null) {
                        sendResult(action, requestID, false, "fcm_unavailable");
                        return;
                    }
                    String fcmToken = task.getResult();
                    new Thread(() -> {
                        try {
                            api.redeemIntent(intent, challenge, fcmToken);
                            sendResult(action, requestID, api.enrolled(), "registered");
                        } catch (Exception ignored) {
                            sendResult(action, requestID, false, "server_unavailable");
                        }
                    }, "verifyd-circle-native-redeem").start();
                });
            }
        } catch (Exception ignored) {
            // Never log arbitrary web messages or credentials.
        }
    }

    private boolean notificationsAllowed() {
        NotificationManager manager = context.getSystemService(NotificationManager.class);
        if (manager == null || !manager.areNotificationsEnabled()) return false;
        return Build.VERSION.SDK_INT < 33 ||
                context.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
                        == PackageManager.PERMISSION_GRANTED;
    }

    private void sendResult(String action, String requestID, boolean enabled, String phase) {
        try {
            JSONObject result = new JSONObject()
                    .put("schema", SCHEMA).put("type", "result")
                    .put("request_id", requestID).put("action", action)
                    .put("enabled", enabled).put("registered", enabled)
                    .put("phase", phase);
            send(result);
        } catch (Exception ignored) { }
    }

    private void send(JSONObject message) {
        main.post(() -> {
            if (validated && channelReady && session != null)
                session.postMessage(message.toString(), null);
        });
    }
}
