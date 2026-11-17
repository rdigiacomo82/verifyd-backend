package com.verifyd.circle.calls;

import com.google.firebase.messaging.FirebaseMessagingService;
import com.google.firebase.messaging.RemoteMessage;

/** Inbound data-only FCM handler; never requires the browser to be in the foreground. */
public final class CircleIncomingCallMessagingService extends FirebaseMessagingService {
    @Override public void onNewToken(String token) {
        CircleCallApi api = new CircleCallApi(this);
        if (!api.enrolled()) return;
        new Thread(() -> {
            try { api.refresh(token); }
            catch (Exception ignored) {
                // Retry on the next authenticated launch; never log a device token.
            }
        }, "verifyd-circle-token-refresh").start();
    }

    @Override public void onMessageReceived(RemoteMessage message) {
        CircleNativeCallPayload invite = CircleNativeCallPayload.parse(
                message.getData(), System.currentTimeMillis() / 1000L);
        if (invite == null) return;
        // Audio is backed by a native WebRTC media adapter; video continues to
        // use existing authenticated browser/web-push UX until video native media is integrated.
        if ("audio".equals(invite.type) && new CircleCallApi(this).enrolled())
            CircleIncomingCallNotification.show(this, invite);
    }
}
