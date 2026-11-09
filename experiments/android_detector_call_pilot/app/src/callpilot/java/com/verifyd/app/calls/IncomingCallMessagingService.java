package com.verifyd.app.calls;

import com.google.firebase.messaging.FirebaseMessagingService;
import com.google.firebase.messaging.RemoteMessage;
import com.verifyd.app.BuildConfig;

public final class IncomingCallMessagingService extends FirebaseMessagingService {
    @Override public void onNewToken(String token) {
        PilotCallApi api = new PilotCallApi(this);
        if (!api.enrolled()) return;
        try {
            api.refresh(token);
        } catch (Exception ignored) {
            // Opening the pilot setup again retries with the latest Firebase token.
        }
    }

    @Override public void onMessageReceived(RemoteMessage message) {
        NativeCallPayload invite = NativeCallPayload.parse(message.getData(), System.currentTimeMillis() / 1000L);
        if (invite == null) return;
        if ("audio".equals(invite.type) && new PilotCallApi(this).enrolled()) {
            PilotIncomingCallNotification.show(this, invite);
        } else if (BuildConfig.NATIVE_CALL_PREVIEW_ENABLED) {
            IncomingCallNotifications.show(this, invite);
        }
    }
}
