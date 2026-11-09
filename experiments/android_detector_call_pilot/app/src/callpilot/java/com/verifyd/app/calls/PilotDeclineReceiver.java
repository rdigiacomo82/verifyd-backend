package com.verifyd.app.calls;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;

public final class PilotDeclineReceiver extends BroadcastReceiver {
    @Override public void onReceive(Context context, Intent intent) {
        String callId = intent.getStringExtra("call_id");
        PilotCallApi api = new PilotCallApi(context);
        if (!NativeCallPayload.validId(callId) || !api.enrolled()) return;
        PendingResult pending = goAsync();
        new Thread(() -> {
            try {
                api.decline(callId);
                PilotIncomingCallNotification.cancel(context, callId);
            } catch (PilotCallApi.ApiError error) {
                if (error.status == 404 || error.status == 409 || error.status == 410) {
                    PilotIncomingCallNotification.cancel(context, callId);
                }
            } catch (Exception ignored) {
                // Leave the notification until expiry so a transient error can be retried.
            } finally {
                pending.finish();
            }
        }).start();
    }
}
