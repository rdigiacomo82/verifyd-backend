package com.verifyd.circle.calls;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;

public final class CircleDeclineReceiver extends BroadcastReceiver {
    @Override public void onReceive(Context context, Intent intent) {
        String callId = intent.getStringExtra("call_id");
        CircleCallApi api = new CircleCallApi(context);
        if (!CircleNativeCallPayload.validId(callId) || !api.enrolled()) return;
        PendingResult pending = goAsync();
        new Thread(() -> {
            try {
                api.decline(callId);
                CircleIncomingCallNotification.cancel(context, callId);
            } catch (CircleCallApi.ApiError error) {
                if (error.status == 404 || error.status == 409 || error.status == 410) {
                    CircleIncomingCallNotification.cancel(context, callId);
                }
            } catch (Exception ignored) {
                // Leave the notification until expiry so a transient error can be retried.
            } finally {
                pending.finish();
            }
        }).start();
    }
}
