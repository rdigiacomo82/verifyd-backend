package com.verifyd.circle.calls;

import android.Manifest;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Person;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.AudioAttributes;
import android.media.RingtoneManager;
import android.net.Uri;
import android.os.Build;
import com.verifyd.circle.R;

final class CircleIncomingCallNotification {
    private static final String CHANNEL = "verifyd_circle_calls";
    private static final int ID = 31517;

    private CircleIncomingCallNotification() { }

    static void show(Context context, CircleNativeCallPayload invite) {
        NotificationManager manager = context.getSystemService(NotificationManager.class);
        long remaining = invite.expirySeconds * 1000L - System.currentTimeMillis();
        if (manager == null || remaining <= 0 || !manager.areNotificationsEnabled()) return;
        if (Build.VERSION.SDK_INT >= 33 && context.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) return;
        if (Build.VERSION.SDK_INT >= 26) {
            NotificationChannel channel = new NotificationChannel(CHANNEL,
                    "VeriFYD Circle incoming calls", NotificationManager.IMPORTANCE_HIGH);
            channel.setLockscreenVisibility(Notification.VISIBILITY_PRIVATE);
            channel.setSound(RingtoneManager.getDefaultUri(RingtoneManager.TYPE_RINGTONE),
                    new AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_NOTIFICATION_RINGTONE).build());
            channel.enableVibration(true);
            manager.createNotificationChannel(channel);
        }
        Intent answer = new Intent(context, CircleAnswerActivity.class)
                .setData(Uri.parse("verifyd-circle://answer/" + invite.id))
                .putExtra("call_id", invite.id)
                .putExtra("expires_at", invite.expirySeconds);
        PendingIntent answerAction = PendingIntent.getActivity(context, 0, answer,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        Intent decline = new Intent(context, CircleDeclineReceiver.class)
                .setData(Uri.parse("verifyd-pilot://decline/" + invite.id))
                .putExtra("call_id", invite.id);
        PendingIntent declineAction = PendingIntent.getBroadcast(context, 0, decline,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        Notification.Builder notification = Build.VERSION.SDK_INT >= 26
                ? new Notification.Builder(context, CHANNEL) : new Notification.Builder(context);
        notification.setSmallIcon(android.R.drawable.sym_action_call)
                .setContentTitle("Incoming VeriFYD Circle call")
                .setContentText("Answer or decline this audio call")
                .setVisibility(Notification.VISIBILITY_PRIVATE)
                .setCategory(Notification.CATEGORY_CALL)
                .setPriority(Notification.PRIORITY_MAX)
                .setOngoing(true);
        if (Build.VERSION.SDK_INT >= 31) {
            Person caller = new Person.Builder().setName("VeriFYD caller").setImportant(true).build();
            notification.setStyle(Notification.CallStyle.forIncomingCall(caller, declineAction, answerAction));
        } else {
            notification.addAction(new Notification.Action.Builder(android.R.drawable.sym_action_call,
                    "Answer", answerAction).build());
            notification.addAction(new Notification.Action.Builder(android.R.drawable.ic_menu_close_clear_cancel,
                    "Decline", declineAction).build());
        }
        // The authenticated pilot is supported on Android 8+ so expiry always uses the OS timer.
        if (Build.VERSION.SDK_INT < 26) return;
        notification.setTimeoutAfter(remaining);\n        // CallStyle falls back to heads-up when Android 14+ denies full-screen intent.\n        notification.setFullScreenIntent(answerAction, true);
        manager.notify(invite.id, ID, notification.build());
    }

    static void cancel(Context context, String callId) {
        NotificationManager manager = context.getSystemService(NotificationManager.class);
        if (manager != null) manager.cancel(callId, ID);
    }
}
