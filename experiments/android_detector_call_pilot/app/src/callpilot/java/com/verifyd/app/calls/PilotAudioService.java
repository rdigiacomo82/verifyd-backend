package com.verifyd.app.calls;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.media.AudioManager;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import com.verifyd.app.R;

public final class PilotAudioService extends Service {
    private static final String HANG_UP = "com.verifyd.app.pilot.HANG_UP";
    private static final int NOTIFICATION_ID = 31518;
    private static final String CHANNEL = "verifyd_detector_active_pilot";
    private HandlerThread worker;
    private Handler executor;
    private PilotWebRtc session;
    private String callId;
    private AudioManager audio;
    private int previousMode;
    private boolean ownsAudioMode;

    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent == null) {
            stopSelf();
            return START_NOT_STICKY;
        }
        if (HANG_UP.equals(intent.getAction())) {
            if (callId != null && callId.equals(intent.getStringExtra("call_id"))) stopSelf();
            return START_NOT_STICKY;
        }
        if (callId != null) return START_NOT_STICKY;
        String requested = intent.getStringExtra("call_id");
        if (!NativeCallPayload.validId(requested)
                || intent.getLongExtra("expires_at", 0) <= System.currentTimeMillis() / 1000) {
            stopSelf();
            return START_NOT_STICKY;
        }
        callId = requested;
        try {
            startForeground(NOTIFICATION_ID, notification("Connecting audio…"));
            PilotIncomingCallNotification.cancel(this, callId);
            audio = getSystemService(AudioManager.class);
            previousMode = audio.getMode();
            audio.setMode(AudioManager.MODE_IN_COMMUNICATION);
            ownsAudioMode = true;
            worker = new HandlerThread("verifyd-pilot-audio");
            worker.start();
            executor = new Handler(worker.getLooper());
            session = new PilotWebRtc(getApplicationContext(), executor, callId,
                    () -> new Handler(getMainLooper()).post(() -> {
                        if (callId != null) getSystemService(NotificationManager.class)
                                .notify(NOTIFICATION_ID, notification("Audio call in progress"));
                    }),
                    () -> new Handler(getMainLooper()).post(this::stopSelf));
            executor.post(session::connect);
        } catch (RuntimeException ignored) {
            stopSelf();
        }
        return START_NOT_STICKY;
    }

    private Notification notification(String text) {
        NotificationManager manager = getSystemService(NotificationManager.class);
        manager.createNotificationChannel(new NotificationChannel(CHANNEL, "VeriFYD active audio call",
                NotificationManager.IMPORTANCE_LOW));
        Intent hangUp = new Intent(this, PilotAudioService.class).setAction(HANG_UP)
                .putExtra("call_id", callId);
        PendingIntent action = PendingIntent.getService(this, 0, hangUp,
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
        return new Notification.Builder(this, CHANNEL).setSmallIcon(R.drawable.ic_stat_call)
                .setContentTitle("VeriFYD pilot call").setContentText(text)
                .setCategory(Notification.CATEGORY_CALL).setOngoing(true)
                .addAction(new Notification.Action.Builder(
                        android.R.drawable.ic_menu_close_clear_cancel, "Hang up", action).build()).build();
    }

    @Override public void onDestroy() {
        callId = null;
        if (executor != null && session != null) executor.post(session::close);
        if (worker != null) worker.quitSafely();
        if (ownsAudioMode && audio.getMode() == AudioManager.MODE_IN_COMMUNICATION) audio.setMode(previousMode);
        stopForeground(STOP_FOREGROUND_REMOVE);
        super.onDestroy();
    }

    @Override public IBinder onBind(Intent intent) { return null; }
}
