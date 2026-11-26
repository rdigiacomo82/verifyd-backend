package com.verifyd.circle.calls;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

/** Ringing UI does NOT auto-answer when the screen wakes. */
public final class CircleAnswerActivity extends Activity {
    private TextView status;
    private String callId;
    private long expiry;
    private boolean answering;

    @Override protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        if (Build.VERSION.SDK_INT >= 27) {
            setShowWhenLocked(true);
            setTurnScreenOn(true);
        } else {
            getWindow().addFlags(WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED
                    | WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON);
        }
        callId = getIntent().getStringExtra("call_id");
        expiry = getIntent().getLongExtra("expires_at", 0);
        if (!isActive()) { finish(); return; }

        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        int pad = Math.round(getResources().getDisplayMetrics().density * 24);
        layout.setPadding(pad, pad, pad, pad);
        status = new TextView(this);
        status.setText("Incoming VeriFYD Circle audio call");
        status.setTextSize(22);
        layout.addView(status);
        Button answer = new Button(this);
        answer.setText("Answer");
        answer.setMinHeight(Math.round(getResources().getDisplayMetrics().density * 48));
        answer.setOnClickListener(view -> answer());
        layout.addView(answer);
        Button decline = new Button(this);
        decline.setText("Decline");
        decline.setMinHeight(Math.round(getResources().getDisplayMetrics().density * 48));
        decline.setOnClickListener(view -> decline());
        layout.addView(decline);
        setContentView(layout);
        // The CallStyle notification Answer action is a direct user gesture;
        // merely displaying a full-screen call must never acquire the microphone.
        if (getIntent().getBooleanExtra("answer_from_notification", false)) answer();
    }

    private boolean isActive() {
        long now = System.currentTimeMillis() / 1000L;
        return CircleNativeCallPayload.validId(callId)
                && expiry > now && expiry <= now + 30
                && new CircleCallApi(this).enrolled();
    }

    private void answer() {
        if (answering || !isActive()) {
            if (status != null) status.setText("This call is no longer available.");
            return;
        }
        answering = true;
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, 1);
            return;
        }
        startAudio();
    }

    @Override public void onRequestPermissionsResult(int code, String[] permissions, int[] grants) {
        super.onRequestPermissionsResult(code, permissions, grants);
        if (code != 1) return;
        if (grants.length > 0 && grants[0] == PackageManager.PERMISSION_GRANTED) startAudio();
        else {
            answering = false;
            status.setText("Allow microphone access to answer this call.");
        }
    }

    private void startAudio() {
        if (!isActive()) {
            answering = false;
            status.setText("This call has expired.");
            return;
        }
        try {
            Intent service = new Intent(this, CircleAudioService.class)
                    .putExtra("call_id", callId).putExtra("expires_at", expiry);
            if (Build.VERSION.SDK_INT >= 26) startForegroundService(service);
            else startService(service);
            status.setText("Connecting secure audio…");
            CircleIncomingCallNotification.cancel(this, callId);
            finish();
        } catch (RuntimeException ignored) {
            answering = false;
            status.setText("Android could not start the call. Please retry.");
        }
    }

    private void decline() {
        if (!isActive()) { finish(); return; }
        final String id = callId;
        new Thread(() -> {
            try { new CircleCallApi(getApplicationContext()).decline(id); }
            catch (Exception ignored) { }
            runOnUiThread(() -> { CircleIncomingCallNotification.cancel(this, id); finish(); });
        }, "verifyd-circle-decline").start();
    }
}
